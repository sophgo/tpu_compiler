//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"
#include "cvibuilder/cvimodel_generated.h"
#include "mlir/Dialect/TPU/buildCviModel.h"

#include <fstream>

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

static CviBackendContext *backend_ctx = nullptr;

static LogicalResult runOperation(Operation &opInst) {
  LLVM_DEBUG(llvm::errs() << "  op " << opInst.getName() << "\n";);

  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(opInst)) {
    if (isa<tpu::TG_CallOp>(opInst))
      success();
    return tpuTGOp.codegen((void *)backend_ctx);
  } else if (isa<tpu::TpuTLOpCodegenInterface>(opInst)) {
    auto tpuTLOp = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(opInst);
    // enable parallel
    if (tpuTLOp.getEnableParallel() && !isa<tpu::TL_LG_LrnOp>(opInst))
      cvi_backend_parallel_enable(backend_ctx);
    // tl codegen
    tpuTLOp.codegen((void *)backend_ctx);
    // disable parallel
    if (tpuTLOp.getDisableParallel() && !isa<tpu::TL_LG_LrnOp>(opInst))
      cvi_backend_parallel_disable(backend_ctx);
  }

  return success();
}

static LogicalResult runBlock(Block &bb) {
  // Traverse operations.
  for (auto &op : bb) {
    if (failed(runOperation(op)))
      return failure();
  }

  return success();
}

static LogicalResult runOneFunction(FuncOp func) {
  LLVM_DEBUG(llvm::errs() << "func " << func.getName() << "\n";);

  // Then, run blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  return success();
}

LogicalResult translateModule(ModuleOp module, llvm::raw_ostream &output) {
  if (!module)
    return failure();

  std::vector<int8_t> weight_data;
  backend_ctx = cvi_backend_create_context(weight_data);

  for (FuncOp function : module.getOps<FuncOp>()) {
    LLVM_DEBUG(llvm::errs() << "run " << function.getName() << "\n";);

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  cvi_backend_submit(backend_ctx);
  std::vector<uint8_t> cmdbuf;
  cvi_backend_get_cmdbuf(backend_ctx, cmdbuf);

  output.write(reinterpret_cast<char *>(cmdbuf.data()), cmdbuf.size());

  return success();
}

LogicalResult translateModule_cvimodel(ModuleOp module, llvm::raw_ostream &output) {
  if (!module)
   return failure();

  std::vector<int8_t> weight_data;
  backend_ctx = cvi_backend_create_context(weight_data);

  for (FuncOp function : module.getOps<FuncOp>()) {
    LLVM_DEBUG(llvm::errs() << "run " << function.getName() << "\n";);

    if (!function.getName().equals("tpu_func")) {
      continue;
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  cvi_backend_submit(backend_ctx);
  std::vector<uint8_t> cmdbuf;
  cvi_backend_get_cmdbuf(backend_ctx, cmdbuf);
  flatbuffers::FlatBufferBuilder builder(1024);
  CviMlirParser parser(module);
  parser.setCmdBuf(cmdbuf);
  CviModel model(&parser, &builder);
  model.storeModel(output);
  return success();
}

static TranslateFromMLIRRegistration
    registration("mlir-to-cmdbuf",
                 [](ModuleOp module, llvm::raw_ostream &output) {
                   return translateModule(module, output);
                 });

static TranslateFromMLIRRegistration
    registration_cvimodel("mlir-to-cvimodel",
                 [](ModuleOp module, llvm::raw_ostream &output) {
                   return translateModule_cvimodel(module, output);
                 });
