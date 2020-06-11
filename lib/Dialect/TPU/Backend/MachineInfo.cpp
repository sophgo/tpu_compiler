//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/TPU/MachineInfo.h"

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "run-chip"

using namespace mlir;

namespace mlir{

uint32_t MInfo::version{0};
uint32_t MInfo::eu_num{0};
uint32_t MInfo::lane_num{0};
uint64_t MInfo::lmem_per_lane{0};

void MInfo::runOnFunction() {
}
void MInfo::getChipInfo(const char *name) {

  std::vector<int8_t> weight_data;
  CviBackendContext *backend_ctx = nullptr;

  backend_ctx = cvi_backend_create_context_chip(weight_data, name);
  version = cvi_backend_chip_context(backend_ctx, CVI_CHIP_VERSION);
  eu_num = cvi_backend_chip_context(backend_ctx, CVI_CHIP_EU_NUM);
  lane_num = cvi_backend_chip_context(backend_ctx, CVI_CHIP_LANE_NUM);
  lmem_per_lane = cvi_backend_chip_context(backend_ctx, CVI_CHIP_LMEM_SIZE);

  LLVM_DEBUG(llvm::errs()<<" version = " << version << "\n";);
  LLVM_DEBUG(llvm::errs()<<" eu = " << eu_num << "\n";);
  LLVM_DEBUG(llvm::errs()<<" lane = " << lane_num << "\n";);
  LLVM_DEBUG(llvm::errs()<<" lane size= " << lmem_per_lane << "\n";);
}
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createMInfo() {
  return std::make_unique<MInfo>();
}

static PassRegistration<MInfo>
    pass("set-chip",
         "Apply chip type.");
