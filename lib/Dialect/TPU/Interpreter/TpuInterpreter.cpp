//===- TpuInterpreter.cpp - Implementation of TPU Op Interpreter ---------===//
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
// This file implements the TPU dialect Interpreter.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {

LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  // #include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
  if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(opInst)) {
    llvm::errs() << "LoadFileOp" << "\n";
    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    llvm::errs() << "LoadWeightOp" << "\n";
    return success();
  }
  if (auto conv2DOp = dyn_cast<tpu::Conv2DOp>(opInst)) {
    llvm::errs() << "Conv2DOp" << "\n";
    //conv2DOp.dump();
    return success();
  }
  if (auto averagePool2DOp = dyn_cast<tpu::AveragePool2DOp>(opInst)) {
    llvm::errs() << "AveragePool2DOp" << "\n";
    return success();
  }
  if (auto maxPool2DOp = dyn_cast<tpu::MaxPool2DOp>(opInst)) {
    llvm::errs() << "MaxPool2DOp" << "\n";
    return success();
  }
  if (auto fullyConnectedOp = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    llvm::errs() << "FullyConnectedOp" << "\n";
    return success();
  }
  if (auto reluOp = dyn_cast<tpu::ReluOp>(opInst)) {
    llvm::errs() << "ReluOp" << "\n";
    return success();
  }
  if (auto batchNormOp = dyn_cast<tpu::BatchNormOp>(opInst)) {
    llvm::errs() << "BatchNormOp" << "\n";
    return success();
  }
  if (auto scaleOp = dyn_cast<tpu::ScaleOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    return success();
  }
  if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(opInst)) {
    llvm::errs() << "ReshapeOp" << "\n";
    return success();
  }

  if (auto returnOp = dyn_cast<ReturnOp>(opInst)) {
    llvm::errs() << "ReturnOp" << "\n";
    return success();
  }

  return opInst.emitError("unsupported operation: ")
         << opInst.getName();
}

LogicalResult ModuleInterpreter::runBlock(Block &bb) {
  // Traverse operations.
  for (auto &op : bb) {
    if (failed(runOperation(op)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runOneFunction(FuncOp func) {
  llvm::errs() << "func " << func.getName() << "\n";
  // Clear the value mappings, it is only relevant within one function.
  valueMapping.clear();
  // Add function arguments to the value remapping table.
  unsigned int argIdx = 0;
  for (auto arg : func.getArguments()) {
    llvm::errs() << "arg " << argIdx << ": ";
    arg->getType().dump();
    llvm::errs() << "\n";

    //BlockArgument *mlirArg = arg;

    //valueMapping[mlirArg] = &input_tensor;
    argIdx++;
  }

  // Then, convert blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    llvm::errs() << "run " << function.getName() << "\n";

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult runTpuModule(ModuleOp m) {
  return ModuleInterpreter::runModule<>(m);
}

} // namespace mlir
