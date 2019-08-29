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
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    llvm::errs() << "run " << function.getName() << "\n";

    //if (failed(convertOneFunction(function)))
    //  return failure();
  }

  return success();
}

LogicalResult runTpuModule(ModuleOp m) {
  return ModuleInterpreter::runModule<>(m);
}

} // namespace mlir
