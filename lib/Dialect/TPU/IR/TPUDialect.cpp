//===- TPUDialect.cpp - MLIR Dialect for TPU implementation -------===//
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
// This file implements the TPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::tpu;

StringRef TPUDialect::getDialectName() { return "tpu"; }

TPUDialect::TPUDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<
    //LaunchOp, LaunchFuncOp,
#define GET_OP_LIST
#include "mlir/Dialect/TPU/TPUOps.cpp.inc"
                >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/TPU/TPUOps.cpp.inc"
