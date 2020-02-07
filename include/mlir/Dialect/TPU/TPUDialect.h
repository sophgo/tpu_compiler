//===- TPUDialect.h - MLIR Dialect for TPU --------------*- C++ -*-===//
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
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_TPUDIALECT_H
#define MLIR_DIALECT_TPU_TPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace tpu {

/// The dialect containing TPU launching operations and related
/// facilities.
class TPUDialect : public Dialect {
public:
  explicit TPUDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tpu"; }
};

//===----------------------------------------------------------------------===//
// TPU OpInterface definitions
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/TPU/TPUInterface.h.inc"

//===----------------------------------------------------------------------===//
// TPU Ops definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir/Dialect/TPU/TPUOps.h.inc"

} // end namespace tpu
} // end namespace mlir

#endif // MLIR_DIALECT_TPU_TPUDIALECT_H
