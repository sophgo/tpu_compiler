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

#ifndef MLIR_DIALECT_TPU_WEIGHTFILE_OP_H
#define MLIR_DIALECT_TPU_WEIGHTFILE_OP_H

// WeightFileOp
class WeightFileOpOperandAdaptor {
public:
  WeightFileOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);

private:
  ArrayRef<Value *> tblgen_operands;
};
class WeightFileOp : public Op<WeightFileOp,
    OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using OperandAdaptor = WeightFileOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  Value *weight();
  StringAttr filenameAttr();
  StringRef filename();
  static void build(Builder *tblgen_builder,
      OperationState &tblgen_state, Type weight, StringAttr filename);
  static void build(Builder *tblgen_builder,
      OperationState &tblgen_state, Type weight, StringRef filename);
  static void build(Builder *tblgen_builder,
      OperationState &tblgen_state, ArrayRef<Type> resultTypes,
      StringAttr filename);
  static void build(Builder *, OperationState &tblgen_state,
      ArrayRef<Type> resultTypes, ValueRange operands,
      ArrayRef<NamedAttribute> attributes);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  TensorFile* get();
};

#endif // MLIR_DIALECT_TPU_WEIGHTFILE_OP_H
