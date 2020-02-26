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
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "weightfile"

using namespace mlir;
using namespace mlir::tpu;

WeightFileOpOperandAdaptor::WeightFileOpOperandAdaptor(
    ArrayRef<Value *> values) {
  tblgen_operands = values;
}

ArrayRef<Value *> WeightFileOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index),
          std::next(tblgen_operands.begin(), index + 1)};
}

StringRef WeightFileOp::getOperationName() {
  return "tpu.weight_file";
}

Operation::operand_range WeightFileOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index),
          std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range WeightFileOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index),
          std::next(getOperation()->result_begin(), index + 1)};
}

Value *WeightFileOp::weight() {
  return *getODSResults(0).begin();
}

StringAttr WeightFileOp::filenameAttr() {
  return this->getAttr("filename").cast<StringAttr>();
}

StringRef WeightFileOp::filename() {
  auto attr = filenameAttr();
  return attr.getValue();
}

void WeightFileOp::build(Builder *tblgen_builder,
    OperationState &tblgen_state, Type weight, StringAttr filename) {
  tblgen_state.addAttribute("filename", filename);
  tblgen_state.addTypes(weight);
}

void WeightFileOp::build(Builder *tblgen_builder,
    OperationState &tblgen_state, Type weight, StringRef filename) {
  tblgen_state.addAttribute("filename",
      (*tblgen_builder).getStringAttr("filename"));
  tblgen_state.addTypes(weight);
}

void WeightFileOp::build(Builder *tblgen_builder,
    OperationState &tblgen_state, ArrayRef<Type> resultTypes,
    StringAttr filename) {
  tblgen_state.addAttribute("filename", filename);
  tblgen_state.addTypes(resultTypes);
}

void WeightFileOp::build(Builder *, OperationState &tblgen_state,
    ArrayRef<Type> resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attributes) {
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  tblgen_state.addTypes(resultTypes);
  assert(operands.size() == 0u && "mismatched number of parameters");
  tblgen_state.addOperands(operands);

  tblgen_state.addAttributes(attributes);
}

LogicalResult WeightFileOp::verify() {
  auto tblgen_filename = this->getAttr("filename");
  if (!tblgen_filename)
    return emitOpError("requires attribute 'filename'");
  {
    if (!((tblgen_filename.isa<StringAttr>())))
      return emitOpError("attribute 'filename' failed to satisfy constraint:"
                         " string attribute");
  }
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    for (Value *v : getODSResults(0)) {
      (void)v;
      if (!(((v->getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index
            << " must be memref of any type values, but got " << v->getType();
      }
      ++index;
    }
  }
  if (this->getOperation()->getNumRegions() != 0) {
    return emitOpError("has incorrect number of regions: expected 0 but found ")
      << this->getOperation()->getNumRegions();
  }
  return mlir::success();
}

void WeightFileOp::print(OpAsmPrinter &p) {
  auto *context = getContext();
  auto dialect = context->getRegisteredDialect("tpu");
  auto tpuDialect = reinterpret_cast<tpu::TPUDialect *>(dialect);
  assert(tpuDialect);
  TensorFile *weightFile = (TensorFile *)tpuDialect->getPriv();
  if (weightFile) {
    std::string newName;
    int updated = weightFile->keep(true, &newName);
    if (updated) {
      setAttr("filename", Builder(context).getStringAttr(newName));
      LLVM_DEBUG(llvm::errs() << "WeightFile " << newName << " saved, "
                              << updated << " tensors updated\n";);
    } else {
      LLVM_DEBUG(llvm::errs() << "WeightFile no updated\n";);
    }
  } else {
    LLVM_DEBUG(llvm::errs() << "WeightFile not opened\n";);
  }

  // print the Op with the updated Weight File name
  Op::print(p);
}

TensorFile* WeightFileOp::get(void) {
  auto *context = getContext();
  auto dialect = context->getRegisteredDialect("tpu");
  auto tpuDialect = reinterpret_cast<TPUDialect *>(dialect);
  assert(tpuDialect);
  if (!tpuDialect->getPriv()) {
    LLVM_DEBUG(llvm::errs() << "Open WeightFile " << filename() << "\n";);
    auto weightFile = openTensorFile(filename());
    assert(weightFile);
    tpuDialect->setPriv((void *)weightFile.release());
  } else {
    LLVM_DEBUG(llvm::errs() << "WeightFile already opened\n";);
  }
  return (TensorFile *)tpuDialect->getPriv();
}
