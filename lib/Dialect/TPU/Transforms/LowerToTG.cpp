//===- LowerToTG.cpp - lower to tg ----------------------------------------===//
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
// This file lower generic op to tg op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "convert_to_tg"

namespace mlir {

Value* tpu::BroadcastMulOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  assert(this->axis() == 1);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(multiplier());
  // This is a little tricky, as there is no bias() operand to reuse
  // we reuse the quant_rshift() to carry the packed per-channel info
  operands.push_back(quant_rshift());

  std::vector<NamedAttribute> attrs;
  // only do_relu is useful for now
  attrs.push_back(builder.getNamedAttr("param",
      tpu::ConvParam::get(
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(1),
          builder.getStringAttr("VALID"),
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(0), // pd_t
          builder.getI32IntegerAttr(0), // pd_b
          builder.getI32IntegerAttr(0), // pd_l
          builder.getI32IntegerAttr(0), // pd_r
          builder.getI32IntegerAttr(1),
          builder.getBoolAttr(true),    // is_dw
          builder.getBoolAttr(false),   // with_bias
          builder.getBoolAttr(this->do_relu()),   // do_relu
          builder.getContext())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    // somehow, existing backend implementation is using per-channel mode
    // to do a per-tensor operation. which means, it needs to copy 1 rshift
    // value to a oc sized vector, so does the 1 multiplier value, then pack
    // these two tensors into one as if this is a per-channel multiplier mode
    // convolution.
    // TODO: the right way maybe doing a `REAL` per-channel multiplier mode
    // convolution. to put the scale tensor as multiplier rather than filter
    // and the multiplier is by nature per-channel.
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
    assert(!isTensorNone(quant_rshift()));
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_BroadcastMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    assert(isTensorNone(quant_rshift()));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_BroadcastMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::ConcatOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));

  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       wTF);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ConcatOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ConcatOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::Conv2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_Conv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
     return newOp.getResult();
    } else {
      // per-tensor, rshift only mode
      assert(getOpQuantParamType() == "RSHIFT_ONLY");
      assert( !isTensorNone(quant_rshift()) );
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("pt_rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PT_Conv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_Conv2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::CropOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  const unsigned nInputs = op->getNumOperands();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("crop_shape", crop_shapeAttr()));
  attrs.push_back(builder.getNamedAttr("crop_offset", crop_offsetAttr()));

  if (getOpQuant() == "INT8") {
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_CropOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_CropOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::DeConv2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_DeConv2DOp>(
          op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
     return newOp.getResult();
    } else {
      // per-tensor, rshift only mode
      assert(getOpQuantParamType() == "RSHIFT_ONLY");
      assert( !isTensorNone(quant_rshift()) );
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("pt_rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PT_DeConv2DOp>(
          op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_DeConv2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::EltwiseAddOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu",
      builder.getBoolAttr(do_relu())));
  if (do_early_stride()) {
    attrs.push_back(builder.getNamedAttr("do_early_stride",
        builder.getBoolAttr(do_early_stride())));
    attrs.push_back(builder.getNamedAttr("early_stride_h",
                                         early_stride_hAttr()));
    attrs.push_back(builder.getNamedAttr("early_stride_w",
                                         early_stride_wAttr()));
  }

  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       wTF);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseAddOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseAddOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::EltwiseMaxOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu",  builder.getBoolAttr(do_relu())));

  if (do_early_stride()) {
    attrs.push_back(builder.getNamedAttr("do_early_stride",
        builder.getBoolAttr(do_early_stride())));
    attrs.push_back(builder.getNamedAttr("early_stride_h",
                                         early_stride_hAttr()));
    attrs.push_back(builder.getNamedAttr("early_stride_w",
                                         early_stride_wAttr()));
  }
  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // MAX
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       wTF);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMaxOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMaxOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::EltwiseMinOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));

  if (do_early_stride()) {
    attrs.push_back(builder.getNamedAttr("do_early_stride",
        builder.getBoolAttr(do_early_stride())));
    attrs.push_back(builder.getNamedAttr("early_stride_h",
                                         early_stride_hAttr()));
    attrs.push_back(builder.getNamedAttr("early_stride_w",
                                         early_stride_wAttr()));
  }
  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // MIN
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       wTF);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMinOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMinOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::EltwiseMulOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
    // MUL
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    // m_i8_output
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                     wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i32_output",
        builder.getI32IntegerAttr(static_cast<int32_t>(multiplier->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::FullyConnectedOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("do_relu",
      builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_ONLY");
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_FullyConnectedOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_FullyConnectedOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::LrnOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input, sqr table, power table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("local_size", local_sizeAttr()));
  attrs.push_back(builder.getNamedAttr("sum_rshift", sum_rshiftAttr()));
  attrs.push_back(builder.getNamedAttr("lrn_rshift", lrn_rshiftAttr()));
  attrs.push_back(builder.getNamedAttr("quant_data0", quant_data0Attr()));
  attrs.push_back(builder.getNamedAttr("quant_data1", quant_data1Attr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LrnOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::LeakyReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(op->getOperand(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("negative_slope", negative_slopeAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

    auto rshift_pos     = readAndDeleteWeightTensor<float>(
                              quant_pos_rshift(), wTF);
    auto multiplier_pos = readAndDeleteWeightTensor<float>(
                              quant_pos_multiplier(), wTF);
    auto rshift_neg     = readAndDeleteWeightTensor<float>(
                              quant_neg_rshift(), wTF);
    auto multiplier_neg = readAndDeleteWeightTensor<float>(
                              quant_neg_multiplier(), wTF);

    bool do_pos_scale = (multiplier_pos->at(0) != 0.0) ? true : false;

    if (do_pos_scale) {
      LLVM_DEBUG(llvm::errs() << "    do_pos_scale\n";);
      attrs.push_back(builder.getNamedAttr("rshift_pos",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift_pos->at(0)))));
      attrs.push_back(builder.getNamedAttr("m_i8_pos",
          builder.getI8IntegerAttr(static_cast<int8_t>(multiplier_pos->at(0)))));
    } else {
      LLVM_DEBUG(llvm::errs() << "    NO pos_scale\n";);
    }
    attrs.push_back(builder.getNamedAttr("rshift_neg",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift_neg->at(0)))));
    attrs.push_back(builder.getNamedAttr("m_i8_neg",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier_neg->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LeakyReluOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LeakyReluOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }

  llvm_unreachable("unsupported type");
}

Value* tpu::PermuteOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("order0", order0Attr()));
  attrs.push_back(builder.getNamedAttr("order1", order1Attr()));
  attrs.push_back(builder.getNamedAttr("order2", order2Attr()));
  attrs.push_back(builder.getNamedAttr("order3", order3Attr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PermuteOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PermuteOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::PoolAvg2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

    assert( !isTensorNone(quant_rshift()) );
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolAvg2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolAvg2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::PoolMax2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolMax2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolMax2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::PowerOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //Operation *op = this->getOperation();
  //auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

  llvm_unreachable("unsupported type");
}

Value *tpu::PReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(getOperand(0));
  operands.push_back(getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    auto rshift_pos =
        readAndDeleteWeightTensor<float>(quant_pos_rshift(), wTF);
    assert(rshift_pos->size() == 1);
    attrs.push_back(builder.getNamedAttr(
        "rshift_pos",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift_pos->at(0)))));
    auto multiplier_pos =
        readAndDeleteWeightTensor<float>(quant_pos_multiplier(), wTF);
    assert(multiplier_pos->size() == 1);
    attrs.push_back(builder.getNamedAttr(
        "m_i8_pos",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier_pos->at(0)))));
    auto rshift_neg =
        readAndDeleteWeightTensor<float>(quant_neg_rshift(), wTF);
    attrs.push_back(builder.getNamedAttr(
        "rshift_neg",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift_neg->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PReluOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PReluOp>(

        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::QuantOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("from", fromAttr()));
  attrs.push_back(builder.getNamedAttr("to", toAttr()));
  attrs.push_back(builder.getNamedAttr("threshold", thresholdAttr()));

  if (this->from() == "INT8" && this->to() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_QuantOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (this->from() == "BF16" && this->to() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_QuantOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else {
    std::vector<NamedAttribute> param;
    param.push_back(builder.getNamedAttr("from", fromAttr()));
    param.push_back(builder.getNamedAttr("to", toAttr()));
    param.push_back(builder.getNamedAttr("threshold", thresholdAttr()));
    auto paramAttr = builder.getDictionaryAttr(param);
    auto operationAttr = builder.getStringAttr(getOperationName());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));
    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
}

Value *tpu::ReciprocalOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::ReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  //TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    // no need to quant
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReluOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReluOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::ReorgOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("stride", strideAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReorgOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReorgOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::ShuffleChannelOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("group", groupAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ShuffleChannelOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ShuffleChannelOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::SwapChannelOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("channel_order", channel_orderAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SwapChannelOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SwapChannelOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::PixelShuffleOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("upscale_factor", upscale_factorAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PixelShuffleOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PixelShuffleOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::ClipOp::convertToTG() {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("min", minAttr()));
  attrs.push_back(builder.getNamedAttr("max", maxAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ClipOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ClipOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::SigmoidOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  int nInputs = 3; // input and table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::SliceOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axis",
      builder.getI32IntegerAttr(axis().getLimitedValue())));
  attrs.push_back(builder.getNamedAttr("offset", offsetAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SliceOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SliceOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value *tpu::SqrtOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  int nInputs = 3; // input and table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::TanHOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value* tpu::UpsampleOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("scale", scaleAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_UpsampleOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_UpsampleOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

template<typename OpTy>
struct DefaultToTGPattern : public RewritePattern {
  DefaultToTGPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op);
    if (!tpuOp) {
      return matchFailure();
    }
    auto newValue = tpuOp.convertToTG();
    if (!newValue) {
      return matchFailure();
    }
    rewriter.replaceOp(op, {newValue});
    return matchSuccess();
  }
};

template<typename OpTy>
struct DefaultErasePattern : public RewritePattern {
  DefaultErasePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {op->getOperand(0)});
    return matchSuccess();
  }
};

template<typename OpTy>
struct FoldReshapePattern : public RewritePattern {
  FoldReshapePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto laterReshapeOp = cast<OpTy>(op);

    auto formerOp = laterReshapeOp.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<OpTy>())) {
      return matchFailure();
    }
    auto formerScaleOp = cast<OpTy>(formerOp);

    laterReshapeOp.getOperation()->setOperand(0, formerScaleOp.getOperand());
    return matchSuccess();
  }
};

static std::unique_ptr<std::vector<uint8_t> > packWeight(
    std::vector<float> *bias, std::vector<float> *rshift,
    std::vector<float> *multiplier, int64_t oc,
    std::vector<int64_t> &shape) {
  if (bias)
    assert(bias->size() == (size_t)oc);
  assert(rshift->size() == (size_t)oc);
  assert(multiplier->size() == (size_t)oc);

  int64_t isz = bias ? 9 : 5;
  shape = std::vector<int64_t>{oc, 1, isz};

  auto packed = std::make_unique<std::vector<uint8_t> >(oc * isz);

  uint8_t *ptr = packed->data();
  for (int i = 0; i < oc; i++) {
    if (bias) {
      uint32_t val = (uint32_t)(*bias)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint32_t val = (uint32_t)(*multiplier)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint8_t val = (uint8_t)(*rshift)[i];
      *ptr = (uint8_t)val;
      ptr++;
    }
  }

  return std::move(packed);
}

template <typename OpTy>
struct PackWeightConv2DOpPattern : public RewritePattern {
  PackWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    if (getOpQuant(op) != "INT8" || !isOpQuantPerchannel(op)
        || getOpQuantParamType(op) != "RSHIFT_AND_M_I32") {
      // for perchannel multiplier mode only
      return matchFailure();
    }
    if ( !isTensorNone(convOp.bias()) ) {
      auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
      if (biasOp.lowered()) {
        // packed already
        return matchFailure();
      }
    }
    assert( !isTensorNone(convOp.quant_rshift()) );
    assert( !isTensorNone(convOp.quant_multiplier()) );
    LLVM_DEBUG(llvm::errs() << "Pack Weight for Conv2D: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // get param
    auto filter_type = convOp.filter()->getType().template cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t oc;
    auto g = convOp.param().group().getValue().getLimitedValue();
    if (g != 1) {
      assert(filter_shape.size() == 5);
      oc = filter_shape[0] * filter_shape[1];
    } else {
      assert(filter_shape.size() == 4);
      oc = filter_shape[0];
    }

    // get tensor
    std::unique_ptr<std::vector<float> > bias = nullptr;
    if ( !isTensorNone(convOp.bias()) ) {
      bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
    }
    auto rshift = readAndDeleteWeightTensor<float>(convOp.quant_rshift(), wTF);
    auto multiplier = readAndDeleteWeightTensor<float>(convOp.quant_multiplier(), wTF);

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(bias.get(), rshift.get(), multiplier.get(), oc,
                             packedShape);

    // store to the packed per_channel operand in "UINT8"
    if (bias) {
      addWeightTensorAndUpdateWeightOp<uint8_t>(convOp.bias(),
          "pack", *packed, packedShape, "UINT8", wTF);
    } else {
      auto packed_op = addWeightTensorAndCreateWeightOp<uint8_t>(
          op, "pack", *packed, packedShape, "UINT8",
          wTF, wfV);
      convOp.setOperand(2, packed_op);
    }
    auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
    biasOp.setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_rshift and quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    convOp.setOperand(5, NoneOp);
    convOp.setOperand(6, NoneOp);

    return matchSuccess();
  }
};

// somehow, existing backend implementation is using per-channel mode
// to do a per-tensor operation. which means, it needs to copy 1 rshift
// value to a oc sized vector, so does the 1 multiplier value, then pack
// these two tensors into one as if this is a per-channel multiplier mode
// convolution.
// TODO: the right way maybe doing a `REAL` per-channel multiplier mode
// convolution. to put the scale tensor as multiplier rather than filter
// and the multiplier is by nature per-channel.
struct PackWeightBroadcastMulOpPattern : public RewritePattern {
  PackWeightBroadcastMulOpPattern(MLIRContext *context)
      : RewritePattern(tpu::BroadcastMulOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::BroadcastMulOp>(op);

    // Only int8 need pack
    if (getOpQuant(op) == "BF16") {
      LLVM_DEBUG(llvm::errs() << "Pack Weight for BroadcastMul ONLY apply INT8 we skip it\n";);
      return matchFailure();
    }

    // after quantizeInt8, the quantparam is "RSHIFT_AND_M_I32"
    auto rshiftOp = cast<tpu::LoadWeightOp>(castOp.quant_rshift()->getDefiningOp());
    if (rshiftOp.lowered()) {
      // packed already
      return matchFailure();
    }
    assert(getOpQuantParamType(op) == "RSHIFT_AND_M_I32");
    assert( !isTensorNone(castOp.quant_rshift()) );
    assert( !isTensorNone(castOp.quant_multiplier()) );
    LLVM_DEBUG(llvm::errs() << "Pack Weight for BroadcastMul: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    // get param
    int64_t oc = getTensorSize(castOp.multiplier());

    // get tensor
    std::unique_ptr<std::vector<float> > pc_info = nullptr;
    auto rshift = readAndDeleteWeightTensor<float>(castOp.quant_rshift(), wTF);
    auto multiplier = readAndDeleteWeightTensor<float>(castOp.quant_multiplier(), wTF);

    // expand
    auto rshift_perchannel = std::make_unique<std::vector<float>>(oc, rshift->at(0));
    auto multiplier_perchannel =
          std::make_unique<std::vector<float>>(oc, multiplier->at(0));

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(nullptr, rshift_perchannel.get(),
        multiplier_perchannel.get(), oc, packedShape);

    // this is tricky, as where is no bias() to reuse, use quant_rshift() instead
    // store to the packed per_channel operand in "UINT8"
    addWeightTensorAndUpdateWeightOp<uint8_t>(castOp.quant_rshift(),
        "pack", *packed, packedShape, "UINT8", wTF);
    rshiftOp.setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    castOp.setOperand(5, NoneOp);

    setOpQuantParamType(op, "RSHIFT_AND_M_I32");
    return matchSuccess();
  }
};

template<typename T>
static void transposeConvolutionFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  int64_t oc, ic, ks;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    ks = s[2] * s[3];
  } else if (s.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = s[0] * s[1];
    ic = s[2];
    ks = s[3] * s[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }

  std::vector<T> w_t(w.size());
  if (ks == 1 || ic == 1) {
    return;
  } else {
    // for other conv, transpose ic <-> kh*kw
    for (int64_t i = 0; i < oc; i++) {
      for (int64_t j = 0; j < ic; j++) {
        for (int64_t k = 0; k < ks; k++) {
          w_t[i * ic * ks + k * ic + j] = w[i * ic * ks + j * ks + k];
        }
      }
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

template<typename T>
static void transposeFullyConnectedFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  assert(s.size() == 2);
  int row = s[0];
  int col = s[1];
  std::vector<T> w_t(w.size());
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = w[i * col  + j];
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

static void transposeBiasInt16(std::vector<int16_t> &w_int16) {
  int8_t *ptr = reinterpret_cast<int8_t *>(w_int16.data());
  std::vector<int8_t> w(ptr, ptr + w_int16.size() * sizeof(int16_t));
  std::vector<int8_t> w_t(w.size());
  for (size_t i = 0; i < w_int16.size(); i++) {
    for (size_t j = 0; j < 2; j++) {
      w_t[j * w_int16.size() + i] = w[i * 2 + j];
    }
  }
  memcpy(ptr, w_t.data(), w_t.size());
}

template <typename OpTy>
struct LowerWeightConv2DOpPattern : public RewritePattern {
  LowerWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(convOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for Conv2D: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        if (isOpQuantPerchannel(op)
            && getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
          // lowered already, in pack
          assert(biasOp.lowered());
          assert(biasOp.storage() == "UINT8");
        } else if (isOpQuantPerchannel(op)) {
          // per-channel mode, bias is INT32
          assert(biasOp.storage() == "INT32");
          llvm_unreachable("REMINDER: NOT sure if per-channel bias"
                           "needs transpose");
          // TODO:

          // save it
          //StringRef storageType = "INT32";
          //addWeightTensorAndUpdateWeightOp<int32_t>(convOp.bias(),
          //    "lowered", bias_int16, shape, storageType, wTF);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        } else {
          // per-tensor mode, bias is INT16
          assert(biasOp.storage() == "INT16");
          std::vector<int64_t> shape;
          int64_t size;
          getTensorShapeAndSize(convOp.bias(), shape, size);
          auto bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
          std::vector<int16_t> bias_int16(bias->begin(), bias->end());
          transposeBiasInt16(bias_int16);
          std::vector<uint16_t> bias_uint16(size);
          memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

          // save it
          // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
          // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
          // to change the shape.
          addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.bias(),
              "lowered", bias_uint16, shape, "UINT16", wTF);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        }
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(convOp.filter(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());

        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(convOp.bias(), wTF);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(convOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }
};

struct LowerWeightFullyConnectedOpPattern : public RewritePattern {
  LowerWeightFullyConnectedOpPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(fcOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for FullyConnectedOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose k,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(fcOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        // per-tensor mode, bias is INT16
        assert(biasOp.storage() == "INT16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
        std::vector<int16_t> bias_int16(bias->begin(), bias->end());
        transposeBiasInt16(bias_int16);
        std::vector<uint16_t> bias_uint16(size);
        memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

        // save it
        // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
        // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
        // to change the shape.
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.bias(),
            "lowered", bias_uint16, shape, "UINT16", wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(fcOp.filter(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(fcOp.bias(), wTF);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(fcOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }
};

struct LowerWeightPReluOpPattern : public RewritePattern {
  LowerWeightPReluOpPattern(MLIRContext *context)
      : RewritePattern("tpu.prelu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto prOp = cast<tpu::PReluOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(prOp.getOperand(1)->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for PReluOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(filterOp, shape, size);
        auto filter = readAndDeleteWeightTensor<float>(prOp.filter(), wTF);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(prOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        llvm_unreachable("TODO BF16");
      }
    }
    return matchSuccess();
  }
};

struct LowerWeightDetectionOutputOpPattern : public RewritePattern {
  LowerWeightDetectionOutputOpPattern(MLIRContext *context)
      : RewritePattern("tpu.detectionoutput", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto prOp = cast<tpu::DetectionOutputOp>(op);
    auto weightOp = cast<tpu::LoadWeightOp>(prOp.getOperand(2)->getDefiningOp());
    assert(weightOp);
    if (weightOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for DetectionOutputOp: "
                            << getOpName(op) << "\n";);
    weightOp.setAttr("lowered", rewriter.getBoolAttr(true));
    return matchSuccess();
  }
};

struct LowerWeightLrnOpPattern : public RewritePattern {
  LowerWeightLrnOpPattern(MLIRContext *context)
      : RewritePattern("tpu.lrn", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lrnOp = cast<tpu::LrnOp>(op);
    assert(getOpQuant(op) == "INT8" && "only support int8 now");
    auto sqTableOp =
        cast<tpu::LoadWeightOp>(lrnOp.getOperand(1)->getDefiningOp());
    auto powerTableOp =
        cast<tpu::LoadWeightOp>(lrnOp.getOperand(2)->getDefiningOp());
    if (sqTableOp.lowered() && powerTableOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    assert(sqTableOp.storage() == "UINT8");
    assert(powerTableOp.storage() == "UINT8");
    assert(sqTableOp.lowered() == false && powerTableOp.lowered() == false);

    TensorFile *wTF = getWeightTensorFile(op);

    std::vector<int64_t> shape;
    int64_t size;
    // update sq table
    getTensorShapeAndSize(sqTableOp, shape, size);
    auto sqTable = readAndDeleteWeightTensor<float>(sqTableOp, wTF);
    std::vector<uint8_t> sqTable_uint8(sqTable->begin(), sqTable->end());
    addWeightTensorAndUpdateWeightOp<uint8_t>(sqTableOp, "lowered", sqTable_uint8,
                                             shape, "UINT8", wTF);
    sqTableOp.setAttr("lowered", rewriter.getBoolAttr(true));
    // update powerTableOp
    getTensorShapeAndSize(powerTableOp, shape, size);
    auto powerTable = readAndDeleteWeightTensor<float>(powerTableOp, wTF);
    std::vector<uint8_t> powerTable_uint8(powerTable->begin(), powerTable->end());
    addWeightTensorAndUpdateWeightOp<uint8_t>(
        powerTableOp, "lowered", powerTable_uint8, shape, "UINT8", wTF);
    powerTableOp.setAttr("lowered", rewriter.getBoolAttr(true));
    return matchSuccess();
  }
};

template <typename OpTy>
struct LowerWeightLutOpPattern : public RewritePattern {
  LowerWeightLutOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lutOp = cast<OpTy>(op);

    auto tableOp = cast<tpu::LoadWeightOp>(lutOp.getOperand(1)->getDefiningOp());
    auto table_mantissaOp = cast<tpu::LoadWeightOp>(lutOp.getOperand(2)->getDefiningOp());


    if (tableOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for lutOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
        assert(tableOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lutOp.table(), shape, size);
        auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
        auto table_mantissa = readAndDeleteWeightTensor<float>(table_mantissaOp, wTF);
        std::vector<int8_t> table_int8(table->begin(), table->end());
        std::vector<int8_t> table_mantissa_int8(table_mantissa->begin(), table_mantissa->end());
        // 1880 support 256 lookup table
        // because of 1880 hardware search table only on each local memory
        // we dupicate table to limit number <32>
        assert(shape[2] * shape[3] == 256);
        assert(shape[1] == 32);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(
            tableOp, "lowered", table_int8, shape, "INT8", wTF);
        tableOp.setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<int8_t>(
            table_mantissaOp, "lowered", table_mantissa_int8, shape, "INT8", wTF);
        table_mantissaOp.setAttr("lowered", rewriter.getBoolAttr(true));

    } else if (getOpQuant(op) == "BF16") {
      // lower filter
        assert(tableOp.storage() == "BF16");
        assert(table_mantissaOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lutOp.table(), shape, size);
        auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
        auto table_mantissa = readAndDeleteWeightTensor<float>(table_mantissaOp,
                                                               wTF);
        std::vector<uint16_t> table_uint16(table->begin(), table->end());
        std::vector<uint16_t> table_mantissa_uint16(table_mantissa->begin(),
                                                    table_mantissa->end());
        // 1880 support 256 lookup table
        // because of 1880 hardware search table only on each local memory
        // we dupicate table to limit number <32>
        assert(shape[2] * shape[3] == 256);
        assert(shape[1] == 32);

        // save it
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            tableOp, "lowered", table_uint16, shape, "UINT16", wTF);
        tableOp.setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            table_mantissaOp, "lowered", table_mantissa_uint16,
            shape, "UINT16", wTF);
        table_mantissaOp.setAttr("lowered", rewriter.getBoolAttr(true));
    }

    return matchSuccess();
  }
};

template <typename OpTy>
struct LowerCpuOpDefaultPattern : public RewritePattern {
  LowerCpuOpDefaultPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs() << "Lower Cpu Op " << castOp.getOperationName() << ":"
                            << getOpName(castOp)<< "\n";);

    auto builder = Builder(op->getContext());
    std::vector<NamedAttribute> param;
    std::vector<NamedAttribute> attrs;
    for (auto& attr : op->getAttrs()) {
      if (attr.first == "name"
         || attr.first == "layer_id"
         || attr.first == "gaddr") {
        continue;
      }
      param.push_back(attr);
    }
    auto operationAttr = builder.getStringAttr(castOp.getOperationName());
    auto paramAttr = builder.getDictionaryAttr(param);

    attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));
    if (castOp.layer_id().hasValue()) {
      int32_t layer_id = castOp.layer_id().getValue().getSExtValue();
      attrs.push_back(builder.getNamedAttr("layer_id",
          builder.getI32IntegerAttr(layer_id)));
    }

    std::vector<Value *> operands(op->getOperands().begin(),
                                  op->getOperands().end());

    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
        castOp.getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    auto result = newOp.getResult();
    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

template <typename OpTy>
struct LowerCustomOpPattern : public RewritePattern {
  LowerCustomOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs() << "Lower Custom Op " << castOp.getOperationName() << ":"
                            << getOpName(castOp)<< "\n";);
    auto builder = Builder(op->getContext());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", castOp.operation_nameAttr()));
    attrs.push_back(builder.getNamedAttr("param", castOp.paramAttr()));
    attrs.push_back(builder.getNamedAttr("layer_id", castOp.layer_idAttr()));

    std::vector<Value *> operands(op->getOperands().begin(),
                                  op->getOperands().end());

    if (castOp.tpu()) {
      if (castOp.getOpQuant() == "INT8") {
        auto newOp = OpBuilder(op).create<tpu::TG_INT8_GenericTpuOp>(
            op->getLoc(), castOp.getResult()->getType(), ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});
        auto result = newOp.getResult();
        rewriter.replaceOp(op, {result});
      } else if (castOp.getOpQuant() == "BF16") {
        auto newOp = OpBuilder(op).create<tpu::TG_BF16_GenericTpuOp>(
            op->getLoc(), castOp.getResult()->getType(), ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});
        auto result = newOp.getResult();
        rewriter.replaceOp(op, {result});
      } else {
        llvm_unreachable("unsupported type");
      }
    } else {
      auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
          castOp.getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
      auto result = newOp.getResult();
      rewriter.replaceOp(op, {result});
    }

    return matchSuccess();
  }
};

class TpuLowerPass : public FunctionPass<TpuLowerPass> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto fn = getFunction();

    // first, merge conv rshift/multiplier/bias into one packed tensor
    OwningRewritePatternList patterns_pack;
    patterns_pack.insert<
        PackWeightConv2DOpPattern<tpu::Conv2DOp>,
        PackWeightConv2DOpPattern<tpu::DeConv2DOp>,
        PackWeightBroadcastMulOpPattern
        >(context);
    applyPatternsGreedily(fn, patterns_pack);

    // second, do weight lower on weight tensors
    // lower means transpose and save as storageType (int8/bf16,etc)
    OwningRewritePatternList patterns_lower;
    patterns_lower.insert<
        LowerWeightConv2DOpPattern<tpu::Conv2DOp>,
        LowerWeightConv2DOpPattern<tpu::DeConv2DOp>,
        LowerWeightLrnOpPattern,
        LowerWeightLutOpPattern<tpu::ReciprocalOp>,
        LowerWeightPReluOpPattern,
        LowerWeightLutOpPattern<tpu::SigmoidOp>,
        LowerWeightLutOpPattern<tpu::SqrtOp>,
        LowerWeightLutOpPattern<tpu::TanHOp>,
        LowerWeightFullyConnectedOpPattern,
        LowerWeightDetectionOutputOpPattern
        >(context);
    applyPatternsGreedily(fn, patterns_lower);

    // do cpu op lowering
    OwningRewritePatternList patterns_cpuop;
    patterns_cpuop.insert<
        LowerCpuOpDefaultPattern<tpu::DetectionOutputOp>,
        LowerCpuOpDefaultPattern<tpu::PreprocessOp>,
        LowerCpuOpDefaultPattern<tpu::RetinaFaceDetectionOp>,
        LowerCpuOpDefaultPattern<tpu::SoftmaxOp>,
        LowerCpuOpDefaultPattern<tpu::TransposeOp>,
        LowerCpuOpDefaultPattern<tpu::YoloDetectionOp>,
        LowerCustomOpPattern<tpu::CustomOp>
        >(context);
    applyPatternsGreedily(fn, patterns_cpuop);

    // do op lower
    OwningRewritePatternList patterns;
    patterns.insert<
        DefaultToTGPattern<tpu::BroadcastMulOp>,
        DefaultToTGPattern<tpu::ClipOp>,
        DefaultToTGPattern<tpu::ConcatOp>,
        DefaultToTGPattern<tpu::Conv2DOp>,
        DefaultToTGPattern<tpu::CropOp>,
        DefaultToTGPattern<tpu::DeConv2DOp>,
        DefaultToTGPattern<tpu::ReciprocalOp>,
        DefaultToTGPattern<tpu::EltwiseAddOp>,
        DefaultToTGPattern<tpu::EltwiseMaxOp>,
        DefaultToTGPattern<tpu::EltwiseMinOp>,
        DefaultToTGPattern<tpu::EltwiseMulOp>,
        DefaultToTGPattern<tpu::FullyConnectedOp>,
        DefaultToTGPattern<tpu::LrnOp>,
        DefaultToTGPattern<tpu::LeakyReluOp>,
        DefaultToTGPattern<tpu::PermuteOp>,
        DefaultToTGPattern<tpu::PixelShuffleOp>,
        DefaultToTGPattern<tpu::PoolAvg2DOp>,
        DefaultToTGPattern<tpu::PoolMax2DOp>,
        DefaultToTGPattern<tpu::PReluOp>,
        DefaultToTGPattern<tpu::QuantOp>,
        DefaultToTGPattern<tpu::ReluOp>,
        DefaultToTGPattern<tpu::ReorgOp>,
        DefaultToTGPattern<tpu::ShuffleChannelOp>,
        DefaultToTGPattern<tpu::SigmoidOp>,
        DefaultToTGPattern<tpu::SliceOp>,
        DefaultToTGPattern<tpu::SqrtOp>,
        DefaultToTGPattern<tpu::SwapChannelOp>,
        DefaultToTGPattern<tpu::TanHOp>,
        DefaultToTGPattern<tpu::UpsampleOp>
        >(context);
    applyPatternsGreedily(fn, patterns);

    // check if every one is not lowered
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)
          || isa<tpu::InputOp>(op)
          || isa<tpu::GenericCpuOp>(op)) {
        // no need to lower
      } else if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op)) {
        llvm::errs() << "didn't lower " << op->getName() << "\n";
        assert(false);
      } else if (auto tgOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
        // lowered already
      } else {
        std::string opName = op->getName().getStringRef();
        llvm_unreachable(("lower didn't handle " + opName).c_str());
      }
    });

    // TODO: this is temporary
    // fold reshape
    patterns.clear();
    patterns.insert<
        FoldReshapePattern<tpu::ReshapeOp>
        >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createTpuLowerPass() {
  return std::make_unique<TpuLowerPass>();
}

static PassRegistration<TpuLowerPass>
    pass("tpu-lower",
         "Lower TPU Dialect from TPU Ops to TPU_TG Ops");

} // namespace mlir
