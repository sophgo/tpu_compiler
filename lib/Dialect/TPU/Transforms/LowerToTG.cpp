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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "convert_to_tg"

llvm::cl::opt<std::string> clInputsType(
    "inputs-type",
    llvm::cl::desc("set result type: AUTO/FP32/INT8/BF16/SAME; if AUTO, use INT8 if first layer is INT8, FP32 if BF16"),
    llvm::cl::init("AUTO"));

llvm::cl::opt<std::string> clOutputsType(
    "outputs-type",
    llvm::cl::desc("set result type: AUTO/FP32/INT8/BF16/SAME; if AUTO, use INT8 if last layer is INT8, FP32 if BF16"),
    llvm::cl::init("FP32"));

namespace mlir {

Value tpu::AbsOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  //TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    // no need to quant
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_AbsOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_AbsOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }

  llvm_unreachable("unsupported type");

}

Value tpu::ArgMaxOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto castOp = cast<tpu::ArgMaxOp>(op);

  auto builder = Builder(op->getContext());

  if (getOpQuant() == "INT8") {
    std::vector<NamedAttribute> attrs;
    int64_t input_size;
    std::vector<int64_t> shape;
    getTensorShapeAndSize(op->getOperand(0), shape, input_size);
    int last_dim = shape.size() - 1;
    int w = shape[last_dim];
    w = (w + 256 - 1) / 256;
    shape[last_dim] = w;
    auto eltType = input().getType().cast<TensorType>().getElementType();
    auto result_type = RankedTensorType::get(shape, eltType);

    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name().str() + "_tpu")));
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ArgMaxOp>(
        op->getLoc(), result_type, ArrayRef<Value>{{input()}},
        ArrayRef<NamedAttribute>{attrs});

    std::vector<NamedAttribute> param;
    for (auto &attr : castOp->getAttrs()) {
      if (attr.first == "name" || attr.first == "gaddr" ||
          attr.first == "quant") {
        continue;
      }
      param.push_back(attr);
    }
    auto paramAttr = builder.getDictionaryAttr(param);
    auto operationAttr = builder.getStringAttr(getOperationName());

    attrs.clear();
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));

    auto cpuOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), castOp.getResult().getType(),
        ArrayRef<Value>{{input(), newOp.getResult()}},
        ArrayRef<NamedAttribute>{attrs});
    return cpuOp.getResult();
  } else if (getOpQuant() == "BF16") {
    std::vector<NamedAttribute> attrs;
    int64_t input_size;
    std::vector<int64_t> shape;
    getTensorShapeAndSize(op->getOperand(0), shape, input_size);
    int last_dim = shape.size() - 1;
    int w = shape[last_dim];
    w = (w + 256 - 1) / 256;
    shape[last_dim] = w;
    auto eltType = input().getType().cast<TensorType>().getElementType();
    auto result_type = RankedTensorType::get(shape, eltType);

    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name().str() + "_tpu")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ArgMaxOp>(
        op->getLoc(), result_type, ArrayRef<Value>{{input()}},
        ArrayRef<NamedAttribute>{attrs});

    std::vector<NamedAttribute> param;
    for (auto &attr : castOp->getAttrs()) {
      if (attr.first == "name" || attr.first == "gaddr" ||
          attr.first == "quant") {
        continue;
      }
      param.push_back(attr);
    }
    auto paramAttr = builder.getDictionaryAttr(param);
    auto operationAttr = builder.getStringAttr(getOperationName());

    attrs.clear();
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));

    auto cpuOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), castOp.getResult().getType(),
        ArrayRef<Value>{{input(), newOp.getResult()}},
        ArrayRef<NamedAttribute>{attrs});
    return cpuOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::BroadcastMulOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  bool align = align_right();
  int64_t n, c, h, w, bn, bc, bh, bw;
  auto shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w, align);
  auto bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw, align);

  if ((bn == n || bn == 1) && bc == c && bh == 1 && bw == 1) {
    // convert to scale op
    std::vector<Value> operands;
    operands.push_back(op->getOperand(0));
    operands.push_back(op->getOperand(1));
    // This is a little tricky, as there is no bias() operand to reuse
    // we reuse the quant_rshift() to carry the packed per-channel info
    operands.push_back(quant_rshift());
    operands.push_back(quant_scale());
    operands.push_back(quant_zeropoint());

    std::vector<NamedAttribute> attrs;
    // only do_relu is useful for now
    attrs.push_back(builder.getNamedAttr(
        "param",
        tpu::ConvParam::get(
            builder.getI32IntegerAttr(1), builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(1), builder.getI32IntegerAttr(1),
            builder.getStringAttr("VALID"), builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(0), // pd_t
            builder.getI32IntegerAttr(0), // pd_b
            builder.getI32IntegerAttr(0), // pd_l
            builder.getI32IntegerAttr(0), // pd_r
            builder.getI32IntegerAttr(1),
            builder.getBoolAttr(true),                      // is_dw
            builder.getBoolAttr(false),                     // with_bias
            builder.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
            builder.getI32IntegerAttr(0),                   // pad_value
            builder.getContext())));
    attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    if (getOpQuant() == "INT8") {
      // somehow, existing backend implementation is using per-channel mode
      // to do a per-tensor operation. which means, it needs to copy 1 rshift
      // value to a oc sized vector, so does the 1 multiplier value, then pack
      // these two tensors into one as if this is a per-channel multiplier mode
      // convolution.
      // TODO: the right way maybe doing a `REAL` per-channel multiplier mode
      // convolution. to put the scale tensor as multiplier rather than filter
      // and the multiplier is by nature per-channel.
      assert(!isTensorNone(quant_rshift()));
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_ScaleOp>(
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    } else if (getOpQuant() == "BF16") {
      assert(isTensorNone(quant_rshift()));
      auto newOp = OpBuilder(op).create<tpu::TG_BF16_ScaleOp>(
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
    llvm_unreachable("unsupported type");
  }
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(op->getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("align_right", builder.getBoolAttr(align)));
  if (getOpQuant() == "INT8") {
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    int8_t rshift_i8 = static_cast<int8_t>(rshift->at(0));
    std::vector<int32_t> m_i8_inputs(2, 1);
    m_i8_inputs[0] = static_cast<int32_t>(multiplier->at(0));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI8IntegerAttr(rshift_i8)));
    attrs.push_back(builder.getNamedAttr(
        "m_i8_inputs",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs}))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_BroadcastMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_BroadcastMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::BroadcastAddOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  uint32_t nInputs = 2;
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(op->getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("align_right", builder.getBoolAttr(align_right())));
  if (getOpQuant() == "INT8") {
    int8_t rshift_i8 = 0;
    std::vector<int32_t> m_i8_inputs(nInputs, 1);
    if (getOpQuantParamType() == "RSHIFT_AND_M_I8") {
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      rshift_i8 = static_cast<int8_t>(rshift->at(0));

      // m_i8_inputs
      auto multiplier =
          readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs[i] = static_cast<int32_t>(multiplier->at(i));
      }
    }
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI8IntegerAttr(rshift_i8)));
    attrs.push_back(builder.getNamedAttr(
        "m_i8_inputs",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs}))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_BroadcastAddOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_BroadcastAddOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::BroadcastSubOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  uint32_t nInputs = 2;
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(op->getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("align_right", builder.getBoolAttr(align_right())));
  if (getOpQuant() == "INT8") {
    int8_t rshift_i8 = 0;
    std::vector<int32_t> m_i8_inputs(nInputs, 1);
    if (getOpQuantParamType() == "RSHIFT_AND_M_I8") {
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      rshift_i8 = static_cast<int8_t>(rshift->at(0));

      // m_i8_inputs
      auto multiplier =
          readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs[i] = static_cast<int32_t>(multiplier->at(i));
      }
    }
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI8IntegerAttr(rshift_i8)));
    attrs.push_back(builder.getNamedAttr(
        "m_i8_inputs",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs}))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_BroadcastSubOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_BroadcastSubOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

static bool is_fused_op(Operation *op) {
  if (isa<tpu::ConcatOp>(op)) {
    return true;
  }
  if (isa<tpu::ReshapeOp>(op)) {
    return true;
  }
  if (isa<tpu::CropOp>(op)) {
    std::vector<int64_t> is_4;
    std::vector<int64_t> os_4;
    std::vector<int> offset_4;
    std::vector<int> step_4;
    bool fusible;
    parseCropParam<tpu::CropOp>(op, is_4, os_4, offset_4, step_4, fusible);
    return fusible;
  }
  return false;
}

Value tpu::ConcatOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  bool relu = do_relu();
  bool only_merge = !relu; // just merge input data
  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (auto input : inputs()) {
    operands.push_back(input);
    if (only_merge == true) {
      if (is_fused_op(input.getDefiningOp())) {
        only_merge = false;
      }
      if (isa<tpu::LoadWeightOp>(input.getDefiningOp())) {
        only_merge = false;
      }
    }
  }
  if (only_merge == true) {
    uint32_t ax = axis();
    auto shape = getTensorShape(op->getResult(0));
    for (uint32_t i = 0; i < ax; i++) {
      if (shape[i] != 1) {
        only_merge = false;
        break;
      }
    }
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    std::vector<int32_t> m_i8_inputs_array(nInputs);
    std::vector<int32_t> m_rshift_array(nInputs);
    for (unsigned i = 0; i < nInputs; ++i) {
      m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      m_rshift_array[i] = static_cast<int32_t>(rshift->at(i));
      if (only_merge == true) {
        if (m_i8_inputs_array[i] != 1 || m_rshift_array[i] != 0) {
          only_merge = false;
        }
      }
    }
    if (only_merge == false) {
      attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
      attrs.push_back(
          builder.getNamedAttr("do_relu", builder.getBoolAttr(relu)));
      attrs.push_back(builder.getNamedAttr(
          "m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
      attrs.push_back(builder.getNamedAttr(
          "rshift",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_rshift_array}))));
      // create op
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_ConcatOp>(
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  } else if (getOpQuant() == "BF16") {
    if (only_merge == false) {
      attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
      attrs.push_back(
          builder.getNamedAttr("do_relu", builder.getBoolAttr(relu)));
      auto newOp = OpBuilder(op).create<tpu::TG_BF16_ConcatOp>(
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  }
  if (only_merge) {
    auto newOp = OpBuilder(op).create<tpu::TG_ConcatNOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::Conv2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  std::vector<Value> operands;

  auto pad_t = param().padding_t().getInt();
  auto pad_b = param().padding_b().getInt();
  auto pad_l = param().padding_l().getInt();
  auto pad_r = param().padding_r().getInt();
  int32_t pad_h_begin = 0, pad_h_end = 0;
  int32_t pad_w_begin = 0, pad_w_end = 0;
  if (pad_t > 15) {
    pad_h_begin = pad_t;
    pad_t = 0;
  }
  if (pad_b > 15) {
    pad_h_end = pad_b;
    pad_b = 0;
  }
  if (pad_l > 15) {
    pad_w_begin = pad_l;
    pad_l = 0;
  }
  if (pad_r > 15) {
    pad_w_end = pad_r;
    pad_r = 0;
  }
  if (pad_h_begin > 0 || pad_h_end > 0 || pad_w_begin > 0 || pad_w_end > 0) {
    std::vector<int32_t> pads = {0, 0, pad_h_begin, pad_w_begin, 0, 0, pad_h_end, pad_w_end};
    std::vector<NamedAttribute> attrs;
    auto inputShape = getTensorShape(input());
    auto type = getResult().getType().template cast<TensorType>();

    std::vector<int64_t> shape(4);
    shape[0] = inputShape[0];
    shape[1] = inputShape[1];
    shape[2] = inputShape[2] + pad_h_begin + pad_h_end;
    shape[3] = inputShape[3] + pad_w_begin + pad_w_end;
    auto resultType = RankedTensorType::get(shape, type.getElementType());

    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name().str() + "_pad")));
    attrs.push_back(builder.getNamedAttr("const_val", builder.getF32FloatAttr(0)));
    attrs.push_back(builder.getNamedAttr("pads", builder.getI32ArrayAttr(ArrayRef<int32_t>({pads}))));
    attrs.push_back(builder.getNamedAttr("mode", builder.getStringAttr("constant")));
    if (getOpQuant() == "INT8") {
      auto padOp = OpBuilder(op).create<tpu::TG_INT8_PadOp>(op->getLoc(),
          resultType, ArrayRef<Value>{input()},
          ArrayRef<NamedAttribute>{attrs});
      operands.push_back(padOp);
    } else if (getOpQuant() == "BF16") {
      auto padOp = OpBuilder(op).create<tpu::TG_BF16_PadOp>(op->getLoc(),
          resultType, ArrayRef<Value>{input()},
          ArrayRef<NamedAttribute>{attrs});
      operands.push_back(padOp);
    }
  } else {
    operands.push_back(input());
  }
  operands.push_back(filter());
  operands.push_back(bias());
  operands.push_back(quant_scale());
  operands.push_back(quant_zeropoint());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "param",
      tpu::ConvParam::get(
          param().kernel_h(), param().kernel_w(), param().stride_h(),
          param().stride_w(), param().padding(), param().dilation_h(),
          param().dilation_w(), builder.getI32IntegerAttr(pad_t),
          builder.getI32IntegerAttr(pad_b), builder.getI32IntegerAttr(pad_l),
          builder.getI32IntegerAttr(pad_r), param().group(), param().is_dw(),
          param().with_bias(), param().ins(),
          param().pad_value(), builder.getContext())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_Conv2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_Conv2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::Conv3DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  std::vector<Value> operands;

  auto pad_d0 = param().padding_d0().getInt();
  auto pad_d1 = param().padding_d1().getInt();
  auto pad_t = param().padding_t().getInt();
  auto pad_b = param().padding_b().getInt();
  auto pad_l = param().padding_l().getInt();
  auto pad_r = param().padding_r().getInt();
  int32_t pad_h_begin = 0, pad_h_end = 0;
  int32_t pad_w_begin = 0, pad_w_end = 0;
  if (pad_t > 15) {
    pad_h_begin = pad_t;
    pad_t = 0;
  }
  if (pad_b > 15) {
    pad_h_end = pad_b;
    pad_b = 0;
  }
  if (pad_l > 15) {
    pad_w_begin = pad_l;
    pad_l = 0;
  }
  if (pad_r > 15) {
    pad_w_end = pad_r;
    pad_r = 0;
  }
  if (pad_h_begin > 0 || pad_h_end > 0 || pad_w_begin > 0 || pad_w_end > 0) {
    std::vector<int32_t> pads = {0, 0, pad_h_begin, pad_w_begin, 0, 0, pad_h_end, pad_w_end};
    std::vector<NamedAttribute> attrs;
    auto inputShape = getTensorShape(input());
    auto type = getResult().getType().template cast<TensorType>();

    std::vector<int64_t> shape(4);
    shape[0] = inputShape[0];
    shape[1] = inputShape[1] + pad_d0 + pad_d1;
    shape[2] = inputShape[2] + pad_h_begin + pad_h_end;
    shape[3] = inputShape[3] + pad_h_begin + pad_h_end;
    shape[4] = inputShape[4] + pad_w_begin + pad_w_end;
    auto resultType = RankedTensorType::get(shape, type.getElementType());

    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name().str() + "_pad")));
    attrs.push_back(builder.getNamedAttr("const_val", builder.getF32FloatAttr(0)));
    attrs.push_back(builder.getNamedAttr("pads", builder.getI32ArrayAttr(ArrayRef<int32_t>({pads}))));
    if (getOpQuant() == "INT8") {
      auto padOp = OpBuilder(op).create<tpu::TG_INT8_PadOp>(op->getLoc(),
          resultType, ArrayRef<Value>{input()},
          ArrayRef<NamedAttribute>{attrs});
      operands.push_back(padOp);
    } else if (getOpQuant() == "BF16") {
      auto padOp = OpBuilder(op).create<tpu::TG_BF16_PadOp>(op->getLoc(),
          resultType, ArrayRef<Value>{input()},
          ArrayRef<NamedAttribute>{attrs});
      operands.push_back(padOp);
    }
  } else {
    operands.push_back(input());
  }
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param",
          tpu::Conv3dParam::get(
              param().kernel_d(),
              param().kernel_h(),
              param().kernel_w(),
              param().stride_d(),
              param().stride_h(),
              param().stride_w(),
              param().padding(),
              param().dilation_d(),
              param().dilation_h(),
              param().dilation_w(),
              builder.getI32IntegerAttr(pad_d0),
              builder.getI32IntegerAttr(pad_d1),
              builder.getI32IntegerAttr(pad_t),
              builder.getI32IntegerAttr(pad_b),
              builder.getI32IntegerAttr(pad_l),
              builder.getI32IntegerAttr(pad_r),
              param().group(),
              param().is_dw(),
              param().with_bias(),
              param().ins(),
              builder.getContext())));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_Conv3DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ConvFcOp::convertToTG() {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(quant_scale());
  operands.push_back(quant_zeropoint());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ConvFcOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::CropOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  const unsigned nInputs = op->getNumOperands();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("crop_offset", crop_offsetAttr()));
  if (steps().hasValue()) {
    attrs.push_back(builder.getNamedAttr("steps", stepsAttr()));
  }

  if (getOpQuant() == "INT8" || getOpQuant() == "UINT8") {
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_CropOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_CropOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value tpu::DeConv2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  std::vector<Value> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());
  operands.push_back(quant_scale());
  operands.push_back(quant_zeropoint());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_DeConv2DOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_DeConv2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::DilateOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("fill_constant", fill_constantAttr()));
  attrs.push_back(builder.getNamedAttr("ins", insAttr()));

  if (getOpQuant() == "INT8") {
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_DilateOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_DilateOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value tpu::EmbeddingOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto castOp = cast<tpu::EmbeddingOp>(op);
  auto builder = Builder(op->getContext());

  std::vector<NamedAttribute> param; // no param need to pass
  auto paramAttr = builder.getDictionaryAttr(param);
  auto operationAttr = builder.getStringAttr(getOperationName());
  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
  attrs.push_back(builder.getNamedAttr("param", paramAttr));
  auto type = castOp.getResult().getType();
  if (getOpQuant() == "BF16" && getOpQuantParamType() == "MIX_BF16") {
    std::string name_i8 = name().str() + "_i8";
    attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_i8)));
    auto eltType = IntegerType::get(builder.getContext(), 8);
    auto shape = type.cast<TensorType>().getShape();
    auto type_i8 = RankedTensorType::get(shape, eltType);
    auto cpuOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), type_i8, ArrayRef<Value>{{input(), table()}},
        ArrayRef<NamedAttribute>{attrs});

    std::vector<NamedAttribute> attrs2;
    attrs2.push_back(builder.getNamedAttr("name", nameAttr()));
    int axis = shape.size() - 1;
    attrs2.push_back(
        builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis)));
    auto dequantOp = OpBuilder(op).create<tpu::TG_DequantOp>(
        op->getLoc(), type,
        ArrayRef<Value>{{cpuOp.getResult(), quant_scale(), quant_zeropoint()}},
        ArrayRef<NamedAttribute>{attrs2});
    return dequantOp.getResult();
  } else {
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    auto cpuOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), type, ArrayRef<Value>{{input(), table()}},
        ArrayRef<NamedAttribute>{attrs});
    return cpuOp.getResult();
  }
}

Value tpu::EltwiseAddOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(
      builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  if (do_early_stride()) {
    attrs.push_back(builder.getNamedAttr(
        "do_early_stride", builder.getBoolAttr(do_early_stride())));
    attrs.push_back(
        builder.getNamedAttr("early_stride_h", early_stride_hAttr()));
    attrs.push_back(
        builder.getNamedAttr("early_stride_w", early_stride_wAttr()));
  }

  if (getOpQuant() == "INT8") {
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    int rshift_i8 = static_cast<int8_t>(rshift->at(0));
    std::vector<int32_t> m_i8_inputs(nInputs, 1);

    // m_i8_inputs
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    for (unsigned i = 0; i < nInputs; ++i) {
      m_i8_inputs[i] = static_cast<int32_t>(multiplier->at(i));
    }

    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI8IntegerAttr(rshift_i8)));
    attrs.push_back(builder.getNamedAttr(
        "m_i8_inputs",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs}))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseAddOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    if (coeff().hasValue())
      attrs.push_back(builder.getNamedAttr("coeff", coeffAttr()));
    else {
      std::vector<float> coeffs(nInputs);
      for (unsigned i = 0; i < nInputs; i++)
        coeffs[i] = 1.0;
      attrs.push_back(builder.getNamedAttr("coeff",
                      builder.getF32ArrayAttr(ArrayRef<float>(coeffs))));
    }
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseAddOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::MulConstOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(
      builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("const_val", const_valAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_MulConstOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_MulConstOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::EltwiseMaxOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
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
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMaxOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::EltwiseMinOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
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
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMinOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::EltwiseMulOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));

  if (getOpQuant() == "INT8") {
    // MUL
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    int8_t rshift_i8 = static_cast<int8_t>(rshift->at(0));

    // m_i8_output
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    int32_t m_i32_output = static_cast<int32_t>(multiplier->at(0));

    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI8IntegerAttr(rshift_i8)));
    attrs.push_back(builder.getNamedAttr(
        "m_i32_output", builder.getI32IntegerAttr(m_i32_output)));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::FullyConnectedOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());
  operands.push_back(quant_scale());
  operands.push_back(quant_zeropoint());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("input_transpose",
                                       builder.getBoolAttr(input_transpose())));
  attrs.push_back(builder.getNamedAttr(
      "output_transpose", builder.getBoolAttr(output_transpose())));
  if (getOpQuant() == "INT8") {
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    std::vector<int32_t> rshift_v(rshift->begin(), rshift->end());
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI32ArrayAttr(rshift_v)));
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    std::vector<int32_t> multiplier_v(multiplier->begin(), multiplier->end());
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI32ArrayAttr(multiplier_v)));
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_FullyConnectedOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_FullyConnectedOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::InterpOp::convertToTG() {
  Operation *op = this->getOperation();
  auto castOp = cast<InterpOp>(op);
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  auto builder =
          Builder(op->getContext());
  std::vector<NamedAttribute> param;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : castOp->getAttrs()) {
    if (attr.first == "name" || attr.first == "gaddr" ||
        attr.first == "quant") {
      continue;
    }
    param.push_back(attr);
  }
  auto operationAttr = builder.getStringAttr(castOp.getOperationName());
  auto paramAttr = builder.getDictionaryAttr(param);

  attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
  attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
  attrs.push_back(builder.getNamedAttr("param", paramAttr));

  std::vector<Value> operands{castOp.input()};

  auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
      op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();

}

Value tpu::InstanceNormOp::convertToTG() {
  Operation *op = this->getOperation();
  auto castOp = cast<InstanceNormOp>(op);
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  auto builder =
          Builder(op->getContext());
  std::vector<NamedAttribute> param;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : castOp->getAttrs()) {
    if (attr.first == "name" || attr.first == "gaddr" ||
        attr.first == "quant") {
      continue;
    }
    param.push_back(attr);
  }
  auto operationAttr = builder.getStringAttr(castOp.getOperationName());
  auto paramAttr = builder.getDictionaryAttr(param);

  attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
  attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
  attrs.push_back(builder.getNamedAttr("param", paramAttr));

  std::vector<Value> operands(op->getOperands().begin(),
      op->getOperands().end());

  auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
      op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();

}

Value tpu::LrnOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  if (getOpQuant() == "INT8") {
    int nInputs = 3; // input, sqr table, power table
    std::vector<Value> operands;
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


    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LrnOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {

    int nInputs = 3; // input
    std::vector<Value> operands;
    for (auto i = 0; i < nInputs; ++i) {
      operands.push_back(op->getOperand(i));
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("local_size", local_sizeAttr()));
    attrs.push_back(builder.getNamedAttr("alpha", alphaAttr()));
    attrs.push_back(builder.getNamedAttr("k", kAttr()));
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("sum_rshift", builder.getI32IntegerAttr(0)));
    attrs.push_back(builder.getNamedAttr("lrn_rshift", builder.getI32IntegerAttr(0)));
    attrs.push_back(builder.getNamedAttr("quant_data0", builder.getI32IntegerAttr(0)));
    attrs.push_back(builder.getNamedAttr("quant_data1", builder.getI32IntegerAttr(0)));

    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LrnOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }

  llvm_unreachable("unsupported type");
  return nullptr;
}

Value tpu::LeakyReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("negative_slope", negative_slopeAttr()));

  if (getOpQuant() == "INT8") {
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
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LeakyReluOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }

  llvm_unreachable("unsupported type");
}

Value tpu::LogOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::MishOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PermuteOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("order", orderAttr()));

  if (getOpQuant() == "INT8" || getOpQuant() == "UINT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PermuteOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PermuteOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PadOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("pads", padsAttr()));
  attrs.push_back(builder.getNamedAttr("const_val", const_valAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("mode", modeAttr()));

  if (getOpQuant() == "INT8" || getOpQuant() == "UINT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PadOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PadOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PoolAvg2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
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
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolAvg2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PoolMax2DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolMax2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolMax2DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PoolMax3DOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolMax3DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolMax3DOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(getOperand(0));
  operands.push_back(getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
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
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PReluOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

// cpu quant, only support fp32 <=> (int8/bf16)
static bool quant_by_cpu(llvm::StringRef from, llvm::StringRef to) {
  if ((from == "NONE" || from == "FP32") && (to == "INT8")) {
    return true;
  }
  if ((from == "INT8") && (to == "NONE" || to == "FP32")) {
    return true;
  }
  return false;
}

Value tpu::QuantOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  auto parentOp = this->getOperand().getDefiningOp();
  if (isa<tpu::InputOp>(parentOp) && quant_by_cpu(from(), to())) {
    std::vector<NamedAttribute> param;
    param.push_back(builder.getNamedAttr("from", fromAttr()));
    param.push_back(builder.getNamedAttr("to", toAttr()));
    param.push_back(builder.getNamedAttr("scale", scaleAttr()));
    auto paramAttr = builder.getDictionaryAttr(param);
    auto operationAttr = builder.getStringAttr(getOperationName());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));
    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else {
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(builder.getNamedAttr("from", fromAttr()));
    attrs.push_back(builder.getNamedAttr("to", toAttr()));
    attrs.push_back(builder.getNamedAttr("scale", scaleAttr()));
    auto newOp = OpBuilder(op).create<tpu::TG_QuantOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
}

Value tpu::ReflectionPadOp::convertToTG() {
  Operation *op = this->getOperation();
  //TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  for (int i = 0; i < 3; i++) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("pads", padsAttr()));
  if (getOpQuant() == "INT8") {
    // no need to quant
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReflectionPadOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReflectionPadOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  //TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    // no need to quant
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReluOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReluOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReorgOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("stride", strideAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReorgOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReorgOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReverseOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReverseOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReverseOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ShuffleChannelOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("group", groupAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ShuffleChannelOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ShuffleChannelOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::SwapChannelOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("channel_order", channel_orderAttr()));

  if (getOpQuant() == "INT8") {
    //assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SwapChannelOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SwapChannelOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

static inline int align_up(int x, int n) {
  if (n == 0 || n == 1) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

Value tpu::CscOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  auto pixel_format = this->pixel_format().str();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  int yuv_type = -1;
  bool need_stride_copy = false;
  if (pixel_format == "YUV420_PLANAR") {
    yuv_type = 1;
  } else if (pixel_format == "YUV_NV12") {
    yuv_type = 2;
  } else if (pixel_format == "YUV_NV21") {
    yuv_type = 3;
  } else if (pixel_format == "RGB_PLANAR" || pixel_format == "BGR_PLANAR" ||
             pixel_format == "RGBA_PLANAR") {
    need_stride_copy = true;
  }

  assert(getOpQuant() == "INT8" || getOpQuant() == "UINT8");
  if (yuv_type > 0) {
    attrs.push_back(builder.getNamedAttr("y_align", y_alignAttr()));
    attrs.push_back(builder.getNamedAttr("w_align", w_alignAttr()));
    attrs.push_back(builder.getNamedAttr("channel_align", channel_alignAttr()));
    attrs.push_back(
        builder.getNamedAttr("pixel_type", builder.getI32IntegerAttr(yuv_type)));
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_Yuv420CscOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (need_stride_copy) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t input_size, n, c, h, w;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
    getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
    getNCHW(input_shape, n, c, h, w);
    getNCHW(output_shape, on, oc, oh, ow);

    std::vector<int> i_stride(4, 0);
    std::vector<int> o_stride(4, 0);
    i_stride[3] = 1;
    i_stride[2] = align_up(ow, w_alignAttr().getInt());
    i_stride[1] = align_up(i_stride[2] * oh, channel_alignAttr().getInt());
    i_stride[0] = i_stride[1] * c;

    o_stride[3] = 1;
    o_stride[2] = ow;
    o_stride[1] = o_stride[2] * oh;
    o_stride[0] = i_stride[1] * oc;

    attrs.push_back(
        builder.getNamedAttr("input_stride", builder.getI32ArrayAttr(i_stride)));
    attrs.push_back(
        builder.getNamedAttr("output_stride", builder.getI32ArrayAttr(o_stride)));
    auto newOp = OpBuilder(op).create<tpu::TG_StrideCopyOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t input_size, n, c, h, w;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
    getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
    getNCHW(input_shape, n, c, h, w);
    getNCHW(output_shape, on, oc, oh, ow);

    int unaligned_w = (int)(oc * oh * ow / (c * h));
    std::vector<int> crop_offset{0, 0, 0, 0};
    attrs.push_back(
        builder.getNamedAttr("crop_offset", builder.getI32ArrayAttr(crop_offset)));
    auto elementType = getResult().getType().cast<TensorType>().getElementType();
    auto cropType = RankedTensorType::get({n, c, h, unaligned_w}, elementType);
    auto cropOp = OpBuilder(op).create<tpu::TG_INT8_CropOp>(op->getLoc(),
        cropType, ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    attrs.clear();
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name().str() + "_reshape")));
    auto newOp = OpBuilder(op).create<tpu::ReshapeOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{{cropOp}},
        ArrayRef<NamedAttribute>{attrs});

    return newOp.getResult();
  }
}

Value tpu::TileOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;

  // keep info to tg
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
  attrs.push_back(builder.getNamedAttr("tiles", tilesAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_TileOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_TileOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");

  return nullptr;
}

Value tpu::PixelShuffleOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  if (0 && this->mode().str() == "DCR") {
    // use cpu op
    std::vector<NamedAttribute> param;
    param.push_back(builder.getNamedAttr("mode", modeAttr()));
    param.push_back(
        builder.getNamedAttr("upscale_factor", upscale_factorAttr()));
    auto paramAttr = builder.getDictionaryAttr(param);
    auto operationAttr = builder.getStringAttr(getOperationName());
    std::vector<NamedAttribute> cpu_attrs;
    cpu_attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    cpu_attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    cpu_attrs.push_back(builder.getNamedAttr("param", paramAttr));
    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{cpu_attrs});
    return newOp.getResult();
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("upscale_factor", upscale_factorAttr()));
  attrs.push_back(builder.getNamedAttr("mode", modeAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PixelShuffleOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PixelShuffleOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ClipOp::convertToTG() {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //   TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("min", minAttr()));
  attrs.push_back(builder.getNamedAttr("max", maxAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ClipOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ClipOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PoolMaskOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0)); // input
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("scale", scaleAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolMaskOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolMaskOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ScaleLutOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  std::vector<Value> operands;
  operands.push_back(op->getOperand(0)); // input
  operands.push_back(op->getOperand(1)); // table
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ScaleLutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::SigmoidOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(
        builder.getNamedAttr("method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::SwishOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(
        builder.getNamedAttr("method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::PowOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  for (auto i = 0; i < 3; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(
        builder.getNamedAttr("method", builder.getStringAttr("mantissa")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::TanHOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::EluOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ExpOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::UpsampleOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("scale_h", scale_hAttr()));
  attrs.push_back(builder.getNamedAttr("scale_w", scale_wAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_UpsampleOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_UpsampleOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReduceL2Op::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto castOp = cast<ReduceL2Op>(op);

  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());
  operands.push_back(table());
  operands.push_back(mantissa_table());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
  attrs.push_back(builder.getNamedAttr("axes", castOp.axesAttr()));

  // TODO: tpu support
  auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReduceL2Op>(
      op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tpu::ReduceMeanOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axes", axesAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert( !isTensorNone(quant_rshift()) );
    TensorFile *wTF = getWeightTensorFile(op);
    auto rshift = readWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReduceMeanOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReduceMeanOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReduceMaxOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axes", axesAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert( !isTensorNone(quant_rshift()) );
    TensorFile *wTF = getWeightTensorFile(op);
    auto rshift = readWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReduceMaxOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReduceMaxOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReduceMinOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axes", axesAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert( !isTensorNone(quant_rshift()) );
    TensorFile *wTF = getWeightTensorFile(op);
    auto rshift = readWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReduceMinOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReduceMinOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ReduceSumOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axes", axesAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert( !isTensorNone(quant_rshift()) );
    TensorFile *wTF = getWeightTensorFile(op);
    auto rshift = readWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ReduceSumOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ReduceSumOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::GruOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  const int nInputs =  8;
  //input + weight + recurrence + bias + initial_h
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("linear_before_reset", linear_before_resetAttr()));
  attrs.push_back(builder.getNamedAttr("bidirectional", bidirectionalAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_GruOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::LstmOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  const int nInputs =  op->getNumOperands();
  //input + recurrence + bias + initial_h + initial_c + cont + 4 tables
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("bidirectional", bidirectionalAttr()));
  attrs.push_back(builder.getNamedAttr("final_h", final_hAttr()));
  attrs.push_back(builder.getNamedAttr("final_c", final_cAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LstmOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::LayerNormOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  const int nInputs =  op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("eps", epsAttr()));
  attrs.push_back(builder.getNamedAttr("normalized_shape", normalized_shapeAttr()));

  if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LayerNormOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::SoftmaxOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  const int nInputs =  5;
  //act + exp/rep table
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
  attrs.push_back(builder.getNamedAttr("do_log", do_logAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SoftmaxOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SoftmaxOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else {
    // Return same opValue
    auto newOp = OpBuilder(op).create<tpu::SoftmaxCpuOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
}

Value tpu::SoftPlusOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  int nInputs = 3; // input and table
  std::vector<Value> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  // get default/assign value
  attrs.push_back(builder.getNamedAttr("max_range",
      builder.getF32FloatAttr(max_range().convertToFloat())));
  attrs.push_back(builder.getNamedAttr("min_range",
      builder.getF32FloatAttr(min_range().convertToFloat())));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr(
        "method", builder.getStringAttr("slope")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::StdOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("start_dim", start_dimAttr()));
  attrs.push_back(builder.getNamedAttr("unbiased", unbiasedAttr()));

  if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_StdOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::QuadraticSumOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr() ? axisAttr() :
      builder.getI32IntegerAttr(this->axis())));
  attrs.push_back(builder.getNamedAttr("high_precision", high_precisionAttr() ? high_precisionAttr() :
      builder.getBoolAttr(this->high_precision().getValue())));
  assert(getOpQuant() == "BF16");
  auto newOp = OpBuilder(op).create<tpu::TG_BF16_QuadraticSumOp>(op->getLoc(),
                  getResult().getType(), ArrayRef<Value>{operands},
                  ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tpu::MatMulOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(op->getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("left_transpose",
                                       builder.getBoolAttr(left_transpose())));
  attrs.push_back(builder.getNamedAttr("right_transpose",
                                       builder.getBoolAttr(right_transpose())));
  attrs.push_back(builder.getNamedAttr(
      "output_transpose", builder.getBoolAttr(output_transpose())));
  attrs.push_back(
      builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));
  if (getOpQuant() == "INT8") {
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr(
        "rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr(
        "multiplier",
        builder.getI32IntegerAttr(static_cast<int32_t>(multiplier->at(0)))));
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_MatMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_MatMulOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::ZeroMaskOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("positive", positiveAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ZeroMaskOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ZeroMaskOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

template<typename OpTy>
struct DefaultToTGPattern : public RewritePattern {
  DefaultToTGPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op);
    if (!tpuOp) {
      return failure();
    }
    auto newValue = tpuOp.convertToTG();
    if (!newValue) {
      return failure();
    }
    rewriter.replaceOp(op, {newValue});
    return success();
  }
};

template<typename OpTy>
struct DefaultErasePattern : public RewritePattern {
  DefaultErasePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {op->getOperand(0)});
    return success();
  }
};

template<typename OpTy>
struct FoldReshapePattern : public RewritePattern {
  FoldReshapePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto laterReshapeOp = cast<OpTy>(op);

    auto formerOp = laterReshapeOp.getOperand().getDefiningOp();
    if (!matchPattern(formerOp, m_Op<OpTy>())) {
      return failure();
    }
    auto formerScaleOp = cast<OpTy>(formerOp);

    laterReshapeOp.getOperation()->setOperand(0, formerScaleOp.getOperand());
    return success();
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

  return packed;
}

template <typename OpTy>
struct PackWeightConv2DOpPattern : public RewritePattern {
  PackWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    if (getOpQuant(op) != "INT8") {
      // for perchannel multiplier mode only
      return failure();
    }
    if ( !isTensorNone(convOp.bias()) ) {
      auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
      if (biasOp.lowered()) {
        // packed already
        return failure();
      }
    }
    assert( !isTensorNone(convOp.quant_rshift()) );
    assert( !isTensorNone(convOp.quant_multiplier()) );
    LLVM_DEBUG(llvm::errs() << "Pack Weight for Conv2D: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    // get param
    auto filter_type = convOp.filter().getType().template cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t oc;
    auto g = convOp.param().group().getInt();
    if (g != 1 || filter_shape.size() == 5) {
      oc = filter_shape[0] * filter_shape[1];
    } else {
      assert(filter_shape.size() == 4);
      oc = filter_shape[0];
    }

    // get tensor
    std::unique_ptr<std::vector<float> > bias = nullptr;
    if ( !isTensorNone(convOp.bias()) ) {
      bias = readWeightTensor<float>(convOp.bias(), wTF);
    }
    auto rshift = readWeightTensor<float>(convOp.quant_rshift(), wTF);
    auto multiplier = readWeightTensor<float>(convOp.quant_multiplier(), wTF);

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
    auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
    biasOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_rshift and quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    convOp.setOperand(5, NoneOp);
    convOp.setOperand(6, NoneOp);

    return success();
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

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::BroadcastMulOp>(op);

    // Only int8 need pack
    if (getOpQuant(op) == "BF16") {
      LLVM_DEBUG(llvm::errs()<< "Pack Weight for BroadcastMul ONLY apply INT8 we skip it\n";);
      return failure();
    }
    bool align = castOp.align_right();
    int64_t n, c, h, w, bn, bc, bh, bw;
    auto shape = getTensorShape(op->getOperand(0));
    getNCHW(shape, n, c, h, w, align);
    auto bshape = getTensorShape(op->getOperand(1));
    getNCHW(bshape, bn, bc, bh, bw, align);
    if ((bn == 1 || bn == n) && c == bc && bh == 1 && bw == 1) {

    } else {
      return failure();
    }

    auto rshiftOp =
        cast<tpu::LoadWeightOp>(castOp.quant_rshift().getDefiningOp());
    if (rshiftOp.lowered()) {
      // packed already
      return failure();
    }
    assert(getOpQuantParamType(op) == "RSHIFT_AND_M_I32");
    assert(!isTensorNone(castOp.quant_rshift()));
    assert(!isTensorNone(castOp.quant_multiplier()));
    LLVM_DEBUG(llvm::errs() << "Pack Weight for BroadcastMul: " << getOpName(op)
                            << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    // get param
    int64_t oc = getTensorSize(op->getOperand(1));

    // get tensor
    std::unique_ptr<std::vector<float>> pc_info = nullptr;
    auto rshift = readAndDeleteWeightTensor<float>(castOp.quant_rshift(), wTF);
    auto multiplier =
        readAndDeleteWeightTensor<float>(castOp.quant_multiplier(), wTF);

    // expand
    auto rshift_perchannel =
        std::make_unique<std::vector<float>>(oc, rshift->at(0));
    auto multiplier_perchannel =
        std::make_unique<std::vector<float>>(oc, multiplier->at(0));

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(nullptr, rshift_perchannel.get(),
                             multiplier_perchannel.get(), oc, packedShape);

    // this is tricky, as where is no bias() to reuse, use quant_rshift()
    // instead store to the packed per_channel operand in "UINT8"
    addWeightTensorAndUpdateWeightOp<uint8_t>(
        castOp.quant_rshift(), "pack", *packed, packedShape, "UINT8", wTF);
    rshiftOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());
    castOp.setOperand(5, NoneOp);

    setOpQuantParamType(op, "RSHIFT_AND_M_I32");
    return success();
  }
};

template <typename T>
static void rotateConvolutionFilter(std::vector<T> &w,
                                    const std::vector<int64_t> &s) {
  int64_t oc, ic, kh, kw;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    kh = s[2];
    kw = s[3];
  } else if (s.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = s[0] * s[1];
    ic = s[2];
    kh = s[3];
    kw = s[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }

  std::vector<T> w_t(w.size());
  if (kh == 1 && kw == 1) {
    return;
  } else {
    // for other conv, rotate 180
    for (int64_t i = 0; i < oc * ic; i++) {
      for (int64_t j = 0; j < kh; j++) {
        for (int64_t k = 0; k < kw; k++) {
          w_t[i * kh * kw + (kh - 1 - j) * kw + (kw - 1) - k] =
                                                w[i * kh * kw + j * kw + k];
        }
      }
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

// shape[oc, ic, kh, kw] => [oc, kh, kw, ic]
template <typename T>
static void transposeConvolutionFilter(std::vector<T> &w,
                                       const std::vector<int64_t> &s) {
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

static void get_strides_from_shapes5d(int strides[5], const int shapes[5],
                                      int ws)
{
  strides[5 - 1] = ws;
  for (int i = 5 - 2; i >= 0; i--)
    strides[i] = shapes[i + 1] * strides[i + 1];
}

static int get_tensor5d_offset(int poss[5], const int strides[5])
{
  int offset = 0;
  for (int i = 0; i < 5; i++)
    offset += poss[i] * strides[i];

  return offset;
}

// (oc, ic, kd, kh, kw) -> (kd, oc, kh, kw, ic)
template<typename T>
static void transposeConvolution3dFilter(std::vector<T> &w,
    const std::vector<int64_t> &s) {
  int oc, ic, kd, kh, kw;
  if (s.size() == 5) {
    // oc, ic, kd, kh, kw
    oc = (int)s[0];
    ic = (int)s[1];
    kd = (int)s[2];
    kh = (int)s[3];
    kw = (int)s[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }

  std::vector<T> w_t(w.size());
  int cpu_shapes[5] = {oc, ic, kd, kh, kw};
  int tpu_shapes[5] = {kd, oc, kh, kw, ic};

  // logical stride, in unit of float
  int cpu_strides[5], tpu_strides[5];
  get_strides_from_shapes5d(cpu_strides, cpu_shapes, 1);
  get_strides_from_shapes5d(tpu_strides, tpu_shapes, 1);

  LLVM_DEBUG(llvm::dbgs()
      << "transposeConvolution3dFilter\n  "
      << "shape(oc=" << oc << ", ic=" << ic << ", kd=" << kd << ", kh=" << kh
      << ", kw=" << kw << ")\n");

  // (oc, ic, id, kh, kw) -> (id, oc, khxkw, ic)
  for (int i = 0; i < cpu_shapes[0]; i++) {
    for (int j = 0; j < cpu_shapes[1]; j++) {
      for (int z = 0; z < cpu_shapes[2]; z++) {
        for (int y = 0; y < cpu_shapes[3]; y++) {
          for (int x = 0; x < cpu_shapes[4]; x++) {
            int cpu_poss[5] = {i, j, z, y, x};
            int tpu_poss[5] = {z, i, y, x, j};
            int cpu_offset = get_tensor5d_offset(cpu_poss, cpu_strides);
            int tpu_offset = get_tensor5d_offset(tpu_poss, tpu_strides);
            w_t[tpu_offset] = w[cpu_offset];

            LLVM_DEBUG(llvm::dbgs()
                << "  [i=" << i << "][j=" << j << "][z=" << z << "][y=" << y
                << "][x=" << x << "] w_t[" << tpu_offset
                << "]=w[" << cpu_offset << "]=" << w[cpu_offset] << "\n");

          }
        }
      }
    }
  }

  w.assign(w_t.begin(), w_t.end());
}

template <typename T>
static void transposeFullyConnectedFilter(std::vector<T> &w,
                                          const std::vector<int64_t> &s) {
  int dim = s.size();
  int batch = std::accumulate(s.data(), s.data() + dim - 2, 1,
                              std::multiplies<int64_t>());
  int row = s[dim - 2];
  int col = s[dim - 1];
  std::vector<T> w_t(row * col);
  for (int b = 0; b < batch; b++) {
    T *pdata = w.data() + b * row * col;
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        w_t[j * row + i] = pdata[i * col + j];
      }
    }
    std::copy(w_t.begin(), w_t.end(), pdata);
  }
}

template <typename T>
static void transpose_row_col(T *data, int row, int col) {
  std::vector<T> w_t(row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = data[i * col + j];
    }
  }
  std::copy(w_t.begin(), w_t.end(), data);
}

template <typename T>
static void transposeRnnFilter(std::vector<T> &data,
                               const std::vector<int64_t> &shape) {
  assert(shape.size() == 3);
  int64_t num_dir = shape[0];
  int64_t hidden_size = shape[2];
  assert(shape[1] % hidden_size == 0);
  int gate_num = shape[1] / hidden_size;
  T *p_data = data.data();
  for (int i = 0; i < gate_num * num_dir; i++) {
    transpose_row_col(p_data, hidden_size, hidden_size);
    p_data += hidden_size * hidden_size;
  }
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

static void transposeBiasInt32(std::vector<int32_t> &w_int32) {
  int8_t *ptr = reinterpret_cast<int8_t *>(w_int32.data());
  std::vector<int8_t> w(ptr, ptr + w_int32.size() * sizeof(int32_t));
  std::vector<int8_t> w_t(w.size());
  for (size_t i = 0; i < w_int32.size(); i++) {
    for (size_t j = 0; j < 4; j++) {
      w_t[j * w_int32.size() + i] = w[i * 4 + j];
    }
  }
  memcpy(ptr, w_t.data(), w_t.size());
}

static void transposeBiasFp32(std::vector<float> &bias_f32,
                              std::vector<uint32_t> &bias_u32) {
  // Split into high/low part
  std::vector<uint16_t> bias_fp32_high;
  std::vector<uint16_t> bias_fp32_low;
  float *biasFloatPtr = bias_f32.data();
  int size = bias_f32.size();
  for (int i = 0; i < size; ++i) {
    unsigned short *temp_short_ptr =
        reinterpret_cast<unsigned short *>(biasFloatPtr + i);
    bias_fp32_high.push_back(temp_short_ptr[1]);
    bias_fp32_low.push_back(temp_short_ptr[0]);
  }
  std::vector<uint16_t> bias_reshape_fp32;
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(),
                           bias_fp32_high.end());
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                           bias_fp32_low.end());
  // then copy into uint32_t
  assert(bias_u32.size() == bias_f32.size());
  memcpy(bias_u32.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

typedef enum weight_process {
  WEIGHT_NORMAL,       // nothing to do
  WEIGHT_FC_TRANSPOSE, // do fc transpose
  WEIGHT_CONV2D_TRANSPOSE,
  WEIGHT_DECONV2D_TRANSPOSE,
  WEIGHT_CONV3D_TRANSPOSE,
  WEIGHT_RNN_TRANSPOSE,
} weight_process_t;

template <typename T>
static void process_weight(std::vector<T> &data,
                           const std::vector<int64_t> &shape,
                           weight_process_t type) {
  switch (type) {
  case WEIGHT_NORMAL:
    return;
  case WEIGHT_FC_TRANSPOSE:
    transposeFullyConnectedFilter(data, shape);
    return;
  case WEIGHT_CONV2D_TRANSPOSE:
    transposeConvolutionFilter(data, shape);
    return;
  case WEIGHT_DECONV2D_TRANSPOSE:
    rotateConvolutionFilter(data, shape);
    transposeConvolutionFilter(data, shape);
    return;
  case WEIGHT_CONV3D_TRANSPOSE:
    transposeConvolution3dFilter(data, shape);
    return;
  case WEIGHT_RNN_TRANSPOSE:
    transposeRnnFilter(data, shape);
    return;
  default:
    llvm_unreachable("not support");
    return;
  }
}

template <typename B>
static LogicalResult lowerWeight(Operation *op, TensorFile *wTF, B &builder,
                                 weight_process_t type) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op);
  if (weightOp == nullptr) {
    return failure();
  }
  if (weightOp.lowered() == true) {
    return failure();
  }
  auto storage = weightOp.storage();
  if (storage == "NONE") {
    // needn't lower
    return failure();
  }
  auto data = readAndDeleteWeightTensor<float>(weightOp, wTF);
  auto shape = getTensorShape(weightOp);
  auto size = getTensorSize(weightOp);

  if (storage == "BF16") {
    std::vector<bfloat16> data_bf16(size);
    F32ToBF16(data->data(), data_bf16.data(), size, false);
    process_weight(data_bf16, shape, type);
    addWeightTensorAndUpdateWeightOp<bfloat16>(weightOp, "lowered", data_bf16,
                                               shape, "BF16", wTF);
  } else if (storage == "INT8") {
    std::vector<int8_t> data_i8(data->begin(), data->end());
    process_weight(data_i8, shape, type);
    addWeightTensorAndUpdateWeightOp<int8_t>(weightOp, "lowered", data_i8,
                                             shape, "INT8", wTF);
  } else if (storage == "UINT8") {
    std::vector<uint8_t> data_u8(data->begin(), data->end());
    process_weight(data_u8, shape, type);
    addWeightTensorAndUpdateWeightOp<uint8_t>(weightOp, "lowered", data_u8,
                                              shape, "UINT8", wTF);
  } else {
    llvm_unreachable((storage.str() + " is not supported").c_str());
  }
  weightOp->setAttr("lowered", builder.getBoolAttr(true));
  return success();
}

static LogicalResult lowerWeight(Value op, TensorFile *wTF,
                                 PatternRewriter &rewriter,
                                 weight_process_t type) {
  if (isTensorNone(op)) {
    return failure();
  }
  return lowerWeight(op.getDefiningOp(), wTF, rewriter, type);
}

static LogicalResult lowerWeightGeneric(Operation *op) {
  Builder builder(op);
  auto wTF = getWeightTensorFile(op);
  return lowerWeight(op, wTF, builder, WEIGHT_NORMAL);
}

static LogicalResult lowerBias(Value op, TensorFile *wTF,
                               PatternRewriter &rewriter) {
  if (isTensorNone(op)) {
    return failure();
  }
  auto weightOp = dyn_cast_or_null<tpu::LoadWeightOp>(op.getDefiningOp());
  if (weightOp == nullptr) {
    return failure();
  }
  if (weightOp.lowered()) {
    return failure();
  }
  auto storage = weightOp.storage();
  std::vector<int64_t> shape;
  int64_t size;
  getTensorShapeAndSize(op, shape, size);
  auto bias = readAndDeleteWeightTensor<float>(op, wTF);
  if (storage == "NONE" || storage == "FP32") {
    // NOTE: for 1880v2, bias is fp32, rather than bf16
    // however, for simplicity, in quantizeBf16, we quantize all tensor into
    // bf16 before lowering to hardware, we need to expand the bf16 to fp32
    // first then transpose into 2 stripes of uint16_t
    std::vector<uint32_t> bias_u32(size);
    transposeBiasFp32(*bias, bias_u32);
    // after expand to FP32 and transpose, this is not FP32 anymore
    // it is 2 stripes of UINT16(BF16)
    // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
    // to change the shape
    addWeightTensorAndUpdateWeightOp<uint32_t>(op, "lowered", bias_u32, shape,
                                               "UINT32", wTF);
  } else if (storage == "INT32") {
    std::vector<int32_t> bias_i32(bias->begin(), bias->end());
    transposeBiasInt32(bias_i32);
    // after transpose, this is not INT32 anymore, it is 2 stripes of UINT8
    // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
    // to change the shape.
    std::vector<uint32_t> bias_u32(size);
    memcpy(bias_u32.data(), bias_i32.data(), size * sizeof(int32_t));
    addWeightTensorAndUpdateWeightOp<uint32_t>(op, "lowered", bias_u32, shape,
                                               "UINT32", wTF);
  } else if (storage == "INT16") {
    std::vector<int16_t> bias_i16(bias->begin(), bias->end());
    transposeBiasInt16(bias_i16);
    std::vector<uint16_t> bias_u16(size);
    memcpy(bias_u16.data(), bias_i16.data(), size * sizeof(int16_t));
    // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
    // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
    // to change the shape.
    addWeightTensorAndUpdateWeightOp<uint16_t>(op, "lowered", bias_u16, shape,
                                               "UINT16", wTF);
  } else {
    llvm_unreachable((storage.str() + " is not supported").c_str());
  }
  weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
  return success();
}

// bias shape [batch, N]
static LogicalResult lowerBiasForGroupFC(Value op, TensorFile *wTF,
                                         PatternRewriter &rewriter, int batch,
                                         int n) {
  if (isTensorNone(op)) {
    return failure();
  }
  auto weightOp = dyn_cast_or_null<tpu::LoadWeightOp>(op.getDefiningOp());
  if (weightOp == nullptr) {
    return failure();
  }
  if (weightOp.lowered()) {
    return failure();
  }
  auto storage = weightOp.storage();
  auto bias = readAndDeleteWeightTensor<float>(op, wTF);
  std::vector<int64_t> shape = {batch, n};
  size_t size = batch * n;
  if (storage == "NONE" || storage == "FP32") {
    std::vector<uint32_t> bias_u32(size);
    std::vector<float> tmp_bias(n);
    std::vector<uint32_t> tmp_u32(n);
    for (int b = 0; b < batch; b++) {
      std::copy(bias->data() + b * n, bias->data() + (b + 1) * n,
                tmp_bias.data());
      transposeBiasFp32(tmp_bias, tmp_u32);
      std::copy(tmp_u32.begin(), tmp_u32.end(), bias_u32.data() + b * n);
    }
    addWeightTensorAndUpdateWeightOp<uint32_t>(op, "lowered", bias_u32, shape,
                                               "UINT32", wTF);
  } else if (storage == "INT32") {
    std::vector<uint32_t> bias_u32(size);
    std::vector<int32_t> bias_i32(n);
    for (int b = 0; b < batch; b++) {
      std::copy(bias->data() + b * n, bias->data() + (b + 1) * n,
                bias_i32.data());
      transposeBiasInt32(bias_i32);
      memcpy(bias_u32.data() + b * n, bias_i32.data(), n * sizeof(int32_t));
    }
    // after transpose, this is not INT32 anymore, it is 2 stripes of UINT8
    // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
    // to change the shape.
    addWeightTensorAndUpdateWeightOp<uint32_t>(op, "lowered", bias_u32, shape,
                                               "UINT32", wTF);
  } else {
    llvm_unreachable((storage.str() + " is not supported").c_str());
  }
  weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
  return success();
}

template <typename OpTy>
struct LowerWeightConv2DOpPattern : public RewritePattern {
  LowerWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    // lower filter
    auto type = isa<tpu::DeConv2DOp>(op) ? WEIGHT_DECONV2D_TRANSPOSE
                                         : WEIGHT_CONV2D_TRANSPOSE;
    auto ret_filter = lowerWeight(convOp.filter(), wTF, rewriter, type);
    // lower bias
    auto ret_bias = lowerBias(convOp.bias(), wTF, rewriter);
    if (succeeded(ret_filter) || succeeded(ret_bias)) {
      return success();
    }
    return failure();
  }
};

template <typename OpTy>
struct LowerWeightConv3DOpPattern : public RewritePattern {
  LowerWeightConv3DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    // lower filter
    auto type = getOpQuant(op) == "BF16" ? WEIGHT_CONV3D_TRANSPOSE
                                         : WEIGHT_CONV2D_TRANSPOSE;
    auto ret_filter = lowerWeight(convOp.filter(), wTF, rewriter, type);
    // lower bias
    auto ret_bias = lowerBias(convOp.bias(), wTF, rewriter);
    if (succeeded(ret_filter) || succeeded(ret_bias)) {
      return success();
    }
    return failure();
  }
};

struct LowerWeightFullyConnectedOpPattern : public RewritePattern {
  LowerWeightFullyConnectedOpPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    // lower filter
    auto type = WEIGHT_FC_TRANSPOSE;
    auto ret_filter = lowerWeight(fcOp.filter(), wTF, rewriter, type);
    // lower bias
    int batch_high, batch_low, m, k, n;
    parseFullyConnectedParam<tpu::FullyConnectedOp>(op, batch_high, batch_low,
                                                    m, k, n);
    int batch = batch_high * batch_low;
    auto ret_bias = failure();
    if (batch == 1) {
      ret_bias = lowerBias(fcOp.bias(), wTF, rewriter);
    } else {
      ret_bias = lowerBiasForGroupFC(fcOp.bias(), wTF, rewriter, batch, n);
    }
    if (succeeded(ret_filter) || succeeded(ret_bias)) {
      return success();
    }
    return failure();
  }
};

template <typename OpTy>
struct LowerWeightRNNOpPattern : public RewritePattern {
  LowerWeightRNNOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (getOpQuant(op) != "BF16") {
      return failure();
    }
    auto castOp = cast<OpTy>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    auto type = WEIGHT_RNN_TRANSPOSE;
    auto ret_r = lowerWeight(castOp.recurrence(), wTF, rewriter, type);
    auto ret_b = lowerBias(castOp.bias(), wTF, rewriter);
    if (succeeded(ret_r) || succeeded(ret_b)) {
      return success();
    }
    return failure();
  }
};

struct LowerWeightDetectionOutputOpPattern : public RewritePattern {
  LowerWeightDetectionOutputOpPattern(MLIRContext *context)
      : RewritePattern("tpu.detectionoutput", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto prOp = cast<tpu::DetectionOutputOp>(op);
    auto weightOp = cast<tpu::LoadWeightOp>(prOp.getOperand(2).getDefiningOp());
    assert(weightOp);
    if (weightOp.lowered()) {
      // lowered already
      return failure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for DetectionOutputOp: "
                            << getOpName(op) << "\n";);
    weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
    return success();
  }
};

struct LowerWeightInstanceNormOpPattern : public RewritePattern {
  LowerWeightInstanceNormOpPattern(MLIRContext *context)
      : RewritePattern("tpu.instance_norm", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto prOp = cast<tpu::InstanceNormOp>(op);
    auto scale = prOp.getOperand(1).getDefiningOp();
    auto bias = prOp.getOperand(2).getDefiningOp();
    auto weightOp = cast<tpu::LoadWeightOp>(scale);
    if (weightOp.lowered()) {
      // lowered already
      return failure();
    }

    // lower scale
    weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
    weightOp->setAttr("storage", rewriter.getStringAttr("FP32"));

    if (auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(bias)) {
      weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
      weightOp->setAttr("storage", rewriter.getStringAttr("FP32"));
    }

    LLVM_DEBUG(llvm::errs() << "Lower Weight for InstanceNormOp: " << getOpName(op) << "\n";);
    return success();
  }
};

template <typename OpTy>
struct LowerCpuOpDefaultPattern : public RewritePattern {
  LowerCpuOpDefaultPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs() << "Lower Cpu Op " << castOp.getOperationName() << ":"
                            << getOpName(castOp)<< "\n";);

    auto builder = Builder(op->getContext());
    std::vector<NamedAttribute> param;
    std::vector<NamedAttribute> attrs;
    for (auto& attr : op->getAttrs()) {
      if (attr.first == "name"
         || attr.first == "gaddr"
         || attr.first == "quant") {
        continue;
      }
      param.push_back(attr);
    }
    auto operationAttr = builder.getStringAttr(castOp.getOperationName());
    auto paramAttr = builder.getDictionaryAttr(param);

    attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));

    std::vector<Value> operands(op->getOperands().begin(),
                                  op->getOperands().end());

    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
        castOp.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    auto result = newOp.getResult();
    rewriter.replaceOp(op, {result});

    return success();
  }
};


template <typename OpTy>
struct LowerCustomOpPattern : public RewritePattern {
  LowerCustomOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs() << "Lower Custom Op " << castOp.getOperationName() << ":"
                            << getOpName(castOp)<< "\n";);
    auto builder = Builder(op->getContext());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", castOp.nameAttr()));
    attrs.push_back(builder.getNamedAttr("operation_name", castOp.operation_nameAttr()));
    attrs.push_back(builder.getNamedAttr("param", castOp.paramAttr()));

    std::vector<Value> operands(op->getOperands().begin(),
                                  op->getOperands().end());

    if (castOp.tpu()) {
      if (castOp.getOpQuant() == "BF16") {
        auto newOp = OpBuilder(op).create<tpu::TG_BF16_GenericTpuOp>(
            op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        auto result = newOp.getResult();
        rewriter.replaceOp(op, {result});
      } else {
        llvm_unreachable("unsupported type");
      }
    } else {
      auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
          castOp.getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      auto result = newOp.getResult();
      rewriter.replaceOp(op, {result});
    }

    return success();
  }
};

template <typename OpTy>
struct EliminateInputQuantOpPattern: public RewritePattern {
  EliminateInputQuantOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  void updateInputOpNameIfNeeded(PatternRewriter &rewriter,
                                 Operation *op, std::string suffix) const {
    bool needed = true;
    auto curr_name = getOpName(op).str();
    if (curr_name.size() > suffix.size()) {
      auto tail = curr_name.substr(curr_name.size() - suffix.size());
      if (tail == suffix) {
        needed = false;
      }
    }
    if (needed) {
      auto nameAttr = rewriter.getStringAttr(curr_name + suffix);
      op->setAttr("name", nameAttr);
    }
  }

  bool ifAllSiblingsAreSameQuantMode(Operation *input_op, StringRef mode) const {
    for (auto &use : input_op->getResult(0).getUses()) {
      auto next = use.getOwner();
      if (isa<tpu::ReshapeOp>(next)) {
        for (auto &use_ : next->getResult(0).getUses()) {
          auto next_ = use_.getOwner();
          auto quantOp = dyn_cast<tpu::QuantOp>(next_);
          if (!quantOp) {
            continue;
          }
          if (quantOp.to() != mode) {
            return false;
          }
        }
      } else {
        auto quantOp = dyn_cast<tpu::QuantOp>(next);
        if (!quantOp) {
          continue;
        }
        if (quantOp.to() != mode) {
          return false;
        }
      }
    }
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto quantOp = cast<OpTy>(op);
    auto prevOp = op->getOperand(0).getDefiningOp();

    auto getEltType = [](Value value){
      auto type = value.getType().template cast<TensorType>();
      return type.getElementType();
    };
    auto fn = op->getParentOfType<FuncOp>();
    auto inputs_type = toupper(clInputsType);
    assert(fn);
    // change the argType of FuncOp
    if (isa<tpu::InputOp>(prevOp) ||
        isa<tpu::InputOp>(prevOp->getOperand(0).getDefiningOp())) {
      if (isa<tpu::ReshapeOp>(prevOp)) {
        prevOp = prevOp->getOperand(0).getDefiningOp();
      }
      // if not all sibliings are same quant mode,
      // change input's mode to BF16
      if (!ifAllSiblingsAreSameQuantMode(prevOp, quantOp.to())) {
        auto argument = prevOp->getOperand(0);
        setOpResultType(argument, FloatType::getBF16(op->getContext()));
        setOpResultType(prevOp->getResult(0), FloatType::getBF16(op->getContext()));
        setOpResultType(op->getOperand(0), FloatType::getBF16(op->getContext()));
        quantOp->setAttr("from", rewriter.getStringAttr("BF16"));
      } else if ((inputs_type == "AUTO" || inputs_type == "INT8" || inputs_type == "SAME") && quantOp.from() == "NONE" &&
                 (quantOp.to() == "INT8" || quantOp.to() == "UINT8")) {
        // remove quantOp and change argType
        // and inputOp's type to int8
        auto argument = prevOp->getOperand(0);
        auto bSigned = (quantOp.to() == "INT8") ? IntegerType::Signed : IntegerType::Unsigned;
        setOpResultType(argument, IntegerType::get(op->getContext(), 8, bSigned));
        setOpResultType(prevOp->getResult(0), IntegerType::get(op->getContext(), 8, bSigned));
        setOpResultType(op->getOperand(0), IntegerType::get(op->getContext(), 8, bSigned));
        updateInputOpNameIfNeeded(rewriter, prevOp, "_quant_i8");
        setOpThreshold(prevOp, (quantOp.to() == "INT8" ? 128 : 256) /
                                quantOp.scale().convertToFloat());
        rewriter.replaceOp(op, {op->getOperand(0)});
      } else if (quantOp.from() == "NONE" &&
                 (quantOp.to() == "UINT16" || quantOp.to() == "INT16")) {
        auto argument = prevOp->getOperand(0);
        if (quantOp.to() == "UINT16") {
          setOpResultType(argument, IntegerType::get(op->getContext(), 16, IntegerType::Unsigned));
          setOpResultType(prevOp->getResult(0), IntegerType::get(op->getContext(), 16, IntegerType::Unsigned));
          setOpResultType(op->getOperand(0), IntegerType::get(op->getContext(), 16, IntegerType::Unsigned));
          updateInputOpNameIfNeeded(rewriter, prevOp, "_quant_u16");
        } else {
          setOpResultType(argument, IntegerType::get(op->getContext(), 16, IntegerType::Signed));
          setOpResultType(prevOp->getResult(0), IntegerType::get(op->getContext(), 16, IntegerType::Signed));
          setOpResultType(op->getOperand(0), IntegerType::get(op->getContext(), 16, IntegerType::Signed));
          updateInputOpNameIfNeeded(rewriter, prevOp, "_quant_i16");
        }
        setOpThreshold(prevOp, 1.0);
        rewriter.replaceOp(op, {op->getOperand(0)});
      } else if (quantOp.from() == "NONE" && quantOp.to() == "BF16" &&
                 (inputs_type == "BF16" || inputs_type == "SAME")) {
        auto argument = prevOp->getOperand(0);
        setOpResultType(argument, FloatType::getBF16(op->getContext()));
        setOpResultType(prevOp->getResult(0), FloatType::getBF16(op->getContext()));
        setOpResultType(op->getOperand(0), FloatType::getBF16(op->getContext()));
        updateInputOpNameIfNeeded(rewriter, prevOp, "_quant_bf16");
        setOpThreshold(prevOp, 1.0);
        rewriter.replaceOp(op, {op->getOperand(0)});
      }

      // make result type of reshapeOp is as some as its' operand.
      auto elementType = getEltType(prevOp->getResult(0));
      for (auto &use : prevOp->getResult(0).getUses()) {
        auto child = use.getOwner();
        if (!isa<tpu::ReshapeOp>(child)) {
          continue;
        }
        setOpResultType(child->getResult(0), elementType);
      }
    } else {
      return failure();
    }

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      arguments.push_back(entryBlock.getArgument(i).getType());
    }
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = rewriter.getFunctionType(
          llvm::ArrayRef<mlir::Type>{arguments},
          llvm::ArrayRef<mlir::Type>{returns});
    fn.setType(fnType);

    return success();
  }
};

struct EliminateOutputQuantOpPattern: public RewritePattern {
  EliminateOutputQuantOpPattern(MLIRContext *context)
      : RewritePattern("tpu.quant", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = getNextOp(op);
    if (!nextOp) {
      return failure();
    }
    if (!isa<ReturnOp>(nextOp)) {
      return failure();
    }
    auto outputs_type = toupper(clOutputsType);
    if (outputs_type == "FP32") {
      return failure();
    }
    auto fn = op->getParentOfType<FuncOp>();
    assert(fn);
    auto quantOp = cast<tpu::QuantOp>(op);
    auto prevOp = quantOp.input().getDefiningOp();
    auto name = getOpName(prevOp);
    auto threshold = getOpThreshold(prevOp);
    bool fixed = false;
    if (outputs_type == "AUTO") {
      if (quantOp.from() == "INT8" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
        fixed = true;
      }
    } else if (outputs_type == "INT8") {
      if (quantOp.from() == "BF16" && quantOp.to() == "NONE" &&
          quantOp.scale().convertToFloat() == 1.0f) {
        auto scale = 128 / threshold;
        std::vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("from", rewriter.getStringAttr("BF16")));
        attrs.push_back(
            rewriter.getNamedAttr("to", rewriter.getStringAttr("INT8")));
        attrs.push_back(
            rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
        std::string new_name = name.str() + "_i8";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        auto eltType = IntegerType::get(rewriter.getContext(), 8);
        auto shape = getTensorShape(quantOp);
        auto type = RankedTensorType::get(shape, eltType);
        std::vector<Value> operands;
        operands.push_back(quantOp.input());
        rewriter.replaceOpWithNewOp<tpu::QuantOp>(
            op, type, ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        fixed = true;
      } else if (quantOp.from() == "INT8" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
        fixed = true;
      }
    } else if (outputs_type == "BF16") {
      if (quantOp.from() == "BF16" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
        fixed = true;
      }
    } else {
      // keep
      if (quantOp.from() == "INT8" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
        fixed = true;
      } else if (quantOp.from() == "BF16" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
        fixed = true;
      }
    }
    if (fixed == false) {
      return failure();
    }

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      arguments.push_back(entryBlock.getArgument(i).getType());
    }
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = rewriter.getFunctionType(
          llvm::ArrayRef<mlir::Type>{arguments},
          llvm::ArrayRef<mlir::Type>{returns});
    fn.setType(fnType);

    return success();
  }
};

template <typename OpTy>
struct EliminateUselessQuantOpPattern: public RewritePattern {
  EliminateUselessQuantOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto quantOp = cast<OpTy>(op);
    if (quantOp.from() != quantOp.to()) {
      return failure();
    }
    rewriter.replaceOp(op, {op->getOperand(0)});
    return success();
  }
};

template <typename OpTy>
struct EliminateOutputReshapeOpPattern: public RewritePattern {
  EliminateOutputReshapeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = getNextOp(op);
    if (!nextOp) {
      return failure();
    }
    if (!isa<ReturnOp>(nextOp)) {
      return failure();
    }
    auto fn = op->getParentOfType<FuncOp>();
    assert(fn);

    rewriter.replaceOp(op, {op->getOperand(0)});

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      arguments.push_back(entryBlock.getArgument(i).getType());
    }
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = rewriter.getFunctionType(
          llvm::ArrayRef<mlir::Type>{arguments},
          llvm::ArrayRef<mlir::Type>{returns});
    fn.setType(fnType);

    return success();
  }
};

static void storeQscaleTableToFile(FuncOp fn, MLIRContext *ctx) {
  std::string tableName = "_qscale_table.txt";
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> table;
  table = openOutputFile(tableName, &errorMessage);
  if (!table) {
    llvm_unreachable(errorMessage.c_str());
  }

  float qscale = 1.0f;
  int32_t zero_point = 0;
  auto &os = table->os();
  fn.walk([&](Operation *op) {
    if (auto castOp = llvm::dyn_cast<tpu::InputOp>(op)) {
      qscale = 1.0f;
      float threshold =
          (float)castOp.quant().threshold().getValue().convertToFloat();
      if (threshold != 0) {
        auto elementType = castOp.getResult().getType().template
                        cast<TensorType>().getElementType();
        int max_val = elementType.isUnsignedInteger(8) ? 255 : 128;
        qscale = max_val / threshold;
      }
      char qscale_str[64] = {0};
      sprintf(qscale_str, "%.12f", qscale);
      zero_point = 0;
      os << castOp.name() << " " << qscale_str << " "<< zero_point << "\n";
    } else if (auto castOp = llvm::dyn_cast<ReturnOp>(op)) {
      for (int i = 0; i < (int)op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        if (isa<tpu::QuantOp>(opd)) {
          opd = opd->getOperand(0).getDefiningOp();
        }
        if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(opd)) {
          float threshold = tpuOp.getOpQuantThreshold();
          qscale = (threshold == 0) ? 1.0f : (threshold / 128.0);
          os << getOpName(opd) << " " << std::to_string(qscale) << " "
             << zero_point << "\n";
        }
      }
    }
  });

  fn->setAttr("qscale_table", Builder(ctx).getStringAttr(tableName));
  table->keep();
}

class TpuLowerPass : public mlir::PassWrapper<TpuLowerPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    patterns.insert<
        EliminateInputQuantOpPattern<tpu::QuantOp>,
        EliminateOutputQuantOpPattern,
        EliminateOutputReshapeOpPattern<tpu::ReshapeOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        EliminateUselessQuantOpPattern<tpu::QuantOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    storeQscaleTableToFile(fn, context);

    // first, merge conv rshift/multiplier/bias into one packed tensor
    patterns.clear();
    patterns.insert<
        PackWeightConv2DOpPattern<tpu::Conv2DOp>,
        PackWeightConv2DOpPattern<tpu::DeConv2DOp>,
        PackWeightBroadcastMulOpPattern
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // second, do weight lower on weight tensors
    // lower means transpose and save as storageType (int8/bf16,etc)
    patterns.clear();
    patterns.insert<
        LowerWeightConv2DOpPattern<tpu::Conv2DOp>,
        LowerWeightConv2DOpPattern<tpu::DeConv2DOp>,
        LowerWeightConv3DOpPattern<tpu::Conv3DOp>,
        LowerWeightFullyConnectedOpPattern,
        LowerWeightDetectionOutputOpPattern,
        LowerWeightInstanceNormOpPattern,
        LowerWeightRNNOpPattern<tpu::GruOp>,
        LowerWeightRNNOpPattern<tpu::LstmOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // common lower, make sure all weight lowered
    fn.walk([&](Operation *op) {
      if (isa<tpu::LoadWeightOp>(op)) {
        lowerWeightGeneric(op);
      }
    });

    // do cpu op lowering
    patterns.clear();
    patterns.insert<
        LowerCpuOpDefaultPattern<tpu::DetectionOutputOp>,
        LowerCpuOpDefaultPattern<tpu::FrcnDetectionOp>,
        LowerCpuOpDefaultPattern<tpu::ProposalOp>,
        LowerCpuOpDefaultPattern<tpu::RetinaFaceDetectionOp>,
        LowerCpuOpDefaultPattern<tpu::ROIPoolingOp>,
        LowerCpuOpDefaultPattern<tpu::YoloDetectionOp>,
        LowerCpuOpDefaultPattern<tpu::SoftmaxCpuOp>,
        LowerCustomOpPattern<tpu::CustomOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // do op lower
    patterns.clear();
    patterns.insert<
        DefaultToTGPattern<tpu::AbsOp>,
        DefaultToTGPattern<tpu::ArgMaxOp>,
        DefaultToTGPattern<tpu::BroadcastMulOp>,
        DefaultToTGPattern<tpu::BroadcastAddOp>,
        DefaultToTGPattern<tpu::BroadcastSubOp>,
        DefaultToTGPattern<tpu::ClipOp>,
        DefaultToTGPattern<tpu::ConcatOp>,
        DefaultToTGPattern<tpu::Conv2DOp>,
        DefaultToTGPattern<tpu::Conv3DOp>,
        DefaultToTGPattern<tpu::ConvFcOp>,
        DefaultToTGPattern<tpu::CropOp>,
        DefaultToTGPattern<tpu::DeConv2DOp>,
        DefaultToTGPattern<tpu::DilateOp>,
        DefaultToTGPattern<tpu::EltwiseAddOp>,
        DefaultToTGPattern<tpu::MulConstOp>,
        DefaultToTGPattern<tpu::EltwiseMaxOp>,
        DefaultToTGPattern<tpu::EltwiseMinOp>,
        DefaultToTGPattern<tpu::EltwiseMulOp>,
        DefaultToTGPattern<tpu::EmbeddingOp>,
        DefaultToTGPattern<tpu::FullyConnectedOp>,
        DefaultToTGPattern<tpu::InterpOp>,
        DefaultToTGPattern<tpu::InstanceNormOp>,
        DefaultToTGPattern<tpu::LayerNormOp>,
        DefaultToTGPattern<tpu::LrnOp>,
        DefaultToTGPattern<tpu::LeakyReluOp>,
        DefaultToTGPattern<tpu::MishOp>,
        DefaultToTGPattern<tpu::PadOp>,
        DefaultToTGPattern<tpu::PermuteOp>,
        DefaultToTGPattern<tpu::PixelShuffleOp>,
        DefaultToTGPattern<tpu::PoolAvg2DOp>,
        DefaultToTGPattern<tpu::PoolMax2DOp>,
        DefaultToTGPattern<tpu::PoolMax3DOp>,
        DefaultToTGPattern<tpu::PoolMaskOp>,
        DefaultToTGPattern<tpu::PReluOp>,
        DefaultToTGPattern<tpu::PowOp>,
        DefaultToTGPattern<tpu::QuantOp>,
        DefaultToTGPattern<tpu::ReluOp>,
        DefaultToTGPattern<tpu::ReorgOp>,
        DefaultToTGPattern<tpu::ReverseOp>,
        DefaultToTGPattern<tpu::ReflectionPadOp>,
        DefaultToTGPattern<tpu::ScaleLutOp>,
        DefaultToTGPattern<tpu::ShuffleChannelOp>,
        DefaultToTGPattern<tpu::SigmoidOp>,
        DefaultToTGPattern<tpu::SwishOp>,
        DefaultToTGPattern<tpu::SwapChannelOp>,
        DefaultToTGPattern<tpu::TanHOp>,
        DefaultToTGPattern<tpu::LogOp>,
        DefaultToTGPattern<tpu::EluOp>,
        DefaultToTGPattern<tpu::ExpOp>,
        DefaultToTGPattern<tpu::TileOp>,
        DefaultToTGPattern<tpu::StdOp>,
        DefaultToTGPattern<tpu::UpsampleOp>,
        DefaultToTGPattern<tpu::ReduceL2Op>,
        DefaultToTGPattern<tpu::ReduceMeanOp>,
        DefaultToTGPattern<tpu::ReduceMaxOp>,
        DefaultToTGPattern<tpu::ReduceMinOp>,
        DefaultToTGPattern<tpu::ReduceSumOp>,
        DefaultToTGPattern<tpu::ReduceL2Op>,
        DefaultToTGPattern<tpu::GruOp>,
        DefaultToTGPattern<tpu::LstmOp>,
        DefaultToTGPattern<tpu::SoftmaxOp>,
        DefaultToTGPattern<tpu::SoftPlusOp>,
        DefaultToTGPattern<tpu::QuadraticSumOp>,
        DefaultToTGPattern<tpu::CscOp>,
        DefaultToTGPattern<tpu::MatMulOp>,
        DefaultToTGPattern<tpu::ZeroMaskOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
    LLVM_DEBUG(llvm::errs() << "Done lower: " << "]\n");

    // check if every one is not lowered
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect()->getNamespace() != "tpu"
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
        std::string opName = op->getName().getStringRef().str();
        llvm_unreachable(("lower didn't handle " + opName).c_str());
      }
    });

    // TODO: this is temporary
    // fold reshape
    patterns.clear();
    patterns.insert<
        FoldReshapePattern<tpu::ReshapeOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

  }
};

std::unique_ptr<mlir::Pass> createTpuLowerPass() {
  return std::make_unique<TpuLowerPass>();
}

} // namespace mlir
