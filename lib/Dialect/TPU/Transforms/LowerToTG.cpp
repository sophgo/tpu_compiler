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

llvm::cl::opt<bool>
    clUseTPUQuantOp("use-tpu-quant-op",
                llvm::cl::desc("Quant op inference by tpu instead of cpu"),
                llvm::cl::init(true));

llvm::cl::opt<bool> clDequantResultsToFp32(
    "dequant-results-to-fp32",
    llvm::cl::desc("Dequant all outputs of network from int8 to fp32"),
    llvm::cl::init(true));

llvm::cl::opt<bool> clQuantInputsToInt8(
    "quant-inputs-to-int8",
    llvm::cl::desc("Quant all inputs of network from fp32 to int8"),
    llvm::cl::init(false));

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

  std::vector<Value> operands;
  operands.push_back(input());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    llvm_unreachable("int8 is not supported for argmax\n");
  } else if (getOpQuant() == "BF16") {
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

    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));

    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(
        op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
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
  assert(this->axis() == 1);
  int64_t n, c, h, w, bn, bc, bh, bw;
  auto shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);
  auto bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);
  assert(bn == n || bn == 1);

  if (bh == 1 && bw == 1) {
    // convert to scale op
    std::vector<Value> operands;
    operands.push_back(input());
    operands.push_back(multiplier());
    // This is a little tricky, as there is no bias() operand to reuse
    // we reuse the quant_rshift() to carry the packed per-channel info
    operands.push_back(quant_rshift());

    std::vector<NamedAttribute> attrs;
    // only do_relu is useful for now
    attrs.push_back(builder.getNamedAttr(
        "param",
        tpu::ConvParam::get(
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
            builder.getBoolAttr(this->do_relu()),           // do_relu
            builder.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
            builder.getI32IntegerAttr(0),                   // pad_value
            builder.getContext())));
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
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
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
  } else if (bc == 1 && bh == h && bw == w) {
    std::vector<Value> operands;
    operands.push_back(op->getOperand(0));
    operands.push_back(op->getOperand(1));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", nameAttr()));
    attrs.push_back(
        builder.getNamedAttr("do_relu", builder.getBoolAttr(do_relu())));

    if (getOpQuant() == "INT8") {
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
      int8_t rshift_i8 = static_cast<int8_t>(rshift->at(0));
      std::vector<int32_t> m_i8_inputs(2,1);
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
  if (isa<tpu::SliceOp>(op)) {
    auto castOp = cast<tpu::SliceOp>(op);
    auto ax = castOp.axis();
    auto shape = getTensorShape(op->getOperand(0));
    for (uint32_t i = 0; i< ax; i++) {
      if (shape[i] != 1) {
        return false;
      }
    }
    return true;
  }
  return false;
}

Value tpu::ConcatOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);
  assert(wTF);
  bool relu = do_relu();
  bool only_merge = !relu; // just merge input data
  const unsigned nInputs = this->getNumInputs();
  std::vector<Value> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
    if (only_merge == true) {
      if (is_fused_op(op->getOperand(i).getDefiningOp())) {
        only_merge = false;
      }
    }
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
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
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),wTF);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      std::vector<int32_t> m_rshift_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
        m_rshift_array[i] = static_cast<int32_t>(rshift->at(i));
        if (only_merge == true) {
          if (m_i8_inputs_array[i] !=1 || m_rshift_array[i] != 0) {
            only_merge = false;
          }
        }
      }
      if (only_merge == false) {
        attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
        attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(relu)));
        attrs.push_back(builder.getNamedAttr("m_i8_inputs",
            builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
        attrs.push_back(builder.getNamedAttr("rshift",
            builder.getI32ArrayAttr(ArrayRef<int32_t>({m_rshift_array}))));
        // create op
        auto newOp = OpBuilder(op).create<tpu::TG_INT8_ConcatOp>(op->getLoc(),
            getResult().getType(), ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        return newOp.getResult();
      } else {
        auto newOp = OpBuilder(op).create<tpu::TG_ConcatNOp>(op->getLoc(),
            getResult().getType(), ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        return newOp.getResult();
      }
  } else if (getOpQuant() == "BF16") {
    attrs.push_back(builder.getNamedAttr("axis", axisAttr()));
    attrs.push_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(relu)));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ConcatOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
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
          tpu::ConvParam::get(
              param().stride_h(),
              param().stride_w(),
              param().padding(),
              param().dilation_h(),
              param().dilation_w(),
              builder.getI32IntegerAttr(pad_t),
              builder.getI32IntegerAttr(pad_b),
              builder.getI32IntegerAttr(pad_l),
              builder.getI32IntegerAttr(pad_r),
              param().group(),
              param().is_dw(),
              param().with_bias(),
              param().do_relu(),
              param().ins(),
              param().pad_value(),
              builder.getContext())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_Conv2DOp>(op->getLoc(),
          getResult().getType(), ArrayRef<Value>{operands},
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
          getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
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
              param().do_relu(),
              param().ins(),
              builder.getContext())));
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
  attrs.push_back(builder.getNamedAttr("crop_shape", crop_shapeAttr()));
  attrs.push_back(builder.getNamedAttr("crop_offset", crop_offsetAttr()));

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

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_DeConv2DOp>(
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
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
          op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
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

      bool is_asymmetric = isOpQuantAsymmetric();
      if (is_asymmetric){
        const unsigned nInputs = op->getNumOperands() - 4;
        std::vector<int> input_offset(nInputs, 0);
        for(size_t i = 0; i < nInputs; ++i){
          input_offset.at(i) = -getPreviousOpZeroPoint(op, i);
        }
        attrs.push_back(builder.getNamedAttr(
              "input_offset",
              builder.getI32ArrayAttr(ArrayRef<int32_t>({input_offset}))));
        attrs.push_back(
            builder.getNamedAttr("output_offset",
              builder.getI32IntegerAttr(getOpZeroPoint(op))));
      }
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

  if (getOpQuant() == "INT8") {
    int8_t rshift_i8 = 0;
    int32_t m_i32_output = 0x7FFFFFFF;
    if (getOpQuantParamType() == "RSHIFT_AND_M_I32") {
      // MUL
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
      assert(rshift->size() == 1);
      rshift_i8 = static_cast<int8_t>(rshift->at(0));

      // m_i8_output
      auto multiplier =
          readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
      assert(multiplier->size() == 1);
      m_i32_output = static_cast<int32_t>(multiplier->at(0));
    }

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
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("do_relu",
      builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), wTF);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr(
        "mutliplier",
        builder.getI32IntegerAttr(static_cast<int32_t>(multiplier->at(0)))));
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
    bool is_asymmetric = isOpQuantAsymmetric();
    if (is_asymmetric) {
      attrs.push_back(builder.getNamedAttr(
          "input_offset",
          builder.getI32IntegerAttr(-getPreviousOpZeroPoint(op))));
      attrs.push_back(builder.getNamedAttr(
          "output_offset", builder.getI32IntegerAttr(getOpZeroPoint(op))));
    }
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
  attrs.push_back(builder.getNamedAttr("order0", order0Attr()));
  attrs.push_back(builder.getNamedAttr("order1", order1Attr()));
  attrs.push_back(builder.getNamedAttr("order2", order2Attr()));
  attrs.push_back(builder.getNamedAttr("order3", order3Attr()));

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

Value tpu::PowerOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //Operation *op = this->getOperation();
  //auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

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

Value tpu::ZeroMaskOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8" || getOpQuant() == "UINT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ZeroMaskOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ZeroMaskOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

static bool is_fmt_support(llvm::StringRef fmt_str) {
  return (fmt_str == "INT8" || fmt_str == "NONE" || fmt_str == "UINT8" ||
          fmt_str == "BF16" || fmt_str == "FP32");
}

Value tpu::QuantOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  auto parentOp = this->getOperand().getDefiningOp();
  if ((isa<tpu::InputOp>(parentOp) && from() != "UINT8") ||
       false == is_fmt_support(from()) ||
       false == is_fmt_support(to())) {
    std::vector<NamedAttribute> param;
    param.push_back(builder.getNamedAttr("from", fromAttr()));
    param.push_back(builder.getNamedAttr("to", toAttr()));
    param.push_back(builder.getNamedAttr("scale", scaleAttr()));
    param.push_back(builder.getNamedAttr("zero_point", zero_pointAttr()));
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
    attrs.push_back(builder.getNamedAttr("zero_point", zero_pointAttr()));
    auto newOp = OpBuilder(op).create<tpu::TG_QuantOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
}

Value tpu::ReQuantOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.push_back(input());

  int input_offset = -getPreviousOpZeroPoint(op);
  int output_offset = getOpZeroPoint(op);

  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr(
      "input_offset", builder.getI32IntegerAttr(input_offset)));
  attrs.push_back(builder.getNamedAttr(
      "output_offset", builder.getI32IntegerAttr(output_offset)));
  attrs.push_back(
      builder.getNamedAttr("qscale", qscaleAttr()));

  auto newOp = OpBuilder(op).create<tpu::TG_ReQuantOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tpu::ReciprocalOp::convertToTG() {
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
        builder.getNamedAttr("method", builder.getStringAttr("mantissa")));
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LutOp>(
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

  if (pixel_format == "YUV420_PLANAR") {
    assert(getOpQuant() == "INT8" || getOpQuant() == "UINT8");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_Yuv420CscOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else {
    assert(getOpQuant() == "INT8" || getOpQuant() == "UINT8");
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t input_size, n, c, h, w;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
    getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
    getNCHW(input_shape, n, c, h, w);
    getNCHW(output_shape, on, oc, oh, ow);

    int unaligned_w = (int)(oc * oh * ow / (c * h));
    std::vector<int> crop_shape{(int)n, (int)c, (int)h, unaligned_w};
    std::vector<int> crop_offset{0, 0, 0, 0};
    attrs.push_back(
        builder.getNamedAttr("crop_shape", builder.getI32ArrayAttr(crop_shape)));
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
  attrs.push_back(builder.getNamedAttr("resp", respAttr()));

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

Value tpu::TileInterpOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;

  // keep info to tg
  attrs.push_back(builder.getNamedAttr("resp",
        builder.getArrayAttr(resp().getValue())));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_TileInterpOp>(
        op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_TileInterpOp>(
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

  if (this->mode().str() == "DCR") {
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
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
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
  attrs.push_back(builder.getNamedAttr("added_offset",
      builder.getBoolAttr(added_offset())));

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

Value tpu::SliceOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axis",
      builder.getI32IntegerAttr(axis())));
  attrs.push_back(builder.getNamedAttr("offset", offsetAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SliceOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SliceOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  llvm_unreachable("unsupported type");
}

Value tpu::SqrtOp::convertToTG() {
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
  attrs.push_back(builder.getNamedAttr("added_offset",
      builder.getBoolAttr(added_offset())));

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
  attrs.push_back(builder.getNamedAttr("added_offset",
      builder.getBoolAttr(added_offset())));

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
  attrs.push_back(builder.getNamedAttr("added_offset",
      builder.getBoolAttr(added_offset())));

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
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto castOp = cast<ReduceL2Op>(op);

  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  operands.push_back(input());
  //std::vector<Value> operands(op->getOperands().begin(),
  //    op->getOperands().end());

  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
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

  // TODO: tpu support
  auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
          castOp.getResult().getType(), ArrayRef<Value>{operands},
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
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

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
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

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

Value tpu::GruOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value> operands;
  const int nInputs =  9;
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
  auto castOp = cast<tpu::LstmOp>(op);
  //  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<Value> operands;
  const int nInputs =  !isTensorNone(castOp.bias()) ? 10 : 9;
  //input + weight + recurrence + bias? + initial_h
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("bidirectional", bidirectionalAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LstmOp>(op->getLoc(),
        getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
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
  const int nInputs =  3;
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

Value tpu::SquareOp::convertToTG() {
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  assert(getOpQuant() == "BF16");
  auto newOp = OpBuilder(op).create<tpu::TG_BF16_SquareOp>(op->getLoc(),
      getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
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
  LLVM_DEBUG(llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(op->getOperand(1));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  assert(getOpQuant() == "BF16");
  auto newOp = OpBuilder(op).create<tpu::TG_BF16_MatMulOp>(op->getLoc(),
      getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
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

  return std::move(packed);
}

template <typename OpTy>
struct PackWeightConv2DOpPattern : public RewritePattern {
  PackWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    if (getOpQuant(op) != "INT8" || !isOpQuantPerchannel(op)
        || getOpQuantParamType(op) != "RSHIFT_AND_M_I32") {
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
      LLVM_DEBUG(llvm::errs() << "Pack Weight for BroadcastMul ONLY apply INT8 we skip it\n";);
      return failure();
    }

    int64_t bn, bc, bh, bw;
    auto bshape = getTensorShape(op->getOperand(1));
    getNCHW(bshape, bn, bc, bh, bw);
    if (bh != 1 || bw != 1) {
      return failure();
    }

    // after quantizeInt8, the quantparam is "RSHIFT_AND_M_I32"
    auto rshiftOp = cast<tpu::LoadWeightOp>(castOp.quant_rshift().getDefiningOp());
    if (rshiftOp.lowered()) {
      // packed already
      return failure();
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
    rshiftOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    castOp.setOperand(5, NoneOp);

    setOpQuantParamType(op, "RSHIFT_AND_M_I32");
    return success();
  }
};

template<typename T>
static void rotateConvolutionFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
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
  if (kh == 1 || kw == 1) {
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
    std::vector<int64_t> &s) {
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

template <typename OpTy>
struct LowerWeightConv2DOpPattern : public RewritePattern {
  LowerWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(convOp.filter().getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return failure();
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
        if (isa<tpu::DeConv2DOp>(op))
          rotateConvolutionFilter<int8_t>(filter_int8, shape);
        transposeConvolutionFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
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
          biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
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
          biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
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
        if (isa<tpu::DeConv2DOp>(op))
          rotateConvolutionFilter<uint16_t>(filter_bf16, shape);
        transposeConvolutionFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
        assert(biasOp.storage() == "FP32");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
        //Split into high/low part
        std::vector<uint16_t> bias_fp32_high;
        std::vector<uint16_t> bias_fp32_low;
        size_t sz = bias->size();
        LLVM_DEBUG(llvm::errs() << "Lower bias for Conv2D size : "
                            << sz << "\n";);
        float *biasFloatPtr = bias->data();
        for (size_t i = 0; i < sz; ++i) {
          unsigned short *temp_short_ptr = reinterpret_cast<unsigned short *>(biasFloatPtr + i);
          bias_fp32_high.push_back(temp_short_ptr[1]);
          bias_fp32_low.push_back(temp_short_ptr[0]);
        }
        std::vector<uint16_t> bias_reshape_fp32;
        bias_reshape_fp32.reserve(2 * sz);
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(), bias_fp32_high.end());
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(), bias_fp32_low.end());
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_reshape_fp32.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FP32 and transpose, this is not FP32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(convOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return success();
  }
};

template <typename OpTy>
struct LowerWeightConv3DOpPattern : public RewritePattern {
  LowerWeightConv3DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(convOp.filter().getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return failure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for Conv3D: "
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
        if (isa<tpu::DeConv2DOp>(op))
          rotateConvolutionFilter<int8_t>(filter_int8, shape);
        transposeConvolutionFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
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
          biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
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
          biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
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

        assert(shape.size() == 5 || shape.size() == 6);
        transposeConvolution3dFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias().getDefiningOp());
        assert(biasOp.storage() == "FP32");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
        //Split into high/low part
        std::vector<uint16_t> bias_fp32_high;
        std::vector<uint16_t> bias_fp32_low;
        size_t sz = bias->size();
        LLVM_DEBUG(llvm::errs() << "Lower bias for Conv3D size : "
                            << sz << "\n";);
        float *biasFloatPtr = bias->data();
        for (size_t i = 0; i < sz; ++i) {
          unsigned short *temp_short_ptr = reinterpret_cast<unsigned short *>(biasFloatPtr + i);
          bias_fp32_high.push_back(temp_short_ptr[1]);
          bias_fp32_low.push_back(temp_short_ptr[0]);
        }
        std::vector<uint16_t> bias_reshape_fp32;
        bias_reshape_fp32.reserve(2 * sz);
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(), bias_fp32_high.end());
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(), bias_fp32_low.end());
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_reshape_fp32.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FP32 and transpose, this is not FP32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(convOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return success();
  }
};

struct LowerWeightFullyConnectedOpPattern : public RewritePattern {
  LowerWeightFullyConnectedOpPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    auto filterOp =
      llvm::dyn_cast_or_null<tpu::LoadWeightOp>(fcOp.filter().getDefiningOp());

    // filterOp is null means rhs is activation
    if (filterOp && filterOp.lowered()) {
      // lowered already
      return failure();
    }

    if (!filterOp) {
      // rhs activation case
      if (isTensorNone(fcOp.bias())) {
          return failure();
      }

      auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias().getDefiningOp());
      if (biasOp.lowered()) {
          return failure();
      }
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for FullyConnectedOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      if (filterOp)
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
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias().getDefiningOp());
        // per-tensor mode, bias is INT32
        assert(biasOp.storage() == "INT32");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
        std::vector<int32_t> bias_int32(bias->begin(), bias->end());
        transposeBiasInt32(bias_int32);
        std::vector<uint32_t> bias_uint32(size);
        memcpy(bias_uint32.data(), bias_int32.data(), size * sizeof(int32_t));

        // save it
        // after transpose, this is not INT32 anymore, it is 2 stripes of UINT8
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape.
        addWeightTensorAndUpdateWeightOp<uint32_t>(fcOp.bias(),
            "lowered", bias_uint32, shape, "UINT32", wTF);
        biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      if (filterOp)
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
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias().getDefiningOp());
        assert(biasOp.storage() == "FP32");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
        //Split into high/low part
        std::vector<uint16_t> bias_fp32_high;
        std::vector<uint16_t> bias_fp32_low;
        size_t sz = bias->size();
        LLVM_DEBUG(llvm::errs() << "Lower bias for Conv2D size : "
                            << sz << "\n";);
        float *biasFloatPtr = bias->data();
        for (size_t i = 0; i < sz; ++i) {
          unsigned short *temp_short_ptr = reinterpret_cast<unsigned short *>(biasFloatPtr + i);
          bias_fp32_high.push_back(temp_short_ptr[1]);
          bias_fp32_low.push_back(temp_short_ptr[0]);
        }
        std::vector<uint16_t> bias_reshape_fp32;
        bias_reshape_fp32.reserve(2 * sz);
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(), bias_fp32_high.end());
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(), bias_fp32_low.end());
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_reshape_fp32.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FP32 and transpose, this is not FP32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(fcOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return success();
  }
};

struct LowerWeightGruOpPattern : public RewritePattern {
  LowerWeightGruOpPattern(MLIRContext *context)
      : RewritePattern("tpu.gru", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto gruOp = cast<tpu::GruOp>(op);
    auto weightOp = cast<tpu::LoadWeightOp>(gruOp.weight().getDefiningOp());
    auto recurrenceOp =
        cast<tpu::LoadWeightOp>(gruOp.recurrence().getDefiningOp());

    if (weightOp.lowered()) {
      // lowered already
      return failure();
    }
    if (getOpQuant(op) != "BF16") {
      return failure();
    }
    std::vector<int64_t> shape;
    int64_t size;
    LLVM_DEBUG(llvm::errs()
                   << "Lower Weight for GruOp: " << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    // lower weight
    getTensorShapeAndSize(gruOp.weight(), shape, size);
    assert(shape.size() == 3);
    int64_t hidden_size = shape[1] / 3;
    int64_t input_size = shape[2];
    auto filter = readAndDeleteWeightTensor<float>(gruOp.weight(), wTF);
    std::vector<uint16_t> filter_bf16(size);
    FloatToBFloat16(filter->data(), filter_bf16.data(), size);
    uint16_t *p_data = filter_bf16.data();
    for (int i = 0; i < 3; i++) {
      transpose_row_col(p_data, hidden_size, input_size);
      p_data += hidden_size * input_size;
    }
    shape[1] = 3 * input_size;
    shape[2] = hidden_size;
    addWeightTensorAndUpdateWeightOp<uint16_t>(gruOp.weight(), "lowered",
                                               filter_bf16, shape, "BF16", wTF);
    weightOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // lower recurrence
    getTensorShapeAndSize(gruOp.recurrence(), shape, size);
    assert(shape.size() == 3);
    assert(shape[1] == 3 * hidden_size);
    assert(shape[2] == hidden_size);
    auto r_data = readAndDeleteWeightTensor<float>(gruOp.recurrence(), wTF);
    std::vector<uint16_t> r_bf16(size);
    FloatToBFloat16(r_data->data(), r_bf16.data(), size);
    p_data = r_bf16.data();
    for (int i = 0; i < 3; i++) {
      transpose_row_col(p_data, hidden_size, hidden_size);
      p_data += hidden_size * hidden_size;
    }
    addWeightTensorAndUpdateWeightOp<uint16_t>(gruOp.recurrence(), "lowered",
                                               r_bf16, shape, "BF16", wTF);
    recurrenceOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // lower bias
    if (!isTensorNone(gruOp.bias())) {
      auto biasOp = cast<tpu::LoadWeightOp>(gruOp.bias().getDefiningOp());
      // NOTE: for 1880v2, bias is fp32, rather than bf16
      // however, for simplicity, in quantizeBf16, we quantize all tensor into
      // bf16 before lowering to hardware, we need to expand the bf16 to fp32
      // first then transpose into 2 stripes of uint16_t
      getTensorShapeAndSize(gruOp.bias(), shape, size);
      auto bias = readAndDeleteWeightTensor<float>(gruOp.bias(), wTF);
      // Split into high/low part
      std::vector<uint16_t> bias_fp32_high;
      std::vector<uint16_t> bias_fp32_low;
      size_t sz = bias->size();
      LLVM_DEBUG(llvm::errs()
                     << "Lower bias for Conv2D size : " << sz << "\n";);
      float *biasFloatPtr = bias->data();
      for (size_t i = 0; i < sz; ++i) {
        unsigned short *temp_short_ptr =
            reinterpret_cast<unsigned short *>(biasFloatPtr + i);
        bias_fp32_high.push_back(temp_short_ptr[1]);
        bias_fp32_low.push_back(temp_short_ptr[0]);
      }
      std::vector<uint16_t> bias_reshape_fp32;
      bias_reshape_fp32.reserve(2 * sz);
      bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(),
                               bias_fp32_high.end());
      bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                               bias_fp32_low.end());
      // then copy into uint32_t
      std::vector<uint32_t> bias_uint32(sz);
      memcpy(bias_uint32.data(), bias_reshape_fp32.data(),
             sz * sizeof(uint32_t));

      // save it
      // after expand to FP32 and transpose, this is not FP32 anymore
      // it is 2 stripes of UINT16(BF16)
      // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
      // to change the shape
      StringRef storageType = "UINT32";
      addWeightTensorAndUpdateWeightOp<uint32_t>(
          gruOp.bias(), "lowered", bias_uint32, shape, storageType, wTF);
      biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
    }

    // lower initial_h
    if (!isTensorNone(gruOp.initial_h())) {
      auto initial_hOp =
          cast<tpu::LoadWeightOp>(gruOp.initial_h().getDefiningOp());
      getTensorShapeAndSize(gruOp.initial_h(), shape, size);
      auto h_data = readAndDeleteWeightTensor<float>(gruOp.initial_h(), wTF);
      std::vector<uint16_t> h_bf16(size);
      FloatToBFloat16(h_data->data(), h_bf16.data(), size);
      addWeightTensorAndUpdateWeightOp<uint16_t>(gruOp.initial_h(), "lowered",
                                                 h_bf16, shape, "BF16", wTF);
      initial_hOp->setAttr("lowered", rewriter.getBoolAttr(true));
    }

    // lower sigmoid table
    auto tableOp = cast<tpu::LoadWeightOp>(gruOp.getOperand(5).getDefiningOp());
    auto tableSlopeOp =
        cast<tpu::LoadWeightOp>(gruOp.getOperand(6).getDefiningOp());
    // lower filter
    assert(tableOp.storage() == "BF16");
    assert(tableSlopeOp.storage() == "BF16");
    getTensorShapeAndSize(gruOp.sigmoid_table(), shape, size);
    auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
    auto table_slope = readAndDeleteWeightTensor<float>(tableSlopeOp, wTF);
    std::vector<uint16_t> table_uint16(table->begin(), table->end());
    std::vector<uint16_t> table_slope_uint16(table_slope->begin(),
                                             table_slope->end());
    // save it
    addWeightTensorAndUpdateWeightOp<uint16_t>(tableOp, "lowered", table_uint16,
                                               shape, "BF16", wTF);
    tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
    addWeightTensorAndUpdateWeightOp<uint16_t>(
        tableSlopeOp, "lowered", table_slope_uint16, shape, "BF16", wTF);
    tableSlopeOp->setAttr("lowered", rewriter.getBoolAttr(true));

    // lower tanh  table
    tableOp = cast<tpu::LoadWeightOp>(gruOp.getOperand(7).getDefiningOp());
    tableSlopeOp = cast<tpu::LoadWeightOp>(gruOp.getOperand(8).getDefiningOp());

    // lower filter
    assert(tableOp.storage() == "BF16");
    assert(tableSlopeOp.storage() == "BF16");
    getTensorShapeAndSize(gruOp.tanh_table(), shape, size);
    table = readAndDeleteWeightTensor<float>(tableOp, wTF);
    table_slope = readAndDeleteWeightTensor<float>(tableSlopeOp, wTF);
    table_uint16.assign(table->begin(), table->end());
    table_slope_uint16.assign(table_slope->begin(), table_slope->end());

    // save it
    addWeightTensorAndUpdateWeightOp<uint16_t>(tableOp, "lowered", table_uint16,
                                               shape, "BF16", wTF);
    tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
    addWeightTensorAndUpdateWeightOp<uint16_t>(
        tableSlopeOp, "lowered", table_slope_uint16, shape, "BF16", wTF);
    tableSlopeOp->setAttr("lowered", rewriter.getBoolAttr(true));
    return success();
  }
};

struct LowerWeightLstmOpPattern : public RewritePattern {
  LowerWeightLstmOpPattern(MLIRContext *context)
      : RewritePattern("tpu.lstm", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto lstmOp = cast<tpu::LstmOp>(op);
    auto weightOp = cast<tpu::LoadWeightOp>(lstmOp.weight().getDefiningOp());
    auto recurrenceOp =
        cast<tpu::LoadWeightOp>(lstmOp.recurrence().getDefiningOp());
    auto initial_hOp =
        cast<tpu::LoadWeightOp>(lstmOp.initial_h().getDefiningOp());
    auto initial_cOp =
        cast<tpu::LoadWeightOp>(lstmOp.initial_c().getDefiningOp());

    if (weightOp.lowered()) {
      // lowered already
      return failure();
    }
    LLVM_DEBUG(llvm::errs()
                   << "Lower Weight for LstmOp: " << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(weightOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.weight(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(lstmOp.weight(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 3);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            lstmOp.weight(), "lowered", filter_bf16, shape, storageType, wTF);
        weightOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower recurrence
      {
        assert(recurrenceOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.recurrence(), shape, size);
        auto filter =
            readAndDeleteWeightTensor<bfloat16>(lstmOp.recurrence(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 3);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(lstmOp.recurrence(),
                                                   "lowered", filter_bf16,
                                                   shape, storageType, wTF);
        recurrenceOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      // lower bias
      if (!isTensorNone(lstmOp.bias())) {
        auto biasOp = cast<tpu::LoadWeightOp>(lstmOp.bias().getDefiningOp());
        assert(biasOp.storage() == "FP32");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into
        // bf16 before lowering to hardware, we need to expand the bf16 to fp32
        // first then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(lstmOp.bias(), wTF);
        // Split into high/low part
        std::vector<uint16_t> bias_fp32_high;
        std::vector<uint16_t> bias_fp32_low;
        size_t sz = bias->size();
        LLVM_DEBUG(llvm::errs()
                       << "Lower bias for Conv2D size : " << sz << "\n";);
        float *biasFloatPtr = bias->data();
        for (size_t i = 0; i < sz; ++i) {
          unsigned short *temp_short_ptr =
              reinterpret_cast<unsigned short *>(biasFloatPtr + i);
          bias_fp32_high.push_back(temp_short_ptr[1]);
          bias_fp32_low.push_back(temp_short_ptr[0]);
        }
        std::vector<uint16_t> bias_reshape_fp32;
        bias_reshape_fp32.reserve(2 * sz);
        bias_reshape_fp32.insert(bias_reshape_fp32.end(),
                                 bias_fp32_high.begin(), bias_fp32_high.end());
        bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                                 bias_fp32_low.end());
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_reshape_fp32.data(),
               sz * sizeof(uint32_t));

        // save it
        // after expand to FP32 and transpose, this is not FP32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(
            lstmOp.bias(), "lowered", bias_uint32, shape, storageType, wTF);
        biasOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      bool isDoLoweredInitialC = true;
      if (!isTensorNone(lstmOp.initial_h()) &&
          !isTensorNone(lstmOp.initial_c())) {
        // check is the same
        auto initial_h_weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            lstmOp.initial_h().getDefiningOp());
        auto initial_c_weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            lstmOp.initial_c().getDefiningOp());
        if (initial_h_weightOp.name() == initial_c_weightOp.name()) {
          isDoLoweredInitialC = false;
        }
      }

      // lower initial_h
      if (!isTensorNone(lstmOp.initial_h())) {
        assert(initial_hOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.initial_h(), shape, size);
        auto filter =
            readAndDeleteWeightTensor<bfloat16>(lstmOp.initial_h(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 3);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(lstmOp.initial_h(),
                                                   "lowered", filter_bf16,
                                                   shape, storageType, wTF);
        initial_hOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower initial_c
      if (!isTensorNone(lstmOp.initial_c())) {
        if (isDoLoweredInitialC) {
          assert(initial_cOp.storage() == "BF16");
          std::vector<int64_t> shape;
          int64_t size;
          getTensorShapeAndSize(lstmOp.initial_c(), shape, size);
          auto filter =
              readAndDeleteWeightTensor<bfloat16>(lstmOp.initial_c(), wTF);
          std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
          // transpose h,n
          assert(shape.size() == 3);

          // save it
          StringRef storageType = "BF16";
          addWeightTensorAndUpdateWeightOp<uint16_t>(lstmOp.initial_c(),
                                                     "lowered", filter_bf16,
                                                     shape, storageType, wTF);
          initial_cOp->setAttr("lowered", rewriter.getBoolAttr(true));
        }
      }

      // lower sigmoid table
      if (!isTensorNone(lstmOp.sigmoid_table())) {
        auto tableOp =
            cast<tpu::LoadWeightOp>(lstmOp.getOperand(6).getDefiningOp());
        auto table_mantissaOp =
            cast<tpu::LoadWeightOp>(lstmOp.getOperand(7).getDefiningOp());

        if (tableOp.lowered()) {
          // lowered already
          return failure();
        }

        // lower filter
        assert(tableOp.storage() == "BF16");
        assert(table_mantissaOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.sigmoid_table(), shape, size);
        auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
        auto table_mantissa =
            readAndDeleteWeightTensor<float>(table_mantissaOp, wTF);
        std::vector<uint16_t> table_uint16(table->begin(), table->end());
        std::vector<uint16_t> table_mantissa_uint16(table_mantissa->begin(),
                                                    table_mantissa->end());
        // 1880 support 256 lookup table
        // because of 1880 hardware search table only on each local memory
        // we dupicate table to limit number <32>
        assert(shape[2] * shape[3] == 256);

        // save it
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            tableOp, "lowered", table_uint16, shape, "BF16", wTF);
        tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<uint16_t>(table_mantissaOp, "lowered",
                                                   table_mantissa_uint16, shape,
                                                   "BF16", wTF);
        table_mantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower tanh  table-
      if (!isTensorNone(lstmOp.tanh_table())) {
        auto tableOp =
            cast<tpu::LoadWeightOp>(lstmOp.getOperand(8).getDefiningOp());
        auto table_mantissaOp =
            cast<tpu::LoadWeightOp>(lstmOp.getOperand(9).getDefiningOp());

        if (tableOp.lowered()) {
          // lowered already
          return failure();
        }

        // lower filter
        assert(tableOp.storage() == "BF16");
        assert(table_mantissaOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(lstmOp.tanh_table(), shape, size);
        auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
        auto table_mantissa =
            readAndDeleteWeightTensor<float>(table_mantissaOp, wTF);
        std::vector<uint16_t> table_uint16(table->begin(), table->end());
        std::vector<uint16_t> table_mantissa_uint16(table_mantissa->begin(),
                                                    table_mantissa->end());
        // 1880 support 256 lookup table
        // because of 1880 hardware search table only on each local memory
        // we dupicate table to limit number <32>
        assert(shape[2] * shape[3] == 256);

        // save it
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            tableOp, "lowered", table_uint16, shape, "BF16", wTF);
        tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<uint16_t>(table_mantissaOp, "lowered",
                                                   table_mantissa_uint16, shape,
                                                   "BF16", wTF);
        table_mantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else {
      return failure();
      assert(0 && "Not supported type isn't bf16");
    }
    return success();
  }
};

struct LowerWeightLayerNormOpPattern : public RewritePattern {
  LowerWeightLayerNormOpPattern(MLIRContext *context)
      : RewritePattern("tpu.layer_norm", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::LayerNormOp>(op);
    if (getOpQuant(op) != "BF16") {
      return failure();
    }

    LLVM_DEBUG(llvm::errs() << "Lower Weight for LayerNormOp: " << getOpName(op)
                            << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    auto lutOp = cast<tpu::LoadWeightOp>(castOp.getOperand(1).getDefiningOp());
    auto lutMantissaOp =
        cast<tpu::LoadWeightOp>(castOp.getOperand(2).getDefiningOp());

    if (lutOp.lowered()) {
      // lowered already
      return failure();
    }
    // lower filter
    assert(lutOp.storage() == "BF16");
    assert(lutMantissaOp.storage() == "BF16");
    std::vector<int64_t> shape;
    int64_t size;
    // sqrt table
    getTensorShapeAndSize(castOp.table(), shape, size);
    auto table = readAndDeleteWeightTensor<float>(lutOp, wTF);
    auto tableMantissa = readAndDeleteWeightTensor<float>(lutMantissaOp, wTF);
    std::vector<uint16_t> table_uint16(table->begin(), table->end());
    std::vector<uint16_t> tableMantissa_uint16(tableMantissa->begin(),
                                               tableMantissa->end());
    addWeightTensorAndUpdateWeightOp<uint16_t>(lutOp, "lowered", table_uint16,
                                               shape, "BF16", wTF);
    lutOp->setAttr("lowered", rewriter.getBoolAttr(true));
    addWeightTensorAndUpdateWeightOp<uint16_t>(
        lutMantissaOp, "lowered", tableMantissa_uint16, shape, "BF16", wTF);
    lutMantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));
    return success();
  }
};

struct LowerWeightSoftmaxOpPattern : public RewritePattern {
  LowerWeightSoftmaxOpPattern(MLIRContext *context)
      : RewritePattern("tpu.softmax", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto softmaxOp = cast<tpu::SoftmaxOp>(op);
    if (getOpQuant(op) != "BF16") {
      return failure();
    }
    if(auto exponential_tableOp = llvm::dyn_cast<tpu::LoadWeightOp>(softmaxOp.exponential_table().getDefiningOp())) {
      if (exponential_tableOp.lowered()) {
        // lowered already
        return failure();
      }
    }

    LLVM_DEBUG(llvm::errs() << "Lower Weight for SoftmaxOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

     if (getOpQuant(op) == "BF16") {

       // lower sigmoid table
      auto exponentialTableOp = cast<tpu::LoadWeightOp>(softmaxOp.getOperand(1).getDefiningOp());
      auto exponentialSlopeTableOp = cast<tpu::LoadWeightOp>(softmaxOp.getOperand(2).getDefiningOp());


      if (exponentialTableOp.lowered()) {
        // lowered already
        return failure();
      }

      // lower filter
      assert(exponentialTableOp.storage() == "BF16");
      assert(exponentialSlopeTableOp.storage() == "BF16");
      std::vector<int64_t> shape;
      int64_t size;
      getTensorShapeAndSize(softmaxOp.exponential_table(), shape, size);
      auto table = readAndDeleteWeightTensor<float>(exponentialTableOp, wTF);
      auto tableSlope = readAndDeleteWeightTensor<float>(exponentialSlopeTableOp,
                                                            wTF);
      std::vector<uint16_t> table_uint16(table->begin(), table->end());
      std::vector<uint16_t> tableSlope_uint16(tableSlope->begin(),
                                                  tableSlope->end());
      // 1880 support 256 lookup table
      // because of 1880 hardware search table only on each local memory
      // we dupicate table to limit number <32>
      assert(shape[2] * shape[3] == 256);

      // save it
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          exponentialTableOp, "lowered", table_uint16, shape, "BF16", wTF);
      exponentialTableOp->setAttr("lowered", rewriter.getBoolAttr(true));
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          exponentialSlopeTableOp, "lowered", tableSlope_uint16,
          shape, "BF16", wTF);
      exponentialSlopeTableOp->setAttr("lowered", rewriter.getBoolAttr(true));

      // lower tanh  table-
        auto reciprocalOp = cast<tpu::LoadWeightOp>(softmaxOp.getOperand(3).getDefiningOp());
        auto reciprocal_mantissaOp = cast<tpu::LoadWeightOp>(softmaxOp.getOperand(4).getDefiningOp());


      if (reciprocalOp.lowered()) {
        // lowered already
        return failure();
      }

      // lower filter
      assert(reciprocalOp.storage() == "BF16");
      assert(reciprocal_mantissaOp.storage() == "BF16");
      getTensorShapeAndSize(softmaxOp.reciprocal_table(), shape, size);
      auto reciprocalTable = readAndDeleteWeightTensor<float>(reciprocalOp, wTF);
      auto table_mantissa = readAndDeleteWeightTensor<float>(reciprocal_mantissaOp,
                                                            wTF);
      std::vector<uint16_t> reciprocal_table_uint16(reciprocalTable->begin(), reciprocalTable->end());
      std::vector<uint16_t> reciprocal_table_mantissa_uint16(table_mantissa->begin(),
                                                  table_mantissa->end());
      // 1880 support 256 lookup table
      // because of 1880 hardware search table only on each local memory
      // we dupicate table to limit number <32>
      assert(shape[2] * shape[3] == 256);

      // save it
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          reciprocalOp, "lowered", reciprocal_table_uint16, shape, "BF16", wTF);
      reciprocalOp->setAttr("lowered", rewriter.getBoolAttr(true));
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          reciprocal_mantissaOp, "lowered", reciprocal_table_mantissa_uint16,
          shape, "BF16", wTF);
      reciprocal_mantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));
    } else {
      return failure();
      assert(0 && "Not supported type isn't bf16");
    }
    return success();
  }
};

struct LowerWeightPReluOpPattern : public RewritePattern {
  LowerWeightPReluOpPattern(MLIRContext *context)
      : RewritePattern("tpu.prelu", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto prOp = cast<tpu::PReluOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(prOp.getOperand(1).getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return failure();
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
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(filterOp, shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(prOp.filter(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());

        // save it
        addWeightTensorAndUpdateWeightOp<uint16_t>(prOp.filter(),
            "lowered", filter_bf16, shape, "BF16", wTF);
        filterOp->setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }
    return success();
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

struct LowerWeightLrnOpPattern : public RewritePattern {
  LowerWeightLrnOpPattern(MLIRContext *context)
      : RewritePattern("tpu.lrn", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lrnOp = cast<tpu::LrnOp>(op);
    if (getOpQuant(op) == "INT8") {
      assert(getOpQuant(op) == "INT8" && "only support int8 now");
      auto sqTableOp =
          cast<tpu::LoadWeightOp>(lrnOp.getOperand(1).getDefiningOp());
      auto powerTableOp =
          cast<tpu::LoadWeightOp>(lrnOp.getOperand(2).getDefiningOp());
      if (sqTableOp.lowered() && powerTableOp.lowered()) {
        // lowered already
        return failure();
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
      sqTableOp->setAttr("lowered", rewriter.getBoolAttr(true));
      // update powerTableOp
      getTensorShapeAndSize(powerTableOp, shape, size);
      auto powerTable = readAndDeleteWeightTensor<float>(powerTableOp, wTF);
      std::vector<uint8_t> powerTable_uint8(powerTable->begin(), powerTable->end());
      addWeightTensorAndUpdateWeightOp<uint8_t>(
          powerTableOp, "lowered", powerTable_uint8, shape, "UINT8", wTF);
      powerTableOp->setAttr("lowered", rewriter.getBoolAttr(true));
    } else if (getOpQuant(op) == "BF16") {
      assert(getOpQuant(op) == "BF16");
      auto powerExpTableOp =
          cast<tpu::LoadWeightOp>(lrnOp.getOperand(1).getDefiningOp());
      auto powerMantissaTableOp =
          cast<tpu::LoadWeightOp>(lrnOp.getOperand(2).getDefiningOp());
      if (powerExpTableOp.lowered() && powerMantissaTableOp.lowered()) {
        // lowered already
        return failure();
      }
      assert(powerExpTableOp.storage() == "BF16");
      assert(powerMantissaTableOp.storage() == "BF16");
      assert(powerExpTableOp.lowered() == false && powerMantissaTableOp.lowered() == false);

      TensorFile *wTF = getWeightTensorFile(op);

      std::vector<int64_t> shape;
      int64_t size;
      // update power exp table
      getTensorShapeAndSize(powerExpTableOp, shape, size);
      auto powerExpTable =
            readAndDeleteWeightTensor<float>(powerExpTableOp, wTF);
      std::vector<uint16_t>  powerExpTable_bf16(powerExpTable->begin(),
                                                powerExpTable->end());
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          powerExpTableOp, "lowered",
          powerExpTable_bf16, shape, "BF16", wTF);
      powerExpTableOp->setAttr("lowered", rewriter.getBoolAttr(true));
      // update power mantissa table
      getTensorShapeAndSize(powerMantissaTableOp, shape, size);
      auto powerMantissaTable =
            readAndDeleteWeightTensor<float>(powerMantissaTableOp, wTF);
      std::vector<uint16_t> powerMantissaTable_bf16(powerMantissaTable->begin(),
                                                    powerMantissaTable->end());
      addWeightTensorAndUpdateWeightOp<uint16_t>(
          powerMantissaTableOp, "lowered",
          powerMantissaTable_bf16, shape, "BF16", wTF);
      powerMantissaTableOp->setAttr("lowered", rewriter.getBoolAttr(true));
    }
    return success();
  }
};

template <typename OpTy>
struct LowerWeightLutOpPattern : public RewritePattern {
  LowerWeightLutOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lutOp = cast<OpTy>(op);
    auto tableOp = cast<tpu::LoadWeightOp>(lutOp.getOperand(1).getDefiningOp());
    auto table_mantissaOp = cast<tpu::LoadWeightOp>(lutOp.getOperand(2).getDefiningOp());

    if (tableOp.lowered()) {
      // lowered already
      return failure();
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

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(
            tableOp, "lowered", table_int8, shape, "INT8", wTF);
        tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<int8_t>(
            table_mantissaOp, "lowered", table_mantissa_int8, shape, "INT8", wTF);
        table_mantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));

    } else if (getOpQuant(op) == "BF16") {
      // lower filter
        assert(tableOp.storage() == "BF16");
        assert(table_mantissaOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(tableOp, shape, size);
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

        // save it
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            tableOp, "lowered", table_uint16, shape, "BF16", wTF);
        tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
        addWeightTensorAndUpdateWeightOp<uint16_t>(
            table_mantissaOp, "lowered", table_mantissa_uint16,
            shape, "BF16", wTF);
        table_mantissaOp->setAttr("lowered", rewriter.getBoolAttr(true));
    }

    return success();
  }
};

struct LowerWeightScaleLutOpPattern : public RewritePattern {
  LowerWeightScaleLutOpPattern(MLIRContext *context)
      : RewritePattern("tpu.scale_lut", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::ScaleLutOp>(op);
    auto tableOp =
        cast<tpu::LoadWeightOp>(castOp.getOperand(1).getDefiningOp());
    if (tableOp.lowered()) {
      // lowered already
      return failure();
    }
    LLVM_DEBUG(llvm::errs() << "Lower Weight for ScaleLutOp: "
                            << getOpName(op) << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      assert(tableOp.storage() == "INT8");
      std::vector<int64_t> shape = getTensorShape(tableOp);
      auto table = readAndDeleteWeightTensor<float>(tableOp, wTF);
      std::vector<int8_t> table_int8(table->begin(), table->end());
      // save it
      addWeightTensorAndUpdateWeightOp<int8_t>(
          tableOp, "lowered", table_int8, shape, "INT8", wTF);
      tableOp->setAttr("lowered", rewriter.getBoolAttr(true));
    } else {
      llvm_unreachable("unsupport bf16 scale table op");
    }
    return success();
  }
};

template <typename OpTy>
struct LowerConstEltwiseOpPattern : public RewritePattern {
  LowerConstEltwiseOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {

    auto eltwiseOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs()
                   << "Lower ConstEltwiseOp Op " << eltwiseOp.getOperationName()
                   << ":" << getOpName(eltwiseOp) << "\n";);
    int opdIdx = -1;
    for (unsigned i = 0; i < 2; ++i) {
      auto defOp = eltwiseOp.getOperand(i).getDefiningOp();
      if (isa<tpu::LoadWeightOp>(defOp)) {
        auto constDefOp = cast<tpu::LoadWeightOp>(defOp);
        if (constDefOp.lowered())
          return failure();
        opdIdx = i;
        break;
      }
    }

    if (opdIdx == -1)
      return failure();

    auto defOp = eltwiseOp.getOperand(opdIdx).getDefiningOp();
    auto constDefOp = cast<tpu::LoadWeightOp>(defOp);
    TensorFile *wTF = getWeightTensorFile(eltwiseOp);

    if (getOpQuant(eltwiseOp) == "INT8") {
      // lower filter
      assert(constDefOp.storage() == "INT8");
      std::vector<int64_t> shape;
      int64_t size;
      getTensorShapeAndSize(constDefOp, shape, size);
      auto constValue = readAndDeleteWeightTensor<float>(
                                   eltwiseOp.getOperand(opdIdx), wTF);
      std::vector<int8_t> constValueInt8(constValue->begin(),
                                         constValue->end());
      // save it
      addWeightTensorAndUpdateWeightOp<int8_t>(eltwiseOp.getOperand(opdIdx),
          "lowered", constValueInt8, shape, "INT8", wTF);
      constDefOp->setAttr("lowered", rewriter.getBoolAttr(true));
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      std::vector<int64_t> shape;
      int64_t size;
      getTensorShapeAndSize(constDefOp, shape, size);
      auto constValue = readAndDeleteWeightTensor<float>(eltwiseOp.getOperand(opdIdx), wTF);
      std::vector<bfloat16> constBf16(size);
      FloatToBFloat16(constValue->data(), constBf16.data(), size);

      // save it
      addWeightTensorAndUpdateWeightOp<bfloat16>(eltwiseOp.getOperand(opdIdx),
          "lowered", constBf16, shape, "BF16", wTF);
      constDefOp->setAttr("lowered", rewriter.getBoolAttr(true));
      constDefOp->setAttr("storage", rewriter.getStringAttr("BF16"));
    }
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
      if (castOp.getOpQuant() == "INT8") {
        auto newOp = OpBuilder(op).create<tpu::TG_INT8_GenericTpuOp>(
            op->getLoc(), castOp.getResult().getType(), ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        auto result = newOp.getResult();
        rewriter.replaceOp(op, {result});
      } else if (castOp.getOpQuant() == "BF16") {
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
struct LowerFunctionTypePattern: public RewritePattern {
  LowerFunctionTypePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto quantOp = cast<OpTy>(op);
    auto prevOp = op->getOperand(0).getDefiningOp();
    auto nextOp = getNextOp(op);
    if (nextOp == nullptr) {
      return failure();
    }
    if (!isa<tpu::InputOp>(prevOp) && !isa<ReturnOp>(nextOp)) {
      return failure();
    }

    auto fn = op->getParentOfType<FuncOp>();
    assert(fn);
    // change the argType of FuncOp
    if (isa<tpu::InputOp>(prevOp)) {
      if (quantOp.from() == "NONE" &&
          (quantOp.to() == "INT8" || quantOp.to() == "UINT8") &&
          !clQuantInputsToInt8) {
        // remove quantOp and change argType
        // and inputOp's type to int8
        auto argument = prevOp->getOperand(0);
        auto bSigned = (quantOp.to() == "INT8") ? IntegerType::Signed : IntegerType::Unsigned;
        setOpResultType(argument, IntegerType::get(op->getContext(), 8, bSigned));
        setOpResultType(prevOp->getResult(0), IntegerType::get(op->getContext(), 8, bSigned));
        prevOp->setAttr("name", quantOp.nameAttr());
        setOpThreshold(prevOp, (quantOp.to() == "INT8" ? 128 : 256) /
                                quantOp.scale().convertToFloat());
        setOpZeroPoint(prevOp,
                       (int)quantOp.zero_point());
        rewriter.replaceOp(op, {op->getOperand(0)});
      }
    } else if (isa<ReturnOp>(nextOp) && !clDequantResultsToFp32) {
      // change the returnType of FuncOp
      if (quantOp.from() == "INT8" && quantOp.to() == "NONE") {
        rewriter.replaceOp(op, {op->getOperand(0)});
      }
    } else {
      return failure();
    }

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    assert(fn);
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
          (float)castOp.quant().threshold_max().getValue().convertToFloat();
      if (threshold != 0) {
        auto elementType = castOp.getResult().getType().template
                        cast<TensorType>().getElementType();
        int max_val = elementType.isUnsignedInteger(8) ? 255 : 128;
        qscale = max_val / threshold;
      }
      char qscale_str[64] = {0};
      sprintf(qscale_str, "%.12f", qscale);
      zero_point = castOp.quant().zero_point().getInt();
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
        LowerFunctionTypePattern<tpu::QuantOp>
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
        LowerWeightLayerNormOpPattern,
        LowerWeightLrnOpPattern,
        LowerWeightLutOpPattern<tpu::ReciprocalOp>,
        LowerWeightPReluOpPattern,
        LowerWeightLutOpPattern<tpu::MishOp>,
        LowerWeightLutOpPattern<tpu::SigmoidOp>,
        LowerWeightLutOpPattern<tpu::SqrtOp>,
        LowerWeightLutOpPattern<tpu::TanHOp>,
        LowerWeightLutOpPattern<tpu::ExpOp>,
        LowerWeightLutOpPattern<tpu::SoftPlusOp>,
        LowerConstEltwiseOpPattern<tpu::EltwiseAddOp>,
        LowerConstEltwiseOpPattern<tpu::EltwiseMulOp>,
        LowerConstEltwiseOpPattern<tpu::EltwiseAddOp>,
        LowerWeightScaleLutOpPattern,
        LowerWeightFullyConnectedOpPattern,
        LowerWeightDetectionOutputOpPattern,
        LowerWeightInstanceNormOpPattern,
        LowerWeightGruOpPattern,
        LowerWeightLstmOpPattern,
        LowerWeightSoftmaxOpPattern
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

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
        DefaultToTGPattern<tpu::CropOp>,
        DefaultToTGPattern<tpu::DeConv2DOp>,
        DefaultToTGPattern<tpu::DilateOp>,
        DefaultToTGPattern<tpu::ReciprocalOp>,
        DefaultToTGPattern<tpu::EltwiseAddOp>,
        DefaultToTGPattern<tpu::EltwiseMaxOp>,
        DefaultToTGPattern<tpu::EltwiseMinOp>,
        DefaultToTGPattern<tpu::EltwiseMulOp>,
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
        DefaultToTGPattern<tpu::PReluOp>,
        DefaultToTGPattern<tpu::QuantOp>,
        DefaultToTGPattern<tpu::ReQuantOp>,
        DefaultToTGPattern<tpu::ReluOp>,
        DefaultToTGPattern<tpu::ReorgOp>,
        DefaultToTGPattern<tpu::ReverseOp>,
        DefaultToTGPattern<tpu::ScaleLutOp>,
        DefaultToTGPattern<tpu::ShuffleChannelOp>,
        DefaultToTGPattern<tpu::SigmoidOp>,
        DefaultToTGPattern<tpu::SliceOp>,
        DefaultToTGPattern<tpu::SqrtOp>,
        DefaultToTGPattern<tpu::SwapChannelOp>,
        DefaultToTGPattern<tpu::TanHOp>,
        DefaultToTGPattern<tpu::ExpOp>,
        DefaultToTGPattern<tpu::TileOp>,
        DefaultToTGPattern<tpu::UpsampleOp>,
        DefaultToTGPattern<tpu::ReduceL2Op>,
        DefaultToTGPattern<tpu::ReduceMeanOp>,
        DefaultToTGPattern<tpu::ReduceMaxOp>,
        DefaultToTGPattern<tpu::GruOp>,
        DefaultToTGPattern<tpu::LstmOp>,
        DefaultToTGPattern<tpu::SoftmaxOp>,
        DefaultToTGPattern<tpu::SoftPlusOp>,
        DefaultToTGPattern<tpu::SquareOp>,
        DefaultToTGPattern<tpu::QuadraticSumOp>,
        DefaultToTGPattern<tpu::CscOp>,
        DefaultToTGPattern<tpu::ZeroMaskOp>,
        DefaultToTGPattern<tpu::MatMulOp>
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
