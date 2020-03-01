//===- GenTanHTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include <float.h>

#define DEBUG_TYPE "gen-lut-table"

using namespace mlir;

// to cleanup


#if 0
struct TpuQuantInt8SigmoidOpPattern : public RewritePattern {
  TpuQuantInt8SigmoidOpPattern(MLIRContext *context)
      : RewritePattern("tpu.sigmoid", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto sigOp = cast<tpu::SigmoidOp>(op);

    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs()
                     << " < " << getOpName(op) << ", quantized already\n";);
      return matchFailure();
    }
    assert(getOpQuantParamType(op) == "THRESHOLD");
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // quantization
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                            << ", threshold_y = " << std::to_string(threshold_y)
                            << ", threshold_x = " << std::to_string(threshold_x)
                            << "\n";);
    int npu_num = 32; //<! 1880v2 hardcode

    //<! 1880v2 hw config
    int table_h;
    int table_w;
    int table_hw;

    int tbl_shape;
    std::vector<float> y0_table;
    //<! 1880v2 hw int8 config
    table_h = 16;
    table_w = 16;
    table_hw = table_h * table_w;

    tbl_shape = npu_num * table_hw;
    y0_table.resize(tbl_shape);

    // input: 0~127, -128~ -1, Y=1/(1+EXP(-X*thx/128)) * 128/thy
    // output:0~127, negative is invalid
    for (int n = 0; n < npu_num; n++) {
      for (int idx = 0; idx < table_hw; ++idx) {
        char lutInput = static_cast<char>(idx);
        float index = -lutInput * threshold_x / 127.0;
        float lutOutput = 1.0 / (1 + std::exp(index)) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      }
    }
    // update op
    auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
    StringRef storageType = "INT8";
    auto y0_table_op = addWeightTensorAndCreateWeightOp<float>(
        op, "y0_table", y0_table, shape, storageType, wTF, wfV);
    sigOp.setOperand(1, y0_table_op);
    setOpQuantPerchannel(op, false);
    setOpQuant(op, "INT8");

    return matchSuccess();
  }
};
#endif



#if 0

struct TpuQuantPowerOpPattern : public RewritePattern {
    TpuQuantPowerOpPattern(MLIRContext *context)
      :RewritePattern("tpu.power", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {

    auto powerOp = cast<tpu::PowerOp>(op);

    if(powerOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << powerOp.name() << " gen already\n";);
      return matchFailure();
    }
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    std::string op_name = powerOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto result_var = powerOp.getResult();

    llvm::ArrayRef<int64_t> input_shape = result_var->getType().dyn_cast<mlir::TensorType>().getShape();
    assert(input_shape.size() == 4);
    auto size = input_shape[1];// get channel number

    // get quant type
    QUANT_INT8_TYPE_e quant;
    if (!clQuantConvPerChannel) {
      assert(!clQuantConvMultiplier
             && "enable per channel before enable multiplier");
      quant = INT8_PER_LAYER;
    } else if (!clQuantConvMultiplier) {
      quant = INT8_PER_CHANNEL;
    } else {
      quant = INT8_MULTIPLER;
    }

    //assign scale and shift tensor
    std::vector<float> scale_weight(size);
    std::vector<float> shift_weight(size);

    float threshold_y,threshold_x,qscale;
    int8_t rshift;
    uint32_t multiplier;

    threshold_y = powerOp.threshold_y().getValue().convertToFloat();
    threshold_x = getPreviousOpThreshold(powerOp);

    qscale = (threshold_x*threshold_x) /(127.0*threshold_y);

    float scale = powerOp.scale().convertToFloat();
    float shift = powerOp.shift().convertToFloat();

    if (quant == INT8_PER_LAYER|| quant == INT8_PER_CHANNEL) {
      rshift = findRShiftAndMultiplierFromQScale(qscale);
      multiplier = findMultiplierI8FromQScaleAndRShift(qscale, rshift);
    }else if(quant == INT8_MULTIPLER){
      rshift = findRShiftAndMultiplierFromQScale(qscale, &multiplier, true,255);
    }

    if (quant == INT8_PER_LAYER|| quant == INT8_PER_CHANNEL) {
      scale = scale*(threshold_y/threshold_x)*multiplier;
      shift = shift*(threshold_y/127.0)*multiplier;
      scale = (float)applyRShiftAndSaturateInt8(scale, rshift);
      shift = (float)applyRShiftAndSaturateInt8(shift, rshift);
    }else if(quant == INT8_MULTIPLER){
      scale = scale*(threshold_y/threshold_x);
      shift = shift*(threshold_y/127.0);
      scale = (float)applyMultiplierAndRShiftAndSaturateInt8(scale,rshift,  multiplier);
      shift = (float)applyMultiplierAndRShiftAndSaturateInt8(shift,rshift,  multiplier);
    }

    for (uint32_t i = 0; i < scale_weight.size(); i++) {
      scale_weight[i] = scale;
    }

    for (uint32_t i = 0; i < shift_weight.size(); i++) {
      shift_weight[i] = shift;
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(powerOp.getOperand(0));
    auto type = RankedTensorType::get(input_shape,FloatType::getF32(rewriter.getContext()));

    //add scale operand
    auto tensor_name = op_name + "_gen_scale";
    wTF->addTensor<float>(tensor_name, scale_weight.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
    auto new_scale_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_scale_op);


    //add scale operand
    tensor_name = op_name + "_gen_shift";
    wTF->addTensor<float>(tensor_name, shift_weight.data(), type);
    std::vector<NamedAttribute> attrs_shift;
    attrs_shift.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs_shift.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
    auto new_shift_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs_shift});
    newOperands.push_back(new_shift_op);

    powerOp.setAttr("has_table", rewriter.getBoolAttr("true"));

          // set quant type
      if (quant == INT8_PER_LAYER) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8"));
      } else if (quant == INT8_PER_CHANNEL) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL"));
      } else if (quant == INT8_MULTIPLER) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8_MULTIPLIER"));
      }

    rewriter.replaceOpWithNewOp<tpu::PowerOp>(
        powerOp, powerOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{powerOp.getAttrs()});

    return matchSuccess();
  }
};

#endif
