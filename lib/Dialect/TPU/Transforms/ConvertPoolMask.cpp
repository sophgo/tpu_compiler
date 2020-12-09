//===- ConvertUpsample.cpp - convert unsample to deconv -----------===//
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
// This file convert upsample op to deconv.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "pool_mask"

using namespace mlir;

namespace {

struct TpuPoolMaskPattern : public RewritePattern {
  TpuPoolMaskPattern(MLIRContext *context)
      : RewritePattern("tpu.pool_mask", 8, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto maskOp = cast<tpu::PoolMaskOp>(op);
    if (maskOp.getNumOperands() < 2) {
      return failure();
    }
    auto poolOp = maskOp.getOperand(0);
    if (isa<tpu::PoolMax2DOp>(poolOp.getDefiningOp()) == false) {
      return failure();
    }
    auto pool_shape = getTensorShape(poolOp);
    auto formerOp = maskOp.getOperand(1);
    auto former_shape = getTensorShape(formerOp);
    bool need_pad = false;

    TensorFile *wTF = getWeightTensorFile(op);
    Value wFV = getWeightFileValue(op);
    std::string op_name = maskOp.name().str();
    auto scale = maskOp.scale();

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> output_shape;
    output_shape.push_back(pool_shape[0]);
    output_shape.push_back(pool_shape[1]);
    output_shape.push_back(pool_shape[2] * scale);
    output_shape.push_back(pool_shape[3] * scale);
    auto pool_type = poolOp.getType().cast<RankedTensorType>();
    RankedTensorType output_type =
        RankedTensorType::get(output_shape, pool_type.getElementType());
    if (output_shape[2] != former_shape[2] ||
        output_shape[3] != former_shape[3]) {
      need_pad = true;
    }

    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());

    // a = upsample(x0)
    auto layer_name = maskOp.name().str();
    auto name = layer_name + "_upsample0";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    operands.push_back(poolOp);
    operands.push_back(NoneOp.getResult());
    auto op_a = rewriter.create<tpu::UpsampleOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // b = -1 * a
    std::vector<float> const_scale_0(output_shape[1], -1);
    std::vector<float> const_bias_0(output_shape[1], 0);
    std::vector<int64_t> const_shape;
    const_shape.push_back(1);
    const_shape.push_back(output_shape[1]);
    const_shape.push_back(1);
    const_shape.push_back(1);
    auto scale_value_0 = addWeightTensorAndCreateWeightOp<float>(
        op_a, "scale", const_scale_0, const_shape, "NONE", wTF, wFV);
    auto bias_value_0 = addWeightTensorAndCreateWeightOp<float>(
        op_a, "bias", const_bias_0, const_shape, "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_scale0";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    operands.push_back(op_a);
    operands.push_back(scale_value_0);
    operands.push_back(bias_value_0);
    auto op_b = rewriter.create<tpu::ScaleOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // check whether need pad
    auto temp_op = formerOp;
    if (need_pad) {
      // pad op1
      attrs.clear();
      operands.clear();
      name = layer_name + "_pad";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      attrs.push_back(
          rewriter.getNamedAttr("const_val", rewriter.getF32FloatAttr(0.0)));
      std::vector<int32_t> pads(8, 0);
      pads[6] = output_shape[2] - former_shape[2]; // pad h right
      pads[7] = output_shape[3] - former_shape[3]; // pad w right
      attrs.push_back(rewriter.getNamedAttr(
          "pads", rewriter.getI32ArrayAttr(ArrayRef<int32_t>({pads}))));
      operands.push_back(formerOp);
      temp_op = rewriter.create<tpu::PadOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
    }
    // c = b + x1
    attrs.clear();
    operands.clear();
    name = layer_name + "_eltwise_add0";
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant_skip", rewriter.getBoolAttr(true)));
    operands.push_back(temp_op);
    operands.push_back(op_b);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    auto op_c = rewriter.create<tpu::EltwiseAddOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // d = zero_mask(c), if c[i] < 0, d[i] = 0; c[i] = 0, d[i] = 1
    attrs.clear();
    operands.clear();
    name = layer_name + "_zero_mask0";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    operands.push_back(op_c);
    auto op_d = rewriter.create<tpu::ZeroMaskOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // e = d * [4, 3, 2, 1]
    auto count =
        output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];

    std::vector<float> const_data(count, 0);
    for (int i = 0; i < output_shape[0] * output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        for (int k = 0; k < output_shape[3]; k++) {
          auto offset =
              i * output_shape[2] * output_shape[3] + j * output_shape[3] + k;
          if (j % 2 == 0) {
            if (k % 2 == 0)
              const_data[offset] = 4;
            if (k % 2 == 1)
              const_data[offset] = 3;
          } else {
            if (k % 2 == 0)
              const_data[offset] = 2;
            if (k % 2 == 1)
              const_data[offset] = 1;
          }
        }
      }
    }
    auto const_value = addWeightTensorAndCreateWeightOp<float>(
        op_d, "const", const_data, output_shape, "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_eltwise_mul0";
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("quant_skip", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    operands.push_back(op_d);
    operands.push_back(const_value);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto op_e = rewriter.create<tpu::EltwiseMulOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // f = pool_max(e)
    attrs.clear();
    operands.clear();
    name = layer_name + "_pool_max0";
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr(
        "param",
        tpu::PoolParam::get(
            rewriter.getI32IntegerAttr(scale),
            rewriter.getI32IntegerAttr(scale), rewriter.getI32IntegerAttr(0),
            rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0),
            rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0),
            rewriter.getI32IntegerAttr(scale),
            rewriter.getI32IntegerAttr(scale), rewriter.getBoolAttr(false),
            rewriter.getBoolAttr(true), rewriter.getContext())));
    operands.push_back(op_e);
    auto op_f = rewriter.create<tpu::PoolMax2DOp>(
        op->getLoc(), ArrayRef<mlir::Type>{pool_type}, operands, attrs);

    // g = upsample(f)
    attrs.clear();
    operands.clear();
    operands.push_back(op_f);
    operands.push_back(NoneOp.getResult());
    name = layer_name + "_" + "upsample1";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(scale)));
    auto op_g = rewriter.create<tpu::UpsampleOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // h = -1 * g
    std::vector<float> const_scale_2(output_shape[1], -1);
    std::vector<float> const_bias_2(output_shape[1], 0);
    auto scale_value_2 = addWeightTensorAndCreateWeightOp<float>(
        op_g, "scale", const_scale_2, const_shape, "NONE", wTF, wFV);
    auto bias_value_2 = addWeightTensorAndCreateWeightOp<float>(
        op_g, "bias", const_bias_2, const_shape, "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_scale1";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    operands.push_back(op_g);
    operands.push_back(scale_value_2);
    operands.push_back(bias_value_2);
    auto op_h = rewriter.create<tpu::ScaleOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // i = h + e
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "eltwise_add1";
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant_skip", rewriter.getBoolAttr(true)));
    operands.push_back(op_h);
    operands.push_back(op_e);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto op_i = rewriter.create<tpu::EltwiseAddOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // j = zero_mask(i)
    attrs.clear();
    operands.clear();
    name = layer_name + "_zero_mask1";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    operands.push_back(op_i);
    auto op_j = rewriter.create<tpu::ZeroMaskOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    rewriter.replaceOp(op, {op_j});

    return success();
  }
};

} // namespace

void tpu::PoolMaskOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuPoolMaskPattern>(context);
}
