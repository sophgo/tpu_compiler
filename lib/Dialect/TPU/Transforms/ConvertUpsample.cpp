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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "upsample_to_deconv"

using namespace mlir;

namespace {

struct TpuUpsampleMaskPattern : public RewritePattern {
  TpuUpsampleMaskPattern(MLIRContext *context)
      : RewritePattern("tpu.upsample", 8, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto upsampleOp = cast<tpu::UpsampleOp>(op);
    if (upsampleOp.getNumOperands() < 2) {
        return matchFailure();
    }
    auto mask_op = upsampleOp.getOperand(1)->getDefiningOp();
    if (isa<tpu::PoolMaskOp>(mask_op) == false) {
      return matchFailure();
    }
    std::vector<int64_t> o_shape;
    int64_t output_size;
    getTensorShapeAndSize(op->getResult(0), o_shape, output_size);
    bool need_crop = false;

    auto poolMaskOp = cast<tpu::PoolMaskOp>(mask_op);

    LLVM_DEBUG(llvm::errs() << poolMaskOp.getOperationName() << "\n";);
    assert(poolMaskOp.getNumOperands() == 2 && "operands num should be 2");

    TensorFile *wTF = getWeightTensorFile(op);
    Value *wFV = getWeightFileValue(op);

    // op_name
    std::string op_name =
        upsampleOp.getAttrOfType<StringAttr>("name").getValue().str();

    auto scale = poolMaskOp.scale().getLimitedValue();

    std::vector<Value *> operands;
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> output_shape;

    auto input = poolMaskOp.getOperand(0);
    auto input_type = input->getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(input_shape[1]);
    output_shape.push_back(input_shape[2] * scale);
    output_shape.push_back(input_shape[3] * scale);
    RankedTensorType output_type =
        RankedTensorType::get(output_shape, input_type.getElementType());
    if (output_shape[2] != o_shape[2] || output_shape[3] != o_shape[3]) {
      need_crop = true;
    }

    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());

    // a = upsample(x0)
    auto layer_name = poolMaskOp.name().str();
    auto name = layer_name + "_upsample0";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    operands.push_back(poolMaskOp.getOperand(0));
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
    mlir::Value * temp_op = poolMaskOp.getOperand(1);
    if (need_crop) {
        // pad op1
        attrs.clear();
        operands.clear();
        name = layer_name + "_pad";
        attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
        attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        attrs.push_back(rewriter.getNamedAttr("const_val", rewriter.getF32FloatAttr(0.0)));
        std::vector<int32_t> pads(8, 0);
        pads[6] = output_shape[2] - o_shape[2]; // pad h right
        pads[7] = output_shape[3] - o_shape[3]; // pad w right
        attrs.push_back(rewriter.getNamedAttr(
          "pads",
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({pads}))));
        operands.push_back(poolMaskOp.getOperand(1));
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
            rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(scale),
            rewriter.getI32IntegerAttr(scale), rewriter.getBoolAttr(false),
            rewriter.getBoolAttr(true), rewriter.getContext())));
    operands.push_back(op_e);
    auto op_f = rewriter.create<tpu::PoolMax2DOp>(
        op->getLoc(), ArrayRef<mlir::Type>{input_type}, operands, attrs);

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

    // k = upsampe(input)
    temp_op = op_a;
    if (poolMaskOp.getOperand(0) != upsampleOp.getOperand(0)) {
        name = op_name + "_nearst";
        attrs.clear();
        operands.clear();
        operands.push_back(upsampleOp.getOperand(0));
        operands.push_back(NoneOp.getResult());
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
        attrs.push_back(
            rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(scale)));
        attrs.push_back(
            rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(scale)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

        temp_op = rewriter.create<tpu::UpsampleOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
    }

    // l = j * k
    attrs.clear();
    operands.clear();
    if (need_crop) {
      name = op_name + "_multi";
    } else {
      name = op_name;
    }
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(
        rewriter.getNamedAttr("quant_skip", rewriter.getBoolAttr(true)));
    operands.push_back(op_j); // mask
    operands.push_back(temp_op);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto op_l = rewriter.create<tpu::EltwiseMulOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
    temp_op = op_l;

    // output = crop
    if (need_crop) {
      attrs.clear();
      operands.clear();
      name = op_name;
      std::vector<int> crop_shape;
      for (auto &dim : o_shape) {
        crop_shape.push_back(dim);
      }
      std::vector<int> crop_offset(4, 0);
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(rewriter.getNamedAttr(
          "crop_shape",
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({crop_shape}))));
      attrs.push_back(rewriter.getNamedAttr(
          "crop_offset",
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({crop_offset}))));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(op_l);
      temp_op = rewriter.create<tpu::CropOp>(
          op->getLoc(), upsampleOp.getResult()->getType(), operands, attrs);
    }
    rewriter.replaceOp(op, {temp_op});

    return matchSuccess();
  }
};

#define MAX_CONV_STRIDE 16
struct TpuUpsampleOpPattern : public RewritePattern {
  TpuUpsampleOpPattern(MLIRContext *context)
      : RewritePattern("tpu.upsample", 7, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto upsampleOp = cast<tpu::UpsampleOp>(op);
    if(op->getNumOperands() == 2) {
        auto mask = op->getOperand(1)->getDefiningOp();
        assert(isa<tpu::NoneOp>(mask) && "upsample op 1 must be none");
    }
    LLVM_DEBUG(llvm::errs() << upsampleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wFV = getWeightFileValue(op);

    // op_name
    std::string op_name =
        upsampleOp.getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "upsample Op: " << op_name << "\n";);

    auto input = op->getOperand(0);
    auto input_type = input->getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    auto scale_h = upsampleOp.scale_h().getLimitedValue();
    auto scale_w = upsampleOp.scale_w().getLimitedValue();
    int g = input_shape[1];
    int oc = input_shape[1] / g;
    int ic = input_shape[1] / g;
    int h = scale_h;
    int w = scale_w;

    // stride exceed hw limitation, can not convert
    if (scale_h >= MAX_CONV_STRIDE || scale_w >= MAX_CONV_STRIDE) {
      return matchFailure();
    }

    int count = g * oc * ic * h * w;
    std::vector<float> filter(count, 1);
    std::vector<int64_t> filter_shape;
    if (g != 1) {
      filter_shape.push_back(g);
    }
    filter_shape.push_back(oc);
    filter_shape.push_back(ic);
    filter_shape.push_back(h);
    filter_shape.push_back(w);
    auto filterValue = addWeightTensorAndCreateWeightOp<float>(
        op, "filter", filter, filter_shape, "NONE", wTF, wFV);
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());
    std::vector<Value *> operands;
    operands.push_back(input);
    operands.push_back(filterValue);
    operands.push_back(NoneOp.getResult()); // bias
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    bool is_dw = true;
    bool with_bias = false;
    std::vector<int64_t> stride(2), padding(2), dilation(2);
    padding[0] = padding[1] = 0;
    dilation[0] = dilation[1] = 1;
    stride[0] = scale_h;
    stride[1] = scale_w;

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", upsampleOp.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "param", tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride[0]),
                                     rewriter.getI32IntegerAttr(stride[1]),
                                     rewriter.getStringAttr("VALID"),
                                     rewriter.getI32IntegerAttr(dilation[0]),
                                     rewriter.getI32IntegerAttr(dilation[1]),
                                     rewriter.getI32IntegerAttr(padding[0]),
                                     rewriter.getI32IntegerAttr(padding[0]),
                                     rewriter.getI32IntegerAttr(padding[1]),
                                     rewriter.getI32IntegerAttr(padding[1]),
                                     rewriter.getI32IntegerAttr(g),
                                     rewriter.getBoolAttr(is_dw),
                                     rewriter.getBoolAttr(with_bias),
                                     rewriter.getBoolAttr(false),
                                     rewriter.getI32ArrayAttr(ArrayRef<int32_t>(
                                         {})), // [0]ins_w/[1]ins_h
                                     rewriter.getI32IntegerAttr(0), // pad_value
                                     rewriter.getContext())));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    rewriter.replaceOpWithNewOp<tpu::DeConv2DOp>(
        upsampleOp, upsampleOp.getResult()->getType(),
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }
};

class ConvertUpsampleToDeconvPass
    : public FunctionPass<ConvertUpsampleToDeconvPass> {
public:
  explicit ConvertUpsampleToDeconvPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuUpsampleOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::UpsampleOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuUpsampleMaskPattern, TpuUpsampleOpPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertUpsampleToDeconvPass() {
  return std::make_unique<ConvertUpsampleToDeconvPass>();
}

static PassRegistration<ConvertUpsampleToDeconvPass>
    pass("convert-upsample-to-deconv",
         "Convert a upsample operation to deconv");
