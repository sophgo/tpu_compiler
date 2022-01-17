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
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "upsample_to_deconv"

using namespace mlir;

namespace {

struct TpuUpsampleMaskPattern : public RewritePattern {
  TpuUpsampleMaskPattern(MLIRContext *context)
      : RewritePattern("tpu.upsample", 8, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto upsampleOp = cast<tpu::UpsampleOp>(op);
    if (upsampleOp.getNumOperands() < 2) {
      return failure();
    }
    auto mask_op = upsampleOp.getOperand(1);
    if (isa<tpu::PoolMaskOp>(mask_op.getDefiningOp()) == false) {
      return failure();
    }
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());

    auto mask_shape = getTensorShape(mask_op);
    auto output_shape = getTensorShape(op->getResult(0));
    bool need_crop = false;
    if (mask_shape[3] != output_shape[3] || mask_shape[2] != output_shape[2]) {
      need_crop = true;
    }

    // op_name
    std::string op_name = upsampleOp.name().str();

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto mask_type = mask_op.getType().cast<RankedTensorType>();
    // k = upsampe(input)
    std::string name = op_name + "_nearst";
    attrs.clear();
    operands.clear();
    operands.push_back(upsampleOp.getOperand(0));
    operands.push_back(NoneOp.getResult());
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr("scale_h", upsampleOp.scale_hAttr()));
    attrs.push_back(rewriter.getNamedAttr("scale_w", upsampleOp.scale_wAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    auto new_op = rewriter.create<tpu::UpsampleOp>(
        op->getLoc(), ArrayRef<mlir::Type>{mask_type}, operands, attrs);

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
    operands.push_back(new_op);
    operands.push_back(mask_op); // mask
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto op_l = rewriter.create<tpu::EltwiseMulOp>(
        op->getLoc(), ArrayRef<mlir::Type>{mask_type}, operands, attrs);
    Value temp_op = op_l;

    // output = crop
    if (need_crop) {
      attrs.clear();
      operands.clear();
      name = op_name;
      std::vector<int> crop_offset(4, 0);
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(rewriter.getNamedAttr(
          "crop_offset",
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({crop_offset}))));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(op_l);
      temp_op = rewriter.create<tpu::CropOp>(
          op->getLoc(), upsampleOp.getResult().getType(), operands, attrs);
    }
    rewriter.replaceOp(op, {temp_op});

    return success();
  }
};

#define MAX_CONV_STRIDE 16
struct TpuUpsampleOpPattern : public RewritePattern {
  TpuUpsampleOpPattern(MLIRContext *context)
      : RewritePattern("tpu.upsample", 7, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto upsampleOp = cast<tpu::UpsampleOp>(op);
    if (op->getNumOperands() == 2) {
      auto mask = op->getOperand(1).getDefiningOp();
      if (isa<tpu::NoneOp>(mask) == false) {
        return failure();
      };
    }
    LLVM_DEBUG(llvm::errs() << upsampleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wFV = getWeightFileValue(op);

    // op_name
    std::string op_name =
        upsampleOp->getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "upsample Op: " << op_name << "\n";);

    auto input = op->getOperand(0);
    auto input_type = input.getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    auto scale_h = upsampleOp.scale_h();
    auto scale_w = upsampleOp.scale_w();
    int g = input_shape[1];
    int oc = input_shape[1] / g;
    int ic = input_shape[1] / g;
    int h = scale_h;
    int w = scale_w;

    // stride exceed hw limitation, can not convert
    if (scale_h >= MAX_CONV_STRIDE || scale_w >= MAX_CONV_STRIDE) {
      return failure();
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
    std::vector<Value> operands;
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
    attrs.push_back(rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs.push_back(rewriter.getNamedAttr(
        "param",
        tpu::ConvParam::get(
            rewriter.getI32IntegerAttr(h), rewriter.getI32IntegerAttr(w),
            rewriter.getI32IntegerAttr(stride[0]),
            rewriter.getI32IntegerAttr(stride[1]),
            rewriter.getStringAttr("VALID"),
            rewriter.getI32IntegerAttr(dilation[0]),
            rewriter.getI32IntegerAttr(dilation[1]),
            rewriter.getI32IntegerAttr(padding[0]),
            rewriter.getI32IntegerAttr(padding[0]),
            rewriter.getI32IntegerAttr(padding[1]),
            rewriter.getI32IntegerAttr(padding[1]),
            rewriter.getI32IntegerAttr(g), rewriter.getBoolAttr(is_dw),
            rewriter.getBoolAttr(with_bias),
            rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})),
            rewriter.getI32IntegerAttr(0), rewriter.getContext())));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    rewriter.replaceOpWithNewOp<tpu::DeConv2DOp>(
        upsampleOp, upsampleOp.getResult().getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

class ConvertUpsampleToDeconvPass
    : public mlir::PassWrapper<ConvertUpsampleToDeconvPass, FunctionPass> {
public:
  explicit ConvertUpsampleToDeconvPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuUpsampleOpPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::UpsampleOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuUpsampleMaskPattern, TpuUpsampleOpPattern>(context);
}

std::unique_ptr<mlir::Pass> mlir::createConvertUpsampleToDeconvPass() {
  return std::make_unique<ConvertUpsampleToDeconvPass>();
}
