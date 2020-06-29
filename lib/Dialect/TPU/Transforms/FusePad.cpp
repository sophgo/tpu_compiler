//==================- FusePad.cpp - fuse pad ------------------------------===//
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
// This file implements the fusion of pad.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_pad"

using namespace mlir;

namespace {
struct TpuMergeCropPattern : public RewritePattern {
  TpuMergeCropPattern(MLIRContext *context)
      : RewritePattern("tpu.crop", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto crop0Op = dyn_cast<tpu::CropOp>(op);
    if (!op->getResult(0)->hasOneUse())
      return matchFailure();

    for (auto &use : op->getResult(0)->getUses()) {
      auto useOp = use.getOwner();
      if (auto crop1Op = dyn_cast<tpu::CropOp>(useOp)) {
        std::vector<int> crop0Offset;
        std::vector<int> crop1Offset;
        std::vector<NamedAttribute> attrs;
        std::vector<Value *> operands;
        SmallVector<Attribute, 4> mergeOffsetAttr;

        operands.push_back(op->getOperand(0));
        arrayAttrToVector(crop0Op.crop_offset().getValue(), crop0Offset);
        arrayAttrToVector(crop1Op.crop_offset().getValue(), crop1Offset);

        for (unsigned int i = 0; i < crop0Offset.size(); i++) {
          auto cropOffset = crop0Offset[i] + crop1Offset[i];
          auto cropOffsetAttr = rewriter.getI32IntegerAttr(cropOffset);
          mergeOffsetAttr.push_back(cropOffsetAttr);
        }

        attrs.push_back(rewriter.getNamedAttr("name", crop1Op.nameAttr()));
        if (crop1Op.layer_idAttr())
          attrs.push_back(rewriter.getNamedAttr("layer_id",
                                                crop1Op.layer_idAttr()));
        attrs.push_back(rewriter.getNamedAttr("crop_shape",
                                              crop1Op.crop_shapeAttr()));
        attrs.push_back(rewriter.getNamedAttr("crop_offset",
                                      rewriter.getArrayAttr(mergeOffsetAttr)));
        attrs.push_back(rewriter.getNamedAttr("quant",
                                              crop1Op.quantAttr()));

        auto mergeCropOp = rewriter.create<tpu::CropOp>(op->getLoc(),
                    crop1Op.getResult()->getType(), ArrayRef<Value *>{operands},
                    ArrayRef<NamedAttribute>{attrs});

        rewriter.replaceOp(crop1Op, {mergeCropOp.getResult()});
        return matchSuccess();
      } else
        return matchFailure();
    }
    return matchSuccess();
  }
};

struct TpuFusePadPattern : public RewritePattern {
  TpuFusePadPattern(MLIRContext *context)
      : RewritePattern("tpu.pad", 1, context) {}

  template <class T>
  void updatePoolParam(T &poolOp, PatternRewriter &rewriter,
                       std::vector<int32_t> &pads) const {

    auto pad_h_begin = pads[2];
    auto pad_w_begin = pads[3];
    auto pad_h_end = pads[6];
    auto pad_w_end = pads[7];
    auto param = poolOp.param();
    auto pt = param.padding_t().getValue().getLimitedValue();
    auto pb = param.padding_b().getValue().getLimitedValue();
    auto pl = param.padding_l().getValue().getLimitedValue();
    auto pr = param.padding_r().getValue().getLimitedValue();
    pt += pad_h_begin;
    pb += pad_h_end;
    pl += pad_w_begin;
    pr += pad_w_end;

    // rewrite pad
    poolOp.setAttr("param",
           tpu::PoolParam::get(
                poolOp.param().kernel_h(),
                poolOp.param().kernel_w(),
                rewriter.getI32IntegerAttr(pt),
                rewriter.getI32IntegerAttr(pb),
                rewriter.getI32IntegerAttr(pl),
                rewriter.getI32IntegerAttr(pr),
                poolOp.param().stride_h(),
                poolOp.param().stride_w(),
                poolOp.param().do_relu(),
                rewriter.getBoolAttr(true),
                rewriter.getContext()));
  }

  template <class T>
  void updateConvParam(T &convOp, PatternRewriter &rewriter,
                       std::vector<int32_t> &pads) const {
    auto pad_h_begin = pads[2];
    auto pad_w_begin = pads[3];
    auto pad_h_end = pads[6];
    auto pad_w_end = pads[7];

    auto param = convOp.param();
    auto pt = param.padding_t().getValue().getLimitedValue();
    auto pb = param.padding_b().getValue().getLimitedValue();
    auto pl = param.padding_l().getValue().getLimitedValue();
    auto pr = param.padding_r().getValue().getLimitedValue();
    pt += pad_h_begin;
    pb += pad_h_end;
    pl += pad_w_begin;
    pr += pad_w_end;

    // rewrite pad
    convOp.setAttr("param",
           tpu::ConvParam::get(
                convOp.param().stride_h(),
                convOp.param().stride_w(),
                convOp.param().padding(),
                convOp.param().dilation_h(),
                convOp.param().dilation_w(),
                rewriter.getI32IntegerAttr(pt),
                rewriter.getI32IntegerAttr(pb),
                rewriter.getI32IntegerAttr(pl),
                rewriter.getI32IntegerAttr(pr),
                convOp.param().group(),
                convOp.param().is_dw(),
                convOp.param().with_bias(),
                convOp.param().do_relu(),
                rewriter.getContext()));
  }

  template <class T> 
  bool noPaddingPool(T &poolOp) const {
    auto param = poolOp.param();
    auto pt = param.padding_t().getValue().getLimitedValue();
    auto pb = param.padding_b().getValue().getLimitedValue();
    auto pl = param.padding_l().getValue().getLimitedValue();
    auto pr = param.padding_r().getValue().getLimitedValue();
    auto count_include_pad = param.count_include_pad().getValue();

    if (pt == 0 && pb == 0 && pl == 0 && pr == 0)
      return true;
    else if (count_include_pad)
      return true;
    else
      return false;
  }

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto padOp = cast<tpu::PadOp>(op);
    LLVM_DEBUG(llvm::errs() << padOp.getOperationName() << ":"
                            << getOpName(padOp)<< "\n";);

    std::vector<int32_t> pads;
    int const_val = padOp.const_val().convertToFloat();
    arrayAttrToVector(padOp.pads().getValue(), pads);

    auto pad_n_begin = pads[0];
    auto pad_c_begin = pads[1];
    auto pad_n_end = pads[4];
    auto pad_c_end = pads[5];

    if (const_val != 0)
      return matchFailure();

    if (pad_n_begin != 0 || pad_n_end != 0 || 
        pad_c_begin != 0 || pad_c_end != 0)
      return matchFailure();

    for (auto &use : op->getResult(0)->getUses()) {
      auto useOp = use.getOwner();
      if (auto poolOp = dyn_cast<tpu::PoolAvg2DOp>(useOp)) {
        if (!noPaddingPool<tpu::PoolAvg2DOp>(poolOp))
          return matchFailure();
        else
          continue;
      }
      else if (auto poolOp = dyn_cast<tpu::PoolMax2DOp>(useOp)) {
        if (!noPaddingPool<tpu::PoolMax2DOp>(poolOp))
          return matchFailure();
        else
          continue;
      }
      else if (llvm::isa<tpu::Conv2DOp>(useOp)) {
        continue;
      }
      else if (llvm::isa<tpu::DeConv2DOp>(useOp)) {
        continue;
      } else {
        return matchFailure();
      }
    }

    for (auto &use : op->getResult(0)->getUses()) {
      auto useOp = use.getOwner();

      if (llvm::isa<tpu::PoolAvg2DOp>(useOp)) {
        auto poolOp = dyn_cast<tpu::PoolAvg2DOp>(useOp);
        updatePoolParam<tpu::PoolAvg2DOp>(poolOp, rewriter, pads);
      } else if (llvm::isa<tpu::PoolMax2DOp>(useOp)) {
        auto poolOp = dyn_cast<tpu::PoolMax2DOp>(useOp);
        updatePoolParam<tpu::PoolMax2DOp>(poolOp, rewriter, pads);
      } else if (llvm::isa<tpu::Conv2DOp>(useOp)) {
        auto convOp = dyn_cast<tpu::Conv2DOp>(useOp);
        updateConvParam<tpu::Conv2DOp>(convOp, rewriter, pads);
      } else if(llvm::isa<tpu::DeConv2DOp>(useOp)) {
        auto deconvOp = dyn_cast<tpu::DeConv2DOp>(useOp);
        updateConvParam<tpu::DeConv2DOp>(deconvOp, rewriter, pads);
      } else {
        assert("unsupported fused op");
      }
    }
    rewriter.replaceOp(op, {op->getOperand(0)});
    return matchSuccess();
  }
};


class FusePadPass : public FunctionPass<FusePadPass> {
public:
  explicit FusePadPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<TpuFusePadPattern,
                    TpuMergeCropPattern
                   >(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::PadOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<
      TpuFusePadPattern>(context);
}

void tpu::CropOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<
      TpuMergeCropPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFusePadPass() {
  return std::make_unique<FusePadPass>();
}

static PassRegistration<FusePadPass>
    pass("fuse-pad",
         "Fuse pad op into next op (pooling/crop etc)");
