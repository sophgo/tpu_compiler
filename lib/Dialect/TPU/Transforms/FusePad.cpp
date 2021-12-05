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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_pad"

using namespace mlir;

namespace {
struct TpuMergeCropPattern : public RewritePattern {
  TpuMergeCropPattern(MLIRContext *context)
      : RewritePattern("tpu.crop", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto crop0Op = dyn_cast<tpu::CropOp>(op);
    auto nextOp = getNextOp(op);
    if (nextOp == nullptr) {
      return failure();
    }
    auto crop1Op = dyn_cast<tpu::CropOp>(nextOp);
    if (crop1Op == nullptr) {
      return failure();
    }
    std::vector<int> crop0Offset;
    std::vector<int> crop1Offset;
    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    SmallVector<Attribute, 4> mergeOffsetAttr;

    operands.push_back(op->getOperand(0));
    arrayAttrToVector(crop0Op.crop_offset(), crop0Offset);
    arrayAttrToVector(crop1Op.crop_offset(), crop1Offset);
    if (crop0Op.steps().hasValue()) {
      std::vector<int> steps;
      arrayAttrToVector(crop0Op.steps().getValue(), steps);
      int total = std::accumulate(steps.begin(), steps.end(), 1,
                                  std::multiplies<int32_t>());
      if (total != 1) {
        return failure();
      }
    }
    if (crop1Op.steps().hasValue()) {
      std::vector<int> steps;
      arrayAttrToVector(crop1Op.steps().getValue(), steps);
      int total = std::accumulate(steps.begin(), steps.end(), 1,
                                  std::multiplies<int32_t>());
      if (total != 1) {
        return failure();
      }
    }

    for (unsigned int i = 0; i < crop0Offset.size(); i++) {
      auto cropOffset = crop0Offset[i] + crop1Offset[i];
      auto cropOffsetAttr = rewriter.getI32IntegerAttr(cropOffset);
      mergeOffsetAttr.push_back(cropOffsetAttr);
    }

    attrs.push_back(rewriter.getNamedAttr("name", crop1Op.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "crop_offset", rewriter.getArrayAttr(mergeOffsetAttr)));
    attrs.push_back(rewriter.getNamedAttr("quant", crop1Op.quantAttr()));

    auto mergeCropOp = rewriter.create<tpu::CropOp>(
        op->getLoc(), crop1Op.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    rewriter.replaceOp(crop1Op, {mergeCropOp.getResult()});
    return success();
  }
};

struct TpuFusePadPattern : public RewritePattern {
  TpuFusePadPattern(MLIRContext *context)
      : RewritePattern("tpu.pad", 1, context) {}

  template <class T>
  void updatePoolParam(T &poolOp, PatternRewriter &rewriter,
                       std::vector<int32_t> &pads, int pad_value) const {

    auto pad_h_begin = pads[2];
    auto pad_w_begin = pads[3];
    auto pad_h_end = pads[6];
    auto pad_w_end = pads[7];
    auto param = poolOp.param();
    auto pt = param.padding_t().getInt();
    auto pb = param.padding_b().getInt();
    auto pl = param.padding_l().getInt();
    auto pr = param.padding_r().getInt();
    pt += pad_h_begin;
    pb += pad_h_end;
    pl += pad_w_begin;
    pr += pad_w_end;

    // rewrite pad
    poolOp->setAttr(
        "param",
        tpu::PoolParam::get(
            poolOp.param().kernel_h(), poolOp.param().kernel_w(),
            rewriter.getI32IntegerAttr(pt), rewriter.getI32IntegerAttr(pb),
            rewriter.getI32IntegerAttr(pl), rewriter.getI32IntegerAttr(pr),
            rewriter.getI32IntegerAttr(pad_value), poolOp.param().stride_h(),
            poolOp.param().stride_w(), poolOp.param().do_relu(),
            rewriter.getBoolAttr(true), rewriter.getContext()));
  }

  template <class T>
  void updateConvParam(T &convOp, PatternRewriter &rewriter,
                       std::vector<int32_t> &pads, int pad_value) const {
    auto pad_h_begin = pads[2];
    auto pad_w_begin = pads[3];
    auto pad_h_end = pads[6];
    auto pad_w_end = pads[7];

    auto param = convOp.param();
    auto pt = param.padding_t().getInt();
    auto pb = param.padding_b().getInt();
    auto pl = param.padding_l().getInt();
    auto pr = param.padding_r().getInt();
    pt += pad_h_begin;
    pb += pad_h_end;
    pl += pad_w_begin;
    pr += pad_w_end;

    // rewrite pad
    convOp->setAttr(
        "param",
        tpu::ConvParam::get(
            convOp.param().kernel_h(), convOp.param().kernel_w(),
            convOp.param().stride_h(), convOp.param().stride_w(),
            convOp.param().padding(), convOp.param().dilation_h(),
            convOp.param().dilation_w(), rewriter.getI32IntegerAttr(pt),
            rewriter.getI32IntegerAttr(pb), rewriter.getI32IntegerAttr(pl),
            rewriter.getI32IntegerAttr(pr), convOp.param().group(),
            convOp.param().is_dw(), convOp.param().with_bias(),
            convOp.param().do_relu(), convOp.param().ins(),
            rewriter.getI32IntegerAttr(pad_value), rewriter.getContext()));
  }

  template <class T>
  bool noPaddingPool(T &poolOp) const {
    auto param = poolOp.param();
    auto pt = param.padding_t().getInt();
    auto pb = param.padding_b().getInt();
    auto pl = param.padding_l().getInt();
    auto pr = param.padding_r().getInt();
    auto count_include_pad = param.count_include_pad().getValue();

    if (pt == 0 && pb == 0 && pl == 0 && pr == 0)
      return true;
    else if (count_include_pad)
      return true;
    else
      return false;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto padOp = cast<tpu::PadOp>(op);
    LLVM_DEBUG(llvm::errs() << padOp.getOperationName() << ":"
                            << getOpName(padOp) << "\n";);

    std::vector<int32_t> pads;
    int const_val = padOp.const_val().convertToFloat();
    arrayAttrToVector(padOp.pads(), pads);
    if (pads.size() != 8) {
      return failure();
    }

    auto pad_n_begin = pads[0];
    auto pad_c_begin = pads[1];
    auto pad_n_end = pads[4];
    auto pad_c_end = pads[5];

    if (padOp.mode().str() == "edge") {
      return failure();
    }

    if (pad_n_begin != 0 || pad_n_end != 0 || pad_c_begin != 0 ||
        pad_c_end != 0)
      return failure();

    const int PAD_H_MAX = 15;
    const int PAD_W_MAX = 15;
    auto pad_h_begin = pads[2];
    auto pad_w_begin = pads[3];
    auto pad_h_end = pads[6];
    auto pad_w_end = pads[7];
    if (pad_h_begin > PAD_H_MAX || pad_h_end > PAD_H_MAX ||
        pad_w_begin > PAD_W_MAX || pad_w_end > PAD_W_MAX)
      return failure();

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (auto poolOp = dyn_cast<tpu::PoolAvg2DOp>(useOp)) {
        if (!noPaddingPool<tpu::PoolAvg2DOp>(poolOp))
          return failure();
        else
          continue;
      } else if (auto poolOp = dyn_cast<tpu::PoolMax2DOp>(useOp)) {
        if (!noPaddingPool<tpu::PoolMax2DOp>(poolOp))
          return failure();
        else
          continue;
      } else if (llvm::isa<tpu::Conv2DOp>(useOp)) {
        continue;
      } else if (llvm::isa<tpu::DeConv2DOp>(useOp)) {
        continue;
      } else {
        return failure();
      }
    }

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();

      if (llvm::isa<tpu::PoolAvg2DOp>(useOp)) {
        auto poolOp = dyn_cast<tpu::PoolAvg2DOp>(useOp);
        updatePoolParam<tpu::PoolAvg2DOp>(poolOp, rewriter, pads, const_val);
      } else if (llvm::isa<tpu::PoolMax2DOp>(useOp)) {
        auto poolOp = dyn_cast<tpu::PoolMax2DOp>(useOp);
        updatePoolParam<tpu::PoolMax2DOp>(poolOp, rewriter, pads, const_val);
      } else if (llvm::isa<tpu::Conv2DOp>(useOp)) {
        auto convOp = dyn_cast<tpu::Conv2DOp>(useOp);
        updateConvParam<tpu::Conv2DOp>(convOp, rewriter, pads, const_val);
      } else if (llvm::isa<tpu::DeConv2DOp>(useOp)) {
        auto deconvOp = dyn_cast<tpu::DeConv2DOp>(useOp);
        updateConvParam<tpu::DeConv2DOp>(deconvOp, rewriter, pads, const_val);
      } else {
        assert("unsupported fused op");
      }
    }
    LLVM_DEBUG(llvm::errs()
                   << "fused pad op: " << getOpName(op) << " to op:"
                   << getOpName(op->getOperand(0).getDefiningOp()) << "\n";);
    rewriter.replaceOp(op, {op->getOperand(0)});
    return success();
  }
};

class FusePadPass : public mlir::PassWrapper<FusePadPass, FunctionPass> {
public:
  explicit FusePadPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<TpuFusePadPattern, TpuMergeCropPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::PadOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<TpuFusePadPattern>(context);
}

void tpu::CropOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<TpuMergeCropPattern>(context);
}

std::unique_ptr<mlir::Pass> mlir::createFusePadPass() {
  return std::make_unique<FusePadPass>();
}
