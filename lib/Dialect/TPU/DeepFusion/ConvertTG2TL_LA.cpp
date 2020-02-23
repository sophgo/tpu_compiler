//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "MachineInfo.h"

#define DEBUG_TYPE "deep-fusion-tg2tl-la"

using namespace mlir;

namespace {

struct TpuTG2TLConv2DOpPattern : public RewritePattern {
  TpuTG2TLConv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
#if 1
    assert(false);
#else
    auto op = cast<tpu::Conv2DOp>(opInst);
    //auto loc = op->getLoc();
    assert(op.quant() == "INT8_MULTIPLIER"
           && "TG2TL support INT8_MULTIPLIER mode only");

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
    uint64_t filterSizePerLane = 0;
    // filter working size *2 for double buffer
    if (g != oc) {
      assert(g == 1);
      // for non-dw conv, assuming oc_step = lane_num
      int oc_step = MInfo::lane_num;
      filterSizePerLane = MInfo::getSizePerLane(ic, oc_step, kh, kw, false) * 2;
    } else {
      // for dw conv, load weight all in once
      filterSizePerLane = MInfo::getSizePerLane(1, oc, kh, kw, false) * 2;
    }
    // load bias all in once
    int bias_size = with_bias ? 9 : 5;
    uint64_t biasSizePerLane = MInfo::getSizePerLane(1, oc, 1, bias_size, false);
    // if eltwise sum is enabled, eltwise input size
    uint64_t eltwiseInputSizePerLane = 0;
    uint64_t eltwiseWorkingSizePerLane = 0;
    if (do_eltwise) {
      eltwiseInputSizePerLane = outputNeuronSizePerLane;
      #define MIN_eltwise_working_size    (32)
      eltwiseWorkingSizePerLane = MIN_eltwise_working_size * 2;
    }
    // total
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                            + filterSizePerLane + biasSizePerLane
                            + eltwiseInputSizePerLane + eltwiseWorkingSizePerLane;
    if (totalPerLane > MInfo::lmem_per_lane) {
      llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem needed " << totalPerLane
                   << "\n";
      return matchFailure();
    }

    if (1) {
      llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";

      assert(op.getNumOperands() == 3);
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));
      newOperands.push_back(op.getOperand(2));

      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
      attrs.push_back(rewriter.getNamedAttr("dilation_h_factor", rewriter.getI32IntegerAttr(dh)));
      attrs.push_back(rewriter.getNamedAttr("dilation_w_factor", rewriter.getI32IntegerAttr(dw)));
      attrs.push_back(rewriter.getNamedAttr("padding", rewriter.getStringAttr(op.padding())));
      attrs.push_back(rewriter.getNamedAttr("stride_h", rewriter.getI32IntegerAttr(sh)));
      attrs.push_back(rewriter.getNamedAttr("stride_w", rewriter.getI32IntegerAttr(sw)));
      attrs.push_back(rewriter.getNamedAttr("group", rewriter.getI32IntegerAttr(g)));
      attrs.push_back(rewriter.getNamedAttr("fused_activation_function",
          rewriter.getStringAttr(op.fused_activation_function())));

      attrs.push_back(rewriter.getNamedAttr("offset", rewriter.getI64IntegerAttr(op.offset().getValue().getLimitedValue())));
      attrs.push_back(rewriter.getNamedAttr("threshold_y", rewriter.getF32FloatAttr(op.threshold_y().getValue().convertToFloat())));
      attrs.push_back(rewriter.getNamedAttr("layer_id", rewriter.getI32IntegerAttr(op.layer_id().getValue().getLimitedValue())));
      rewriter.replaceOpWithNewOp<tpu::TL_LA_Conv2DOp>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
#endif
  }
};

class DeepFusionTG2TL_LA : public FunctionPass<DeepFusionTG2TL_LA> {
public:
  explicit DeepFusionTG2TL_LA() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuTG2TLConv2DOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionTG2TL_LA() {
  return std::make_unique<DeepFusionTG2TL_LA>();
}

static PassRegistration<DeepFusionTG2TL_LA>
    pass("deep-fusion-tg2tl-la",
         "convert Ops from TG to TL, "
         "this is a trivial conversion, yielding no improvement at all");
