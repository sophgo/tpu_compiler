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

#define DEBUG_TYPE "deep-fusion-tl-la2lw"

using namespace mlir;

namespace {

struct TpuTL_LA_Conv2DOpPattern : public RewritePattern {
  TpuTL_LA_Conv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_la_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LA_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    if (1) {
      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id() << "\n";

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

      // postpone lmem assignment to next pattern
      uint32_t la_invalid = 0xffffffff;
      attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

      rewriter.replaceOpWithNewOp<tpu::TL_LW_Conv2DOp>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

struct TpuTL_LW_Conv2DOp_AssignLayoutPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_AssignLayoutPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    if (op.lm_layout() != "NONE") {
      // assigned already
      return matchFailure();
    }

    if (1) {
      //assert(op.getNumOperands() == 3);
      //auto prevOpInst = op.getOperand(0)->getDefiningOp();
      //auto prevOp = cast<tpu::TL_LW_Conv2DOp>(prevOpInst);
      assert(op.getResult()->hasOneUse());
      for (auto &use : op.getResult()->getUses()) {
        Operation *operandOp = use.getOwner();
        if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(operandOp)) {
          // next is another TL_LW_Conv2DOp
          assert(next_op.lm_layout() != "NONE");
          if (next_op.lm_layout() == "IWO") {
            op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
          } else if (next_op.lm_layout() == "OWI") {
            op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
          } else {
            assert(0);
          }
          op.setAttr("tl_store_flag", rewriter.getBoolAttr(false));
          next_op.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
        } else {
          // use op is not another TL_LW_Conv2DOp, end of chain
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        }
      }

      llvm::errs() << "TL_LW: layer ID " << op.layer_id()
                   << ", assign LM_LAYOUT " << op.lm_layout() << "\n";

      return matchSuccess();
    }
  }
};


struct TpuTL_LW_Conv2DOp_AssignLAddrPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    assert (op.lm_layout() != "NONE");
    uint32_t la_invalid = 0xffffffff;
    if (op.la_output() != la_invalid) {
      // assigned already
      return matchFailure();
    }

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
    uint64_t workingSizePerLane = filterSizePerLane + biasSizePerLane;
    assert(MInfo::lmem_per_lane >=
        inputNeuronSizePerLane + outputNeuronSizePerLane + workingSizePerLane);
    if (1) {
      if (op.lm_layout() == "IWO") {
        op.setAttr("la_input", rewriter.getI32IntegerAttr(0));
        op.setAttr("la_output", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - outputNeuronSizePerLane));
        op.setAttr("la_working", rewriter.getI32IntegerAttr(inputNeuronSizePerLane));
      } else if (op.lm_layout() == "OWI") {
        op.setAttr("la_output", rewriter.getI32IntegerAttr(0));
        op.setAttr("la_input", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - inputNeuronSizePerLane));
        op.setAttr("la_working", rewriter.getI32IntegerAttr(outputNeuronSizePerLane));
      } else {
        assert(0);
      }

      llvm::errs() << "TL_LW: layer ID " << op.layer_id()
                   << ", " << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";

      return matchSuccess();
    }
  }
};

class DeepFusionTL_LA2LW : public FunctionPass<DeepFusionTL_LA2LW> {
public:
  explicit DeepFusionTL_LA2LW() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuTL_LA_Conv2DOpPattern>(context);
    patterns.insert<TpuTL_LW_Conv2DOp_AssignLayoutPattern>(context);
    patterns.insert<TpuTL_LW_Conv2DOp_AssignLAddrPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionTL_LA2LW() {
  return std::make_unique<DeepFusionTL_LA2LW>();
}

static PassRegistration<DeepFusionTL_LA2LW>
    pass("deep-fusion-tl-la2lw",
         "convert TL Conv Ops from LA to LW.");
