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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
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
#include "SimpleAnalysis.h"

#define DEBUG_TYPE "deep-fusion-tl-la2lw"

using namespace mlir;

namespace {

struct TpuTL_LA_Conv2DOpPattern : public RewritePattern {
  TpuTL_LA_Conv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_la_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LA_Conv2DOp>(opInst);
    assert(op);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    if (1) {
      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                   << ", convert to LW\n";

      assert(op.getNumOperands() == 3);
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));
      newOperands.push_back(op.getOperand(2));

      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));
      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));

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

struct TpuTL_LW_Conv2DOp_MarkShortPathPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_MarkShortPathPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    if (op.in_short_path().hasValue()) {
      return matchFailure();
    }

    if (! op.getResult()->hasOneUse()) {
      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                 << ", Conv2D " << op.name()
                 << " has more than one Use " << op.getResult()->hasOneUse() << "\n";
      op.setAttr("in_short_path", rewriter.getBoolAttr(false));
      return matchSuccess();
    }
    assert(op.getResult()->hasOneUse());

    // TODO: this is a naive version, looking for one step path, mark as short
    auto next_op = getNextOp(op);
    auto prev_op = op.getOperand(0);

    if (!prev_op->hasOneUse() && isa<tpu::TL_EltwiseAddOp>(next_op)) {
      op.setAttr("in_short_path", rewriter.getBoolAttr(true));
    } else {
      op.setAttr("in_short_path", rewriter.getBoolAttr(false));
    }
    return matchSuccess();
  }
};

struct TpuTL_LW_Conv2DOp_AssignLayoutPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_AssignLayoutPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    //bool is_dw, with_bias, do_relu;
    //int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    //parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
    //               n, ic, ih, iw, oc, oh, ow, g,
    //               kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    if (op.lm_layout() != "NONE") {
      // assigned already
      return matchFailure();
    }

    if (! op.getResult()->hasOneUse()) {
      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                 << ", Conv2D " << op.name()
                 << " has more than one Use " << op.getResult()->hasOneUse() << "\n";
      std::vector<Operation *> conv_ops;
      std::vector<Operation *> elta_ops;
      for (auto &use : op.getResult()->getUses()) {
        Operation *next_opInst = use.getOwner();
        if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(next_opInst)) {
          // next is TL_LW_Conv2DOp
          if (next_op.lm_layout() == "NONE") {
            // next_op has not been assign layout, return for now
            return matchSuccess();
          }
          conv_ops.push_back(next_opInst);
        } else if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_EltwiseAddOp>(next_opInst)) {
          if (next_op.lm_layout() == "NONE") {
            // next_op has not been assign layout, return for now
            return matchSuccess();
          }
          elta_ops.push_back(next_opInst);
        }
      }
      if (conv_ops.size() == 2) {
        // inception_v3, two conv Ops, steal them
        auto conv_op_next = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[0]);
        auto conv_op_steal = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[1]);
        if (conv_op_next.lm_layout() == "IWO") {
          op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        } else if (conv_op_next.lm_layout() ==  "OWI") {
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        } else {
          assert(0);
        }
        // steal the op
        op.setAttr("tl_store_flag", rewriter.getBoolAttr(false));
        conv_op_next.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
        conv_op_steal.setAttr("lm_layout",
            rewriter.getStringAttr(conv_op_next.lm_layout()));
        conv_op_steal.setAttr("tl_load_flag", rewriter.getBoolAttr(false));

        return matchSuccess();
      } else if (conv_ops.size() == 1) {
        assert(elta_ops.size() == 1);
        // mobilenet_v2, one conv and one eltwise
        // fuse with the conv
        auto conv_op_next = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[0]);
        if (conv_op_next.lm_layout() == "IWO") {
          op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        } else if (conv_op_next.lm_layout() ==  "OWI") {
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        } else {
          assert(0);
        }
        // however, since the op has another use, the current OP needs to do store
        op.setAttr("tl_store_flag", rewriter.getBoolAttr(true));
        // next_conv can skip loading though
        conv_op_next.setAttr("tl_load_flag", rewriter.getBoolAttr(false));

        return matchSuccess();
      } else {
        assert(false);
      }
    }

    assert(op.getResult()->hasOneUse());
    Operation *next_opInst = getNextOp(op);
    if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(next_opInst)) {
      // next is another TL_LW_Conv2DOp
      if (next_op.lm_layout() == "NONE") {
        // next_op not set layout yet, return for now, wait for next round
        return matchSuccess();
      }
      if (next_op.lm_layout() == "IWO") {
        op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
      } else if (next_op.lm_layout() == "OWI") {
        op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
      } else {
        assert(0);
      }
      op.setAttr("tl_store_flag", rewriter.getBoolAttr(false));
      next_op.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
    } else if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_EltwiseAddOp>(next_opInst)) {
      // next is TL_EltwiseAddOp
      assert(op.in_short_path().hasValue());
      if (op.in_short_path().getValue()) {
        // for short path Conv Op
        // TODO: to make this generic
        // CAUTION: for resnet50 only for now
        // Asumption 1: short path is always walked before long path
        // Asumption 2: there is only one step in this short path
        // with the above 2 asumptions, we can steal this op
        // we have to terminate fuse on the conv Op, by "store" it
        // treat as new chain
        op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
      } else if (next_op.lm_layout() == "NONE") {
        // next_op not set layout yet, return for now, wait for next round
        return matchSuccess();
      } else {
        // for the long path conv Op, fuse them
        if (next_op.lm_layout() == "IWO") {
          op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        } else if (next_op.lm_layout() == "OWI") {
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        } else {
          assert(0);
        }
        op.setAttr("tl_store_flag", rewriter.getBoolAttr(false));
        next_op.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
      }
    } else {
      // next op is not another TL_Op, start a new chain
      op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
    }

    llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                 << ", Conv LM_LAYOUT " << op.lm_layout() << "\n";
    return matchSuccess();
  }
};

struct TpuTL_EltwiseAddOp_AssignLayoutPattern : public RewritePattern {
  TpuTL_EltwiseAddOp_AssignLayoutPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_eltwise_add", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_EltwiseAddOp>(opInst);

    if (op.lm_layout() != "NONE") {
      // assigned already
      return matchFailure();
    }

    if (op.getResult()->hasOneUse()) {

      // one user case
      Operation *next_opInst = getNextOp(op);
      assert(!isa<tpu::TL_EltwiseAddOp>(next_opInst));
      if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(next_opInst)) {
        // next is TL_LW_Conv2DOp, fuse it
        if (next_op.lm_layout() == "NONE") {
          // next_op not set layout yet, return for now, wait for next round
          return matchSuccess();
        }
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
        // start a new chain
        op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
      }
    } else {

      // for 2 users EltwiseAdd case
      // a. both are Conv
      //    we can steal an Op if both assumptions are met
      //    Asumption 1: short path is always walked before long path
      //    Asumption 2: there is only one step in this short path
      // b. one is Conv, another is EltwiseAdd
      std::vector<Operation *> conv_ops;
      std::vector<Operation *> elta_ops;
      for (auto &use : op.getResult()->getUses()) {
        Operation *next_opInst = use.getOwner();
        if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(next_opInst)) {
          // next is TL_LW_Conv2DOp
          if (next_op.lm_layout() == "NONE") {
            // next_op has not been assign layout, return for now
            return matchSuccess();
          }
          conv_ops.push_back(next_opInst);
        } else if (auto next_op = llvm::dyn_cast_or_null<tpu::TL_EltwiseAddOp>(next_opInst)) {
          if (next_op.lm_layout() == "NONE") {
            // next_op has not been assign layout, return for now
            return matchSuccess();
          }
          elta_ops.push_back(next_opInst);
        }
      }

      if (conv_ops.empty() && elta_ops.empty()) {
        // start a new chain
        op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
      } else if (conv_ops.empty() && elta_ops.size() == 1) {
        // one elta case
        assert(0);
      } else if (elta_ops.empty() && conv_ops.size() == 1) {
        // one conv case
        assert(0);
      } else if (elta_ops.empty() && conv_ops.size() == 2) {
        // two conv cast
        auto conv_op_0 = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[0]);
        auto conv_op_1 = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[1]);
        int next_op_idx;
        if (conv_op_0.in_short_path().getValue()) {
          assert(!conv_op_1.in_short_path().getValue());
          // conv_op_1 is the long path
          next_op_idx = 1;
        } else {
          assert(conv_op_1.in_short_path().getValue());
          // convOps[0] is the long path
          next_op_idx = 0;
        }

        auto conv_op_next = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[next_op_idx]);
        auto conv_op_short = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[1 - next_op_idx]);
        if (conv_op_next.lm_layout() == "IWO") {
          op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        } else if (conv_op_next.lm_layout() ==  "OWI") {
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        } else {
          assert(0);
        }
        // steal the op
        op.setAttr("tl_store_flag", rewriter.getBoolAttr(false));
        conv_op_next.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
        conv_op_short.setAttr("lm_layout",
            rewriter.getStringAttr(conv_op_next.lm_layout()));
        conv_op_short.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
        // TODO: make sure the short path been walked first
      } else if (elta_ops.size() == 1 && conv_ops.size() == 1) {
        // one conv and one elt case
        auto conv_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(conv_ops[0]);
        if (conv_op.lm_layout() == "IWO") {
          op.setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        } else if (conv_op.lm_layout() ==  "OWI") {
          op.setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        } else {
          assert(0);
        }
        conv_op.setAttr("tl_load_flag", rewriter.getBoolAttr(false));
      } else {
        assert(0);
      }
    }
    llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                 << ", EltA LM_LAYOUT " << op.lm_layout()
                 << ", LD " << op.tl_load_flag()
                 << ", ST " << op.tl_store_flag()
                 << "\n";

    return matchSuccess();
  }
};

struct TpuTL_LW_Conv2DOp_AssignLAddrPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
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

      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                   << ", Conv, " << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";

      return matchSuccess();
    }
  }
};

struct TpuTL_EltwiseAddOp_AssignLAddrPattern : public RewritePattern {
  TpuTL_EltwiseAddOp_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_eltwise_add", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_EltwiseAddOp>(opInst);

    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    //bool do_relu = op.do_relu();
    assert(op.getNumOperands() == 2 && "support 2 inputs only");

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
      // assigned already
      return matchFailure();
    }

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
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

      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id()
                   << ", EltA, " << op.lm_layout()
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
    patterns.insert<
        TpuTL_LA_Conv2DOpPattern
        >(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<
        TpuTL_LW_Conv2DOp_MarkShortPathPattern
        >(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<
        TpuTL_LW_Conv2DOp_AssignLayoutPattern,
        TpuTL_EltwiseAddOp_AssignLayoutPattern
        >(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<
        TpuTL_LW_Conv2DOp_AssignLAddrPattern,
        TpuTL_EltwiseAddOp_AssignLAddrPattern
        >(context);
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
