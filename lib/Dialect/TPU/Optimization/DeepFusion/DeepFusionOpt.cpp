//===- DeepFusionOpt.cpp - Implementation Deep Fusion optimization---------===//
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
// This file implements the deep fusion optimization pass.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/SimpleAnalysis.h"

#define DEBUG_TYPE "deep-fusion-opt"

using namespace mlir;

namespace {

struct TpuTL_LA_Conv2DOpPattern : public RewritePattern {
  TpuTL_LA_Conv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_la_conv_2d", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LA_Conv2DOp>(opInst);
    assert(op);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw,
        pad_value;
    parseConvParam(op.param(), false, op.input(), op.output(), n,
                   ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);

    LLVM_DEBUG(llvm::errs() << "TL_LA2LW: layer ID " << getOpLayerId(opInst)
                 << ", convert to LW\n";);

    assert(op.getNumOperands() == 3 && "support 3 inputs only");
    std::vector<Value> newOperands;
    newOperands.push_back(op.getOperand(0));
    newOperands.push_back(op.getOperand(1));
    newOperands.push_back(op.getOperand(2));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));
    attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
    if(op.do_ic_alignment().hasValue()){
      attrs.push_back(rewriter.getNamedAttr("do_ic_alignment", rewriter.getBoolAttr(op.do_ic_alignment().getValue())));
    }

    if (op.do_leaky_relu()) {
      attrs.push_back(rewriter.getNamedAttr("do_leaky_relu", op.do_leaky_reluAttr()));
      if (op.rshift_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_pos", op.rshift_posAttr()));
      if (op.m_i8_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_pos", op.m_i8_posAttr()));
      if (op.rshift_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_neg", op.rshift_negAttr()));
      if (op.m_i8_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_neg", op.m_i8_negAttr()));
    }

    // postpone lmem assignment to next pattern
    uint32_t la_invalid = 0xffffffff;
    attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
    attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
    attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
    attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
    attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

    // create op
    rewriter.replaceOpWithNewOp<tpu::TL_LW_Conv2DOp>(
        op, op.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

struct TpuFuseLeakyReluOpPattern : public RewritePattern {
  TpuFuseLeakyReluOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_leaky_relu", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto lreluOp = dyn_cast<tpu::TG_INT8_LeakyReluOp>(opInst);
    auto prev_op = opInst->getOperand(0).getDefiningOp();
    if (auto convOp = dyn_cast<tpu::TL_LW_Conv2DOp>(prev_op)) {
      int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
      if (lreluOp.m_i8_pos().hasValue()) {
        pos_m_i8 = lreluOp.m_i8_pos().getValue();
        pos_rshift = lreluOp.rshift_pos().getValue();
        assert(pos_m_i8);
      } else {
        pos_m_i8 = 0;
        pos_rshift = 0;
      }
      if (lreluOp.m_i8_neg().hasValue()) {
        neg_m_i8 = lreluOp.m_i8_neg().getValue();
        neg_rshift = lreluOp.rshift_neg().getValue();
        assert(neg_m_i8);
      } else {
        neg_m_i8 = 0;
        neg_rshift = 0;
      }

      convOp->setAttr("do_leaky_relu", rewriter.getBoolAttr(true));
      convOp->setAttr("rshift_pos", rewriter.getI8IntegerAttr(pos_rshift));
      convOp->setAttr("m_i8_pos", rewriter.getI8IntegerAttr(pos_m_i8));
      convOp->setAttr("rshift_neg", rewriter.getI8IntegerAttr(neg_rshift));
      convOp->setAttr("m_i8_neg", rewriter.getI8IntegerAttr(neg_m_i8));
      convOp->setAttr("name", lreluOp.nameAttr());
      rewriter.replaceOp(opInst, {convOp.getResult()});
      return success();
    }

    return failure();
  }
};

struct TpuTL_LW_Conv2DOp_AssignLAddrPattern : public RewritePattern {
  TpuTL_LW_Conv2DOp_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lw_conv_2d", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LW_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw,
        pad_value;
    parseConvParam(op.param(), false, op.input(), op.output(), n,
                   ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
      // assigned already
      return failure();
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
        op->setAttr("la_input", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_output", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - outputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(inputNeuronSizePerLane));
      } else if (op.lm_layout() == "OWI") {
        op->setAttr("la_output", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_input", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - inputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(outputNeuronSizePerLane));
      } else {
        llvm_unreachable("unsupported layout");
      }

      LLVM_DEBUG(llvm::errs() << "TL_LA2LW: layer ID " << getOpLayerId(opInst)
                   << ", Conv, " << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";);

      return success();
    }
  }
};

template<typename OpTy>
struct TpuTL_EltwiseOp_AssignLAddrPattern : public RewritePattern {
  TpuTL_EltwiseOp_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<OpTy>(opInst);

    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    std::vector<int64_t> output_shape;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op.getResult(), output_shape, output_size);
    getNCHW(output_shape, on, oc, oh, ow);
    //bool do_relu = op.do_relu();
    assert(op.getNumOperands() == 2 && "support 2 inputs only");

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
      // assigned already
      return failure();
    }

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, oh, ow, true);
    if (1) {
      if (op.lm_layout() == "IWO") {
        op->setAttr("la_input", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_output", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - outputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(inputNeuronSizePerLane));
      } else if (op.lm_layout() == "OWI") {
        op->setAttr("la_output", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_input", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - inputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(outputNeuronSizePerLane));
      } else {
        llvm_unreachable("unsupported layout");
      }

      LLVM_DEBUG(llvm::errs() << "TL_LA2LW: layer ID " << getOpLayerId(opInst)
                   << OpTy::getOperationName() << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";);

      return success();
    }
  }
};

struct TpuTL_PixelShuffle_AssignLAddrPattern : public RewritePattern {
  TpuTL_PixelShuffle_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_pixel_shuffle", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_PixelShuffleOp>(opInst);

    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    std::vector<int64_t> output_shape;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op.getResult(), output_shape, output_size);
    getNCHW(output_shape, on, oc, oh, ow);

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
      // assigned already
      return failure();
    }

    auto factor = op.factor();
    uint64_t inputNeuronSizePerLane =
            MInfo::getSizePerLane(factor, factor * MInfo::lane_num, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
    if (1) {
      if (op.lm_layout() == "IWO") {
        op->setAttr("la_input", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_output", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - outputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(inputNeuronSizePerLane));
      } else if (op.lm_layout() == "OWI") {
        op->setAttr("la_output", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_input", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - inputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(outputNeuronSizePerLane));
      } else {
        assert(0);
      }

      LLVM_DEBUG(llvm::errs() << "TL_LA2LW: layer ID " << getOpLayerId(opInst)
                   << ", PixelShuffle, " << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";);

      return success();
    }
  }
};

template<typename OpTy>
struct TpuTL_Default_AssignLAddrPattern : public RewritePattern {
  TpuTL_Default_AssignLAddrPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<OpTy>(opInst);

    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    std::vector<int64_t> output_shape;
    int64_t output_size, on, oc, oh, ow;
    getTensorShapeAndSize(op.getResult(), output_shape, output_size);
    getNCHW(output_shape, on, oc, oh, ow);

    assert (op.lm_layout() != "NONE");
    if (op.la_output() != 0xffffffff) {
      // assigned already
      return failure();
    }

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, oh, ow, true);
    if (1) {
      if (op.lm_layout() == "IWO") {
        op->setAttr("la_input", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_output", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - outputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(inputNeuronSizePerLane));
      } else if (op.lm_layout() == "OWI") {
        op->setAttr("la_output", rewriter.getI32IntegerAttr(0));
        op->setAttr("la_input", rewriter.getI32IntegerAttr(
             MInfo::lmem_per_lane - inputNeuronSizePerLane));
        op->setAttr("la_working", rewriter.getI32IntegerAttr(outputNeuronSizePerLane));
      } else {
        assert(0);
      }

      LLVM_DEBUG(llvm::errs() << "TL_LA2LW: layer ID " << getOpLayerId(opInst)
                   << ", LUT, " << op.lm_layout()
                   << ", la_i=" << op.la_input()
                   << ", la_o=" << op.la_output()
                   << ", la_w=" << op.la_working()
                   <<"\n";);

      return success();
    }
  }
};


template<typename OpTy>
struct TpuInsertFakeLdStPattern : public RewritePattern {
  TpuInsertFakeLdStPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<OpTy>(opInst);
    auto prevDefOp = opInst->getOperand(0).getDefiningOp();
    if (isa<tpu::TL_Fake_LoadOp>(prevDefOp)) {
      return failure();
    }

    OpBuilder fakeBuilder(opInst);
    std::vector<NamedAttribute> attrs;

    std::vector<Value> loadOperands;
    loadOperands.push_back(opInst->getOperand(0));

    std::string loadOpName = mlir::getOpName(op).str() + std::string("_load");
    attrs.push_back(rewriter.getNamedAttr("name",
                                         rewriter.getStringAttr(loadOpName)));

    attrs.push_back(rewriter.getNamedAttr("lm_order",
                                          rewriter.getStringAttr("NONE")));
    fakeBuilder.setInsertionPoint(opInst);
    auto loadOp = fakeBuilder.create<tpu::TL_Fake_LoadOp>(
                   opInst->getLoc(), opInst->getOperand(0).getType(),
                   loadOperands, attrs);

    opInst->replaceUsesOfWith(opInst->getOperand(0), loadOp.getResult());

    std::vector<Value> storeOperands;
    storeOperands.push_back(opInst->getResult(0));
    auto uses = opInst->getResult(0).getUses();

    attrs.clear();
    std::string storeOpName = mlir::getOpName(op).str() + std::string("_store");
    attrs.push_back(rewriter.getNamedAttr("name",
                                         rewriter.getStringAttr(storeOpName)));
    attrs.push_back(rewriter.getNamedAttr("lm_order",
                                          rewriter.getStringAttr("NONE")));
    fakeBuilder.setInsertionPointAfter(opInst);
    auto storeOp = fakeBuilder.create<tpu::TL_Fake_StoreOp>(
                   opInst->getLoc(), opInst->getResult(0).getType(),
                   storeOperands, attrs);

    for (auto &use : uses) {
      auto useOp = use.getOwner();
      useOp->replaceUsesOfWith(opInst->getResult(0), storeOp.getResult());
    }
    return success();
  }
};

static bool isLoadCloseToStore(Operation *loadOp) {
  auto prevOp = loadOp->getPrevNode();
  while(isa<tpu::LoadWeightOp>(prevOp)) {
    prevOp = prevOp->getPrevNode();
  }
  assert(prevOp);
  if (!isa<tpu::TL_Fake_StoreOp>(prevOp))
    return false;

  auto defOp = loadOp->getOperand(0).getDefiningOp();
  if (prevOp == defOp)
    return true;
  return false;
}

// some instrurctions must load input data in backend.
static bool isInnerLoadInst(Operation *opInst) {
  if (isa<tpu::TL_PixelShuffleOp>(opInst))
    return true;
  return false;
}

struct TpuMarkLdStFlagPattern : public RewritePattern {
  TpuMarkLdStFlagPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_fake_load", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto nextNormalOp = opInst->getNextNode();
    auto nextStoreOp = nextNormalOp->getNextNode();
    auto prevStoreOp = opInst->getOperand(0).getDefiningOp();

    auto curFakeLoadOp = cast<tpu::TL_Fake_LoadOp>(opInst);
    auto nextFakeStoreOp = cast<tpu::TL_Fake_StoreOp>(nextStoreOp);

    // handle last store
    for (auto &use : nextStoreOp->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (!isa<tpu::TL_Fake_LoadOp>(useOp)) {
          nextFakeStoreOp->setAttr("lm_order", rewriter.getStringAttr("O"));
      }
    }

    // %1 = load(%0)
    // %2 = conv(%1)  ----> prevNormalOp
    // %3 = store(%2) ----> prevStoreOp
    // %4 = load(%3)  ----> opInst
    // %5 = conv(%4)  ----> nextNormalOp
    // %6 = store(%5) ----> nextStoreOp
    if (auto prevFakeStoreOp =
             llvm::dyn_cast_or_null<tpu::TL_Fake_StoreOp>(prevStoreOp)) {
      bool isFirstLoad = isLoadCloseToStore(opInst);
      auto prevNormalOp = prevStoreOp->getOperand(0).getDefiningOp();
      auto tlPrevNormalOp =
           dyn_cast_or_null<tpu::TpuTLSimpleOpCodegenInterface>(prevNormalOp);
      if (tlPrevNormalOp ) {
        if (isFirstLoad) {
          tlPrevNormalOp->setAttr("tl_store_flag", rewriter.getBoolAttr(false));
          if (!prevNormalOp->getResult(0).hasOneUse())
            tlPrevNormalOp->setAttr("tl_store_flag", rewriter.getBoolAttr(true));
        } else {
          tlPrevNormalOp->setAttr("tl_store_flag", rewriter.getBoolAttr(true));
        }
      }
      auto tlNextNormalOp =
           dyn_cast_or_null<tpu::TpuTLSimpleOpCodegenInterface>(nextNormalOp);
      if (tlNextNormalOp) {
        if (isFirstLoad) {
          tlNextNormalOp->setAttr("tl_load_flag", rewriter.getBoolAttr(false));
        } else {
          tlNextNormalOp->setAttr("tl_load_flag", rewriter.getBoolAttr(true));
        }
      }

      // For load input data in backend inst, the prev normal inst need
      // store the result.
      if (isInnerLoadInst(tlNextNormalOp)) {
        tlNextNormalOp->setAttr("tl_load_flag", rewriter.getBoolAttr(true));
        if (tlPrevNormalOp)
          tlPrevNormalOp->setAttr("tl_store_flag", rewriter.getBoolAttr(true));
      }

      if (prevFakeStoreOp.lm_order() == "NONE") {
        if (nextFakeStoreOp.lm_order() == "I") {
          prevFakeStoreOp->setAttr("lm_order", rewriter.getStringAttr("O"));
        } else if (nextFakeStoreOp.lm_order() == "O") {
          prevFakeStoreOp->setAttr("lm_order", rewriter.getStringAttr("I"));
        } else {
          assert("unset lm_order");
        }
      }

      if (curFakeLoadOp.lm_order() == "NONE") {
        if (nextFakeStoreOp.lm_order() == "I") {
          curFakeLoadOp->setAttr("lm_order", rewriter.getStringAttr("O"));
        } else if (nextFakeStoreOp.lm_order() == "O") {
          curFakeLoadOp->setAttr("lm_order", rewriter.getStringAttr("I"));
        } else {
          assert("unset lm_order");
        }
      }
    } else {
      for (auto &use : opInst->getResult(0).getUses()) {
        auto useOp = use.getOwner();
        auto tlUseOp =
           dyn_cast_or_null<tpu::TpuTLSimpleOpCodegenInterface>(useOp);
        if (tlUseOp)
          tlUseOp->setAttr("tl_load_flag", rewriter.getBoolAttr(true));
        }
      if (nextFakeStoreOp.lm_order() == "I") {
        curFakeLoadOp->setAttr("lm_order", rewriter.getStringAttr("O"));
      } else if (nextFakeStoreOp.lm_order() == "O") {
        curFakeLoadOp->setAttr("lm_order", rewriter.getStringAttr("I"));
      } else {
        assert("unset lm_order");
      }
    }

    return success();
  }
};


struct TpuMarkLayoutPattern: public RewritePattern {
  TpuMarkLayoutPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_fake_load", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto normalOp = opInst->getNextNode();
    auto nextFakeOp = normalOp->getNextNode();
    auto fakeLoadOp = dyn_cast_or_null<tpu::TL_Fake_LoadOp>(opInst);
    auto fakeStoreOp = dyn_cast_or_null<tpu::TL_Fake_StoreOp>(nextFakeOp);
    assert(normalOp);
    assert(fakeStoreOp);
    // %1 = load(%0)
    // %2 = conv(%1)
    // %3 = store(%2)

    auto tlNormalOp =
         dyn_cast_or_null<tpu::TpuTLSimpleOpCodegenInterface>(normalOp);

    if (tlNormalOp) {
      if (fakeLoadOp.lm_order() == "I") {
        tlNormalOp->setAttr("lm_layout", rewriter.getStringAttr("IWO"));
        assert(fakeStoreOp.lm_order() == "O");
      } else if (fakeLoadOp.lm_order() == "O") {
        tlNormalOp->setAttr("lm_layout", rewriter.getStringAttr("OWI"));
        assert(fakeStoreOp.lm_order() == "I");
      } else {
        assert("unset lm_order");
      }
    }
    return success();
  }
};

template<typename OpTy>
struct TpuDeleteFakeLdStPattern : public RewritePattern {
  TpuDeleteFakeLdStPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (auto loadOp = dyn_cast_or_null<tpu::TL_Fake_LoadOp>(opInst)) {
      auto useLoadOp = opInst->getNextNode();
      useLoadOp->replaceUsesOfWith(opInst->getResult(0), opInst->getOperand(0));
    } else if (auto storeOp = dyn_cast_or_null<tpu::TL_Fake_StoreOp>(opInst)) {
      auto opdDefOp = opInst->getPrevNode();
      opInst->replaceAllUsesWith(opdDefOp);
    }
    return success();
  }
};


class DeepFusionOpt : public mlir::PassWrapper<DeepFusionOpt, FunctionPass> {
public:
  explicit DeepFusionOpt() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    MInfo::getChipInfo(fn);
    assert(MInfo::version && "refer to set-chip");

    OwningRewritePatternList patterns;
    patterns.insert<
        TpuTL_LA_Conv2DOpPattern,
        TpuFuseLeakyReluOpPattern
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuInsertFakeLdStPattern<tpu::TL_LW_Conv2DOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_PoolAvg2DOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_EltwiseAddOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_EltwiseMulOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_LutOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_ScaleOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_PixelShuffleOp>,
                    TpuInsertFakeLdStPattern<tpu::TL_PReluOp>
                   >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuMarkLdStFlagPattern
                   >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuMarkLayoutPattern
                   >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuDeleteFakeLdStPattern<tpu::TL_Fake_LoadOp>,
                    TpuDeleteFakeLdStPattern<tpu::TL_Fake_StoreOp>
                   >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        TpuTL_LW_Conv2DOp_AssignLAddrPattern,
        TpuTL_PixelShuffle_AssignLAddrPattern,
        TpuTL_EltwiseOp_AssignLAddrPattern<tpu::TL_EltwiseAddOp>,
        TpuTL_EltwiseOp_AssignLAddrPattern<tpu::TL_EltwiseMulOp>,
        TpuTL_Default_AssignLAddrPattern<tpu::TL_LutOp>,
        TpuTL_Default_AssignLAddrPattern<tpu::TL_PoolAvg2DOp>,
        TpuTL_Default_AssignLAddrPattern<tpu::TL_ScaleOp>,
        TpuTL_Default_AssignLAddrPattern<tpu::TL_PReluOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createDeepFusionOpt() {
  return std::make_unique<DeepFusionOpt>();
}
