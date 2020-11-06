//===- FuseRelu.cpp - fuse relu -------------------------------------------===//
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
// This file implements the fusion of relu.
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
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_relu"

using namespace mlir;

namespace {

static bool supportRelu(Operation *op) {
  if (matchPattern(op, m_Op<tpu::Conv2DOp>()) ||
      matchPattern(op, m_Op<tpu::DeConv2DOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseAddOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMaxOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMinOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMulOp>()) ||
      matchPattern(op, m_Op<tpu::FullyConnectedOp>()) ||
      matchPattern(op, m_Op<tpu::BroadcastMulOp>()) ||
      matchPattern(op, m_Op<tpu::ConcatOp>()) ||
      matchPattern(op, m_Op<tpu::ScaleOp>()) ||
      matchPattern(op, m_Op<tpu::ConcatOp>())) {
    return true;
  }
  return false;
}

struct TpuFuseReluPattern : public RewritePattern {
  TpuFuseReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    LLVM_DEBUG(llvm::errs() << reluOp.getOperationName() << ":"
                            << getOpName(reluOp)<< "\n";);

    // match relu Op that is following conv or eltwise Ops
    auto formerOp = op->getOperand(0)->getDefiningOp();
    if (!formerOp->getResult(0)->hasOneUse())
      return matchFailure();

    if (matchPattern(formerOp, m_Op<tpu::Conv2DOp>())) {
      auto convOp = cast<tpu::Conv2DOp>(formerOp);
      if(convOp.param().do_relu().getValue() == true){
        LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                                << getOpName(convOp) << " is already fused relu"
                                << "\n";);
        return matchFailure();
      }
      // set do_relu
      convOp.setAttr("param",
          tpu::ConvParam::get(
              convOp.param().stride_h(),
              convOp.param().stride_w(),
              convOp.param().padding(),
              convOp.param().dilation_h(),
              convOp.param().dilation_w(),
              convOp.param().padding_t(),
              convOp.param().padding_b(),
              convOp.param().padding_l(),
              convOp.param().padding_r(),
              convOp.param().group(),
              convOp.param().is_dw(),
              convOp.param().with_bias(),
              rewriter.getBoolAttr(true),
              convOp.param().ins(),
              convOp.param().pad_value(),
              rewriter.getContext()));
      convOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::DeConv2DOp>())) {
      auto deconvOp = cast<tpu::DeConv2DOp>(formerOp);
      if(deconvOp.param().do_relu().getValue() == true){
        LLVM_DEBUG(llvm::errs() << deconvOp.getOperationName() << ":"
                                << getOpName(deconvOp) << " is already fused relu"
                                << "\n";);
        return matchFailure();
      }
      // set do_relu
      deconvOp.setAttr("param",
          tpu::ConvParam::get(
              deconvOp.param().stride_h(),
              deconvOp.param().stride_w(),
              deconvOp.param().padding(),
              deconvOp.param().dilation_h(),
              deconvOp.param().dilation_w(),
              deconvOp.param().padding_t(),
              deconvOp.param().padding_b(),
              deconvOp.param().padding_l(),
              deconvOp.param().padding_r(),
              deconvOp.param().group(),
              deconvOp.param().is_dw(),
              deconvOp.param().with_bias(),
              rewriter.getBoolAttr(true),
              deconvOp.param().ins(),
              deconvOp.param().pad_value(),
              rewriter.getContext()));
      deconvOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseAddOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      assert(!eltOp.do_relu() && "done relu");
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMaxOp>())) {
      auto eltOp = cast<tpu::EltwiseMaxOp>(formerOp);
      assert(!eltOp.do_relu() && "done relu");
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMinOp>())) {
      auto eltOp = cast<tpu::EltwiseMinOp>(formerOp);
      assert(!eltOp.do_relu() && "done relu");
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    }  else if (matchPattern(formerOp, m_Op<tpu::EltwiseMulOp>())) {
      auto eltOp = cast<tpu::EltwiseMulOp>(formerOp);
      assert(!eltOp.do_relu() && "done relu");
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::FullyConnectedOp>())) {
      auto fcOp = cast<tpu::FullyConnectedOp>(formerOp);
      fcOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      fcOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::BroadcastMulOp>())) {
      auto bcastOp = cast<tpu::BroadcastMulOp>(formerOp);
      assert(!bcastOp.do_relu() && "done relu");
      bcastOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      bcastOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      auto concatOp = cast<tpu::ConcatOp>(formerOp);
      assert(!concatOp.do_relu() && "done relu");
      concatOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      concatOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::ScaleOp>())) {
      //assert(false);
      // TODO: convert to conv
      return matchFailure();
    } else {
      llvm_unreachable(("unhandled relu fuse with " +
                         getOpName(formerOp).str()).c_str());
      return matchFailure();
    }
    // remove the relu Op
    rewriter.replaceOp(op, {op->getOperand(0)});
    return matchSuccess();
  }
};

struct TpuMoveReluAheadConcatPattern : public RewritePattern {
  TpuMoveReluAheadConcatPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 3, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    // match relu Op that is following concat Ops
    auto formerOp = op->getOperand(0)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      return matchFailure();
    }

    if (!formerOp->getResult(0)->hasOneUse())
      return matchFailure();

    auto concatOp = cast<tpu::ConcatOp>(formerOp);
    LLVM_DEBUG(llvm::errs() << reluOp.getOperationName() << ":"
                            << getOpName(reluOp) << ", after concat" << "\n";);
    size_t nInputs = concatOp.getNumInputs();
    for (uint32_t i = 0; i < nInputs; i ++) {
      auto inOp = formerOp->getOperand(i)->getDefiningOp();
      if (false == supportRelu(inOp)) {
        return matchFailure();
      }
    }

    rewriter.setInsertionPoint(formerOp);
    for (unsigned i = 0; i < nInputs; i++) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(getOpName(op).str() + "_" +
                                             std::to_string(i))));
      attrs.push_back(rewriter.getNamedAttr("quant",
                                            getDefaultQuantParam(rewriter)));
      auto op = rewriter.create<tpu::ReluOp>(
          formerOp->getLoc(), formerOp->getOperand(i)->getType(),
          ArrayRef<Value *>{ formerOp->getOperand(i) },
                             ArrayRef<NamedAttribute>{attrs});
      formerOp->setOperand(i, op.getResult());
    }

    // change the concat Op's name to avoid data comparing with
    // caffe for this op
    concatOp.setAttr("name", rewriter.getStringAttr(concatOp.name().str() +
                                                    "_relu"));
    // remove the relu op after concat
    rewriter.replaceOp(op, {concatOp});
    return matchSuccess();
  }
};

struct TpuDelRedundantReluPattern : public RewritePattern {
  TpuDelRedundantReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 10, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto formerOp = op->getOperand(0)->getDefiningOp();
    auto firstUseOp = op;
    if (formerOp->getResult(0)->hasOneUse())
      return matchFailure();

    for (auto &use : formerOp->getResult(0)->getUses()) {
      auto useOp = use.getOwner();
      if (!llvm::isa<tpu::ReluOp>(useOp)) {
        continue;
      }
      if (useOp->isBeforeInBlock(firstUseOp))
        firstUseOp = useOp;
    }

    for (auto &use : formerOp->getResult(0)->getUses()) {
      auto useOp = use.getOwner();
      if (useOp == firstUseOp)
        continue;

      if (!llvm::isa<tpu::ReluOp>(useOp))
        continue;

      auto reluOp = cast<tpu::ReluOp>(firstUseOp);
      LLVM_DEBUG(llvm::errs() << "deleted ReluOp "
                              << reluOp.getOperationName() << ":"
                              << getOpName(useOp) <<  "\n";);
      // delete the redundant relu op
      rewriter.replaceOp(useOp, {reluOp});
    }
    return matchSuccess();
  }
};

class FuseReluPass : public FunctionPass<FuseReluPass> {
public:
  explicit FuseReluPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<TpuDelRedundantReluPattern>(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<TpuMoveReluAheadConcatPattern>(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<TpuFuseReluPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::ReluOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<
      TpuDelRedundantReluPattern,
      TpuMoveReluAheadConcatPattern,
      TpuFuseReluPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFuseReluPass() {
  return std::make_unique<FuseReluPass>();
}

static PassRegistration<FuseReluPass>
    pass("fuse-relu",
         "Fuse relu op into previous op (conv/eltwise etc)");
