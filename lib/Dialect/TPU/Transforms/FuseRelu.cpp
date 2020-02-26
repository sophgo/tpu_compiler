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
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TpuFuseReluPattern : public RewritePattern {
  TpuFuseReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    llvm::errs() << reluOp.getOperationName() << ":" << getOpName(reluOp)<< "\n";

    // match relu Op that is following conv or eltwise Ops
    auto formerOp = op->getOperand(0)->getDefiningOp();

    if (matchPattern(formerOp, m_Op<tpu::Conv2DOp>())) {
      auto convOp = cast<tpu::Conv2DOp>(formerOp);
      assert(convOp.param().do_relu().getValue() == false);
      // set do_relu
      convOp.setAttr("param",
          tpu::ConvParam::get(
              convOp.param().stride_h(),
              convOp.param().stride_w(),
              convOp.param().padding(),
              convOp.param().dilation_h(),
              convOp.param().dilation_w(),
              convOp.param().group(),
              convOp.param().is_dw(),
              convOp.param().with_bias(),
              rewriter.getBoolAttr(true),
              rewriter.getContext()));
      convOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseAddOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      assert(!eltOp.do_relu());
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMaxOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      assert(!eltOp.do_relu());
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMulOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      assert(!eltOp.do_relu());
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::FullyConnectedOp>())) {
      auto fcOp = cast<tpu::FullyConnectedOp>(formerOp);
      fcOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      fcOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    } else if (matchPattern(formerOp, m_Op<tpu::BroadcastMulOp>())) {
      auto bcastOp = cast<tpu::BroadcastMulOp>(formerOp);
      assert(!bcastOp.do_relu());
      bcastOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      bcastOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    }  else if (matchPattern(formerOp, m_Op<tpu::PoolAvg2DOp>())) {
      auto poolOp = cast<tpu::PoolAvg2DOp>(formerOp);
      assert(poolOp.param().do_relu().getValue() == false);
      poolOp.setAttr("param",
          tpu::PoolParam::get(
              poolOp.param().kernel_h(),
              poolOp.param().kernel_w(),
              poolOp.param().padding_t(),
              poolOp.param().padding_b(),
              poolOp.param().padding_l(),
              poolOp.param().padding_r(),
              poolOp.param().stride_h(),
              poolOp.param().stride_w(),
              rewriter.getBoolAttr(true),
              rewriter.getContext()));
      poolOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    }  else if (matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      // TODO: need to fuse
      //assert(false);
      return matchFailure();
    } else if (matchPattern(formerOp, m_Op<tpu::ScaleOp>())) {
      //assert(false);
      // TODO: convert to conv
      return matchFailure();
    } else {
      llvm::errs() << "unhandled relu fuse with " << getOpName(formerOp) << "\n";
      assert(0);
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
    llvm::errs() << reluOp.getOperationName() << "\n";

    // match relu Op that is following concat Ops
    auto formerOp = op->getOperand(0)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      return matchFailure();
    }
    auto concatOp = cast<tpu::ConcatOp>(formerOp);

    size_t nInputs = concatOp.getNumInputs();
    for (unsigned i = 0; i < nInputs; i++) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name",
          rewriter.getStringAttr(getOpName(op).str() + "_" + std::to_string(i))));
      attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      auto op = rewriter.create<tpu::ReluOp>(
          formerOp->getLoc(), formerOp->getOperand(i)->getType(),
          ArrayRef<Value *>{formerOp->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
      formerOp->setOperand(i, op.getResult());
    }

    // change the concat Op's name to avoid data comparing with caffe for this op
    concatOp.setAttr("name", rewriter.getStringAttr(concatOp.name().str() + "_relu"));
    // remove the relu op after concat
    rewriter.replaceOp(op, {concatOp});
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
    patterns.insert<TpuMoveReluAheadConcatPattern>(context);
    applyPatternsGreedily(fn, patterns);

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
      TpuMoveReluAheadConcatPattern,
      TpuFuseReluPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFuseReluPass() {
  return std::make_unique<FuseReluPass>();
}

static PassRegistration<FuseReluPass>
    pass("fuse-relu",
         "Fuse relu op into previous op (conv/eltwise etc)");
