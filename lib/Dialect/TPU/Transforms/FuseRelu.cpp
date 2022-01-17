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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_relu"

using namespace mlir;

namespace {

static bool supportRelu(Operation *op) {
  if (matchPattern(op, m_Op<tpu::Conv2DOp>()) ||
      matchPattern(op, m_Op<tpu::Conv3DOp>()) ||
      matchPattern(op, m_Op<tpu::DeConv2DOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseAddOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMaxOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMinOp>()) ||
      matchPattern(op, m_Op<tpu::EltwiseMulOp>()) ||
      matchPattern(op, m_Op<tpu::MatMulOp>()) ||
      matchPattern(op, m_Op<tpu::FullyConnectedOp>()) ||
      matchPattern(op, m_Op<tpu::BroadcastMulOp>()) ||
      matchPattern(op, m_Op<tpu::BroadcastAddOp>()) ||
      matchPattern(op, m_Op<tpu::MulConstOp>()) ||
      matchPattern(op, m_Op<tpu::ScaleOp>()) ||
      matchPattern(op, m_Op<tpu::ConcatOp>())) {
    return true;
  }
  return false;
}

struct TpuFuseReluPattern : public RewritePattern {
  TpuFuseReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    LLVM_DEBUG(llvm::errs() << reluOp.getOperationName() << ":"
                            << getOpName(reluOp)<< "\n";);

    // match relu Op that is following conv or eltwise Ops
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    if (matchPattern(formerOp, m_Op<tpu::ScaleOp>()) || !supportRelu(formerOp)) {
      return failure();
    }

    formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    formerOp->setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
    // remove the relu Op
    rewriter.replaceOp(op, {op->getOperand(0)});
    return success();
  }
};

struct TpuMoveReluAheadConcatPattern : public RewritePattern {
  TpuMoveReluAheadConcatPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 3, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    // match relu Op that is following concat Ops
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      return failure();
    }

    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    auto concatOp = cast<tpu::ConcatOp>(formerOp);
    LLVM_DEBUG(llvm::errs() << reluOp.getOperationName() << ":"
                            << getOpName(reluOp) << ", after concat" << "\n";);
    size_t nInputs = concatOp.getNumInputs();
    for (uint32_t i = 0; i < nInputs; i ++) {
      auto inOp = formerOp->getOperand(i).getDefiningOp();
      if (false == supportRelu(inOp)) {
        return failure();
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
          formerOp->getLoc(), formerOp->getOperand(i).getType(),
          ArrayRef<Value>{ formerOp->getOperand(i) },
                             ArrayRef<NamedAttribute>{attrs});
      formerOp->setOperand(i, op.getResult());
    }

    // change the concat Op's name to avoid data comparing with
    // caffe for this op
    concatOp->setAttr("name", rewriter.getStringAttr(concatOp.name().str() +
                                                    "_relu"));
    // remove the relu op after concat
    rewriter.replaceOp(op, {concatOp});
    return success();
  }
};

struct TpuDelRedundantReluPattern : public RewritePattern {
  TpuDelRedundantReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto formerOp = op->getOperand(0).getDefiningOp();
    auto firstUseOp = op;
    if (formerOp->getResult(0).hasOneUse())
      return failure();

    for (auto &use : formerOp->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (!llvm::isa<tpu::ReluOp>(useOp)) {
        continue;
      }
      if (useOp->isBeforeInBlock(firstUseOp))
        firstUseOp = useOp;
    }

    for (auto &use : formerOp->getResult(0).getUses()) {
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
    return success();
  }
};

class FuseReluPass : public mlir::PassWrapper<FuseReluPass, FunctionPass> {
public:
  explicit FuseReluPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<TpuDelRedundantReluPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuMoveReluAheadConcatPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<TpuFuseReluPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
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

std::unique_ptr<mlir::Pass> mlir::createFuseReluPass() {
  return std::make_unique<FuseReluPass>();
}
