//===- FuseReshape.cpp - Fuse reshape -------------------------------------===//
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
// This file fuse the input and tht output of the reshape op.
//
//===----------------------------------------------------------------------===//


#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_reshape"

using namespace mlir;

namespace {

// fold reshape ops:  reshape -> reshape -> reshape => reshape
struct FoldReshapePattern : public RewritePattern {
  FoldReshapePattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::ReshapeOp>(op);

    auto formerOp = castOp.getOperand().getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ReshapeOp>())) {
      return failure();
    }
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    castOp.setOperand(formerOp->getOperand(0));
    rewriter.replaceOp(formerOp, {castOp.getResult()});
    return success();
  }
};

// remove input---- reshape --output1
//              |-- reshape -- output2
struct FoldSiblingReshapePattern : public RewritePattern {
  FoldSiblingReshapePattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::ReshapeOp>(op);
    auto parentOp = castOp.getOperand().getDefiningOp();
    std::vector<Operation*> targets;
    int cnt = 0;
    for (auto &use : parentOp->getResult(0).getUses()) {
      auto child = use.getOwner();
      if (child == op) {
        continue;
      }
      if (!isa<tpu::ReshapeOp>(child)) {
        continue;
      }
      if (child->getResult(0).getType() == castOp.getResult().getType()) {
        targets.push_back(child);
      }
      cnt++;
    }
    if (cnt == 0) {
      return failure();
    }

    for (auto t : targets) {
      op->moveBefore(t);
      rewriter.replaceOp(t, {castOp.getResult()});
    }
    return success();
  }
};

struct TpuRemoveIdentityReshapePattern : public RewritePattern {
  TpuRemoveIdentityReshapePattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(formerOp->getResult(0));

    if (inputShapes.size() != outputShapes.size()) {
      return failure();
    }
    for (unsigned i = 0; i < inputShapes.size(); ++i) {
      if (inputShapes[i] != outputShapes[i]) {
        return failure();
      }
    }
    rewriter.replaceOp(op, {formerOp->getResult(0)});
    return success();
  }
};

class FuseReshapePass : public mlir::PassWrapper<FuseReshapePass, FunctionPass> {
public:
  explicit FuseReshapePass() {}

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<TpuRemoveIdentityReshapePattern>(&getContext());
    patterns.insert<FoldSiblingReshapePattern>(&getContext());
    patterns.insert<FoldReshapePattern>(&getContext());
    applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

};

} // anonymous namespace

void tpu::ReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
    TpuRemoveIdentityReshapePattern,
    FoldSiblingReshapePattern,
    FoldReshapePattern
  >(context);
}

std::unique_ptr<mlir::Pass>  mlir::createFuseReshapePass() {
  return std::make_unique<FuseReshapePass>();
}
