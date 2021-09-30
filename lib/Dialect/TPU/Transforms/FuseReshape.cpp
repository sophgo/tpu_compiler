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

// reshape: 1x32x32x32 -> 1x1x32x32x32
//   reduce_max: 1x1x32x32x32 -> 1x32x32x32
struct TpuReshapeReduceMaxPattern : public RewritePattern {
  TpuReshapeReduceMaxPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return failure();

    if (std::distance(op->getResult(0).use_begin(),
                      op->getResult(0).use_end()) != 1) {
      return failure();
    }

    Operation *nextOp = nullptr;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0).getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMaxOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue()) {
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        }
        break;
      } else {
        return failure();
      }
    }

    if (axes_array.size() != 1)
      return failure();

    if (outputShapes[axes_array[0]] != 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
      << "ReshapeReduceMaxPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp) << "\n");

    nextOp->getResult(0).replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);

    return success();
  }
};

struct TpuReshapeReduceMinPattern : public RewritePattern {
  TpuReshapeReduceMinPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return failure();

    if (std::distance(op->getResult(0).use_begin(),
                      op->getResult(0).use_end()) != 1) {
      return failure();
    }

    Operation *nextOp = nullptr;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0).getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMinOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue()) {
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        }
        break;
      } else {
        return failure();
      }
    }

    if (axes_array.size() != 1)
      return failure();

    if (outputShapes[axes_array[0]] != 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
      << "ReshapeReduceMinPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp) << "\n");

    nextOp->getResult(0).replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);

    return success();
  }
};

// reshape: 1x32x32x32 -> 1x1x32x32x32
//   reshape: 1x1x32x32x32 -> 1x32x32x32
//   reduce_max: 1x1x32x32x32 -> 1x32x32x32
struct TpuDReshapeReduceMaxPattern : public RewritePattern {
  TpuDReshapeReduceMaxPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return failure();

    if (std::distance(op->getResult(0).use_begin(),
                      op->getResult(0).use_end()) != 2) {
      return failure();
    }

    Operation *nextOp = nullptr, *nextOp2 = nullptr;
    std::vector<int64_t> nextShapes;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0).getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMaxOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue())
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        break;
      } else if (auto tpuOp = llvm::dyn_cast<tpu::ReshapeOp>(use.getOwner())) {
        nextOp2 = tpuOp.getOperation();
        nextShapes = getTensorShape(nextOp2->getResult(0));
      } else
        return failure();
    }

    if (axes_array.size() != 1)
      return failure();

    if (inputShapes.size() != nextShapes.size())
      return failure();

    for (unsigned i = 0; i < inputShapes.size(); ++i)
      if (inputShapes[i] != nextShapes[i])
        return failure();

    if (outputShapes[axes_array[0]] != 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
      << "DReshapeReduceMaxPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp)
      << ", " << getOpName(nextOp2) << "\n");

    nextOp2->getResult(0).replaceAllUsesWith(nextOp2->getOperand(0));
    nextOp->getResult(0).replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);
    rewriter.eraseOp(nextOp2);

    return success();
  }
};

// reshape: 1x32x32x32 -> 1x1x32x32x32
//   reshape: 1x1x32x32x32 -> 1x32x32x32
//   reduce_max: 1x1x32x32x32 -> 1x32x32x32
struct TpuDReshapeReduceMinPattern : public RewritePattern {
  TpuDReshapeReduceMinPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return failure();

    if (std::distance(op->getResult(0).use_begin(),
                      op->getResult(0).use_end()) != 2) {
      return failure();
    }

    Operation *nextOp = nullptr, *nextOp2 = nullptr;
    std::vector<int64_t> nextShapes;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0).getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMinOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue())
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        break;
      } else if (auto tpuOp = llvm::dyn_cast<tpu::ReshapeOp>(use.getOwner())) {
        nextOp2 = tpuOp.getOperation();
        nextShapes = getTensorShape(nextOp2->getResult(0));
      } else
        return failure();
    }

    if (axes_array.size() != 1)
      return failure();

    if (inputShapes.size() != nextShapes.size())
      return failure();

    for (unsigned i = 0; i < inputShapes.size(); ++i)
      if (inputShapes[i] != nextShapes[i])
        return failure();

    if (outputShapes[axes_array[0]] != 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
      << "DReshapeReduceMinxPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp)
      << ", " << getOpName(nextOp2) << "\n");

    nextOp2->getResult(0).replaceAllUsesWith(nextOp2->getOperand(0));
    nextOp->getResult(0).replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);
    rewriter.eraseOp(nextOp2);

    return success();
  }
};

class FuseReshapePass : public mlir::PassWrapper<FuseReshapePass, FunctionPass> {
public:
  explicit FuseReshapePass() {}

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<TpuRemoveIdentityReshapePattern>(&getContext());
    patterns.insert<TpuReshapeReduceMaxPattern>(&getContext());
    patterns.insert<TpuDReshapeReduceMaxPattern>(&getContext());
    patterns.insert<TpuReshapeReduceMinPattern>(&getContext());
    patterns.insert<TpuDReshapeReduceMinPattern>(&getContext());
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
    TpuReshapeReduceMaxPattern,
    TpuDReshapeReduceMaxPattern,
    TpuReshapeReduceMinPattern,
    TpuDReshapeReduceMinPattern,
    FoldSiblingReshapePattern,
    FoldReshapePattern
  >(context);
}

std::unique_ptr<mlir::Pass>  mlir::createFuseReshapePass() {
  return std::make_unique<FuseReshapePass>();
}
