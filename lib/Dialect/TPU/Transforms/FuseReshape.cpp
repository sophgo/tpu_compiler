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


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_reshape"

using namespace mlir;

namespace {
struct TpuFuseReshapePattern : public RewritePattern {
  TpuFuseReshapePattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    // 1. Remove single useless reshape
    if (inputShapes.size() == outputShapes.size()) {
      bool useless = true;
      for (unsigned i = 0; i < inputShapes.size(); ++i) {
        if (inputShapes[i] != outputShapes[i]) {
          useless = false;
          break;
        }
      }

      if (useless) {
        LLVM_DEBUG(llvm::dbgs()
            << "ReshapePattern: " << getOpName(op)
            << ", layer ID " << getOpLayerId(op)
            << ", inputs " << inputShapes.size()
            << ", outputs " << outputShapes.size()
            << ", remove " << getOpName(op) << "\n");
        op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
        rewriter.eraseOp(op);
        return matchSuccess();
      }
    }

    // 2. Remove redundant reshape -> reshape
    //    e.g. input - reshape - reshape - quant => input - quant
    auto opdOp = op->getOperand(0)->getDefiningOp();
    if (llvm::dyn_cast<tpu::ReshapeOp>(opdOp)) {
      auto opdInputShape = getTensorShape(opdOp->getOperand(0));

      if (opdInputShape.size() != outputShapes.size())
        return matchFailure();

      for (unsigned i = 0; i < outputShapes.size(); ++i)
        if (opdInputShape[i] != outputShapes[i])
          return matchFailure();

      // Check reshape - reshape has only one use
      if (std::distance(op->getOperand(0)->use_begin(),
                        op->getOperand(0)->use_end()) > 1)
        return matchFailure();

      if (std::distance(op->getResult(0)->use_begin(),
                        op->getResult(0)->use_end()) > 1)
        return matchFailure();

      LLVM_DEBUG(llvm::dbgs()
          << "ReshapePattern: " << getOpName(op)
          << ", layer ID " << getOpLayerId(op)
          << ", inputs " << inputShapes.size()
          << ", outputs " << outputShapes.size()
          << ", remove " << getOpName(op)
          << ", " << getOpName(opdOp) << "\n");

      op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
      opdOp->getResult(0)->replaceAllUsesWith(opdOp->getOperand(0));
      rewriter.eraseOp(op);
      rewriter.eraseOp(opdOp);
      return matchSuccess();
    }

    return matchFailure();
  }
};

// reshape: 1x32x32x32 -> 1x1x32x32x32
//   reduce_max: 1x1x32x32x32 -> 1x32x32x32
struct TpuReshapeReduceMaxPattern : public RewritePattern {
  TpuReshapeReduceMaxPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return matchFailure();

    if (std::distance(op->getResult(0)->use_begin(),
                      op->getResult(0)->use_end()) != 1) {
      return matchFailure();
    }

    Operation *nextOp = nullptr;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0)->getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMaxOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue())
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        break;
      } else
        return matchFailure();
    }

    if (axes_array.size() != 1)
      return matchFailure();

    if (outputShapes[axes_array[0]] != 1)
      return matchFailure();

    LLVM_DEBUG(llvm::dbgs()
      << "ReshapeReduceMaxPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp) << "\n");

    nextOp->getResult(0)->replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);

    return matchSuccess();
  }
};

// reshape: 1x32x32x32 -> 1x1x32x32x32
//   reshape: 1x1x32x32x32 -> 1x32x32x32
//   reduce_max: 1x1x32x32x32 -> 1x32x32x32
struct TpuDReshapeReduceMaxPattern : public RewritePattern {
  TpuDReshapeReduceMaxPattern(MLIRContext *context)
      : RewritePattern(tpu::ReshapeOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto outputShapes = getTensorShape(op->getResult(0));
    auto inputShapes = getTensorShape(op->getOperand(0));

    if (outputShapes.size() != 5)
      return matchFailure();

    if (std::distance(op->getResult(0)->use_begin(),
                      op->getResult(0)->use_end()) != 2) {
      return matchFailure();
    }

    Operation *nextOp = nullptr, *nextOp2 = nullptr;
    std::vector<int64_t> nextShapes;
    std::vector<int32_t> axes_array;
    for (auto &use : op->getResult(0)->getUses()) {
      if (auto tpuOp = llvm::dyn_cast<tpu::ReduceMaxOp>(use.getOwner())) {
        nextOp = tpuOp.getOperation();
        if (tpuOp.axes().hasValue())
          arrayAttrToVector(tpuOp.axes().getValue(), axes_array);
        break;
      } else if (auto tpuOp = llvm::dyn_cast<tpu::ReshapeOp>(use.getOwner())) {
        nextOp2 = tpuOp.getOperation();
        nextShapes = getTensorShape(nextOp2->getResult(0));
      } else
        return matchFailure();
    }

    if (axes_array.size() != 1)
      return matchFailure();

    if (inputShapes.size() != nextShapes.size())
      return matchFailure();

    for (unsigned i = 0; i < inputShapes.size(); ++i)
      if (inputShapes[i] != nextShapes[i])
        return matchFailure();

    if (outputShapes[axes_array[0]] != 1)
      return matchFailure();

    LLVM_DEBUG(llvm::dbgs()
      << "DReshapeReduceMaxPattern: " << getOpName(op)
      << ", layer ID " << getOpLayerId(op)
      << ", inputs " << inputShapes.size()
      << ", outputs " << outputShapes.size()
      << ", remove " << getOpName(op)
      << ", " << getOpName(nextOp)
      << ", " << getOpName(nextOp2) << "\n");

    nextOp2->getResult(0)->replaceAllUsesWith(nextOp2->getOperand(0));
    nextOp->getResult(0)->replaceAllUsesWith(nextOp->getOperand(0));
    op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(nextOp);
    rewriter.eraseOp(nextOp2);

    return matchSuccess();
  }
};

class FuseReshapePass : public FunctionPass<FuseReshapePass> {
public:
  explicit FuseReshapePass() {}

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<TpuFuseReshapePattern>(&getContext());
    patterns.insert<TpuReshapeReduceMaxPattern>(&getContext());
    patterns.insert<TpuDReshapeReduceMaxPattern>(&getContext());
    applyPatternsGreedily(getFunction(), patterns);
  }

};

} // anonymous namespace

void tpu::ReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
    TpuFuseReshapePattern,
    TpuReshapeReduceMaxPattern,
    TpuDReshapeReduceMaxPattern
  >(context);
}

std::unique_ptr<OpPassBase<FuncOp>>  mlir::createFuseReshapePass() {
  return std::make_unique<FuseReshapePass>();
}

static PassRegistration<FuseReshapePass>
    pass("fuse-reshape",
         "Fuse the input and the output of reshape op");
