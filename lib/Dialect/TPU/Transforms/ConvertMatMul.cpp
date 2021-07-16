//===- ConvertUpsample.cpp - convert unsample to deconv -----------===//
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
// This file convert upsample op to deconv.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "matmul"

using namespace mlir;

namespace mlir {
struct MergeTransposeMatMulPattern : public RewritePattern {
  MergeTransposeMatMulPattern(MLIRContext *context)
      : RewritePattern("tpu.matmul", 2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::MatMulOp>(op);
    auto leftOp = op->getOperand(0).getDefiningOp();
    auto rightOp = op->getOperand(1).getDefiningOp();
    auto outputOp = getNextOp(op);

    auto leftType = leftOp->getResult(0).getType().template cast<TensorType>();
    std::vector<int64_t> leftShape(leftType.getShape());
    auto rightType = rightOp->getResult(0).getType().cast<TensorType>();
    std::vector<int64_t> rightShape(rightType.getShape());
    if (rightShape.size() != 4 || leftShape.size() != 4) {
      return failure();
    }
    if (castOp.left_transpose() != false || castOp.right_transpose() != false ||
        castOp.output_transpose() != false) {
      return failure();
    }
    bool match = false;
    int pattern[] = {0, 2, 1, 3};
    std::vector<int> p(pattern, pattern + 4);
    if (isa<tpu::PermuteOp>(leftOp) && leftOp->getResult(0).hasOneUse()) {
      auto pmOp = cast<tpu::PermuteOp>(leftOp);
      std::vector<int> order;
      arrayAttrToVector(pmOp.order(), order);
      if (order == p) {
        castOp->setAttr("left_transpose", rewriter.getBoolAttr(true));
        rewriter.replaceOp(leftOp, {leftOp->getOperand(0)});
        match = true;
      }
    }
    if (isa<tpu::PermuteOp>(rightOp) && rightOp->getResult(0).hasOneUse()) {
      auto pmOp = cast<tpu::PermuteOp>(rightOp);
      std::vector<int> order;
      arrayAttrToVector(pmOp.order(), order);
      if (order == p) {
        castOp->setAttr("right_transpose", rewriter.getBoolAttr(true));
        rewriter.replaceOp(rightOp, {rightOp->getOperand(0)});
        match = true;
      }
    }
    if (outputOp != nullptr && isa<tpu::PermuteOp>(outputOp)) {
      auto pmOp = cast<tpu::PermuteOp>(outputOp);
      std::vector<int> order;
      arrayAttrToVector(pmOp.order(), order);
      if (order == p) {
        castOp->setAttr("output_transpose", rewriter.getBoolAttr(true));
        castOp->setAttr("name", pmOp.nameAttr());
        rewriter.replaceOp(outputOp, {outputOp->getOperand(0)});
        match = true;
      }
    }
    return match ? success() : failure();
  }
};

struct TransposeRightMatrixPattern : public RewritePattern {
  TransposeRightMatrixPattern(MLIRContext *context)
      : RewritePattern("tpu.matmul", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto leftOp = op->getOperand(0).getDefiningOp();
    auto rightOp = op->getOperand(1).getDefiningOp();

    auto leftType = leftOp->getResult(0).getType().template cast<TensorType>();
    std::vector<int64_t> leftShape(leftType.getShape());
    auto rightType = rightOp->getResult(0).getType().cast<TensorType>();
    std::vector<int64_t> rightShape(rightType.getShape());
    if (rightShape.size() != 2 || leftShape.size() != 2) {
      return failure();
    }
    if (rightShape[0] == leftShape[1]) {
      return failure();
    }
    if (rightShape[1] != leftShape[1]) {
      llvm_unreachable("matmul shape uncorrect");
      return failure();
    }

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    auto prevOp = rightOp->getOperand(0).getDefiningOp();

    std::string name = getOpName(prevOp).str() + "_transposed";
    std::vector<int> order;
    order = {1, 0};
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI32ArrayAttr(order)));
    operands.push_back(prevOp->getResult(0));
    std::vector<int64_t> oshape;
    oshape = {rightShape[1], rightShape[0]};
    auto result_type =
        RankedTensorType::get(oshape, rightType.getElementType());
    auto permuteOp = rewriter.create<tpu::PermuteOp>(
        op->getLoc(), result_type, ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    op->setOperand(1, permuteOp.getResult());
    return success();
  }
};

} // namespace mlir

void tpu::MatMulOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MergeTransposeMatMulPattern, TransposeRightMatrixPattern>(
      context);
}
