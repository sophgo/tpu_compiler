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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "upsample_to_deconv"

using namespace mlir;

namespace {

template <typename OpTy>
struct TransposeRightMatrixPattern : public RewritePattern {
  TransposeRightMatrixPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

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
    int k = leftShape[1];
    bool need_transpose = false;
    int n = rightShape[1];
    if (k != rightShape[0]) {
      need_transpose = true;
      n = rightShape[0];
    }
    if (!need_transpose) {
      return failure();
    }

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    auto elementType = rightOp->getResult(0).getType().cast<RankedTensorType>().getElementType();

    auto prevReshapeOp = rightOp->getOperand(0).getDefiningOp();
    if (isa<tpu::ReshapeOp>(prevReshapeOp)) {
      return failure();
    }
    auto type = RankedTensorType::get({n, k, 1, 1}, elementType);
    prevReshapeOp->getResult(0).setType(type);

    type = RankedTensorType::get({k, n, 1, 1}, elementType);
    std::string name = getOpName(prevReshapeOp).str() + "_transposed";

    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr("order0", rewriter.getI32IntegerAttr(1)));
    attrs.push_back(rewriter.getNamedAttr("order1", rewriter.getI32IntegerAttr(0)));
    attrs.push_back(rewriter.getNamedAttr("order2", rewriter.getI32IntegerAttr(2)));
    attrs.push_back(rewriter.getNamedAttr("order3", rewriter.getI32IntegerAttr(3)));
    attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    operands.push_back(prevReshapeOp->getResult(0));
    auto permuteOp = rewriter.create<tpu::PermuteOp>(
        op->getLoc(), type, ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});

    attrs.clear();
    operands.clear();
    name += "_reshape";
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
    operands.push_back(permuteOp);
    type = RankedTensorType::get({k, n}, elementType);
    auto reshapeOp = rewriter.create<tpu::ReshapeOp>(
        op->getLoc(), type, ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});

    op->setOperand(1, reshapeOp.getResult());
    return success();
  }
};

} // namespace

void tpu::MatMulOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TransposeRightMatrixPattern<tpu::MatMulOp>>(context);
}
