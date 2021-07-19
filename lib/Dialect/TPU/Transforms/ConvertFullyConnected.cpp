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

#define DEBUG_TYPE "fully_connected"

using namespace mlir;

namespace mlir {
struct MergeTransposeFCPattern : public RewritePattern {
  MergeTransposeFCPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::FullyConnectedOp>(op);
    if (castOp.input_transpose() != false ||
        castOp.output_transpose() != false) {
      return failure();
    }
    auto leftOp = op->getOperand(0).getDefiningOp();
    auto outputOp = getNextOp(op);
    int batch_high, batch_low, m, k, n;
    parseFullyConnectedParam<tpu::FullyConnectedOp>(op, batch_high, batch_low,
                                                    m, k, n);
    if (batch_low == 1) {
      return failure();
    }
    bool match = false;
    int pattern[] = {0, 2, 1, 3};
    std::vector<int> p(pattern, pattern + 4);
    std::vector<int64_t> shape_4;
    std::vector<int> order_4;
    if (isa<tpu::PermuteOp>(leftOp) && leftOp->getResult(0).hasOneUse()) {
      parsePermuteParam<tpu::PermuteOp>(leftOp, shape_4, order_4);
      if (order_4 == p) {
        castOp->setAttr("input_transpose", rewriter.getBoolAttr(true));
        rewriter.replaceOp(leftOp, {leftOp->getOperand(0)});
        match = true;
      }
    }
    if (outputOp != nullptr && isa<tpu::PermuteOp>(outputOp)) {
      parsePermuteParam<tpu::PermuteOp>(outputOp, shape_4, order_4);
      if (order_4 == p) {
        auto pmOp = cast<tpu::PermuteOp>(outputOp);
        castOp->setAttr("output_transpose", rewriter.getBoolAttr(true));
        castOp->setAttr("name", pmOp.nameAttr());
        rewriter.replaceOp(outputOp, {outputOp->getOperand(0)});
        match = true;
      }
    }
    return match ? success() : failure();
  }
};

} // namespace mlir

void tpu::FullyConnectedOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MergeTransposeFCPattern>(context);
}
