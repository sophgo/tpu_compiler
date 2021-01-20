//===- ConvertPermute.cpp - convert
// Permute----------------------------------===//
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
// This file implements the permute to reshape
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_permute"

using namespace mlir;

namespace {

// Eliminate Csc op if pixel_format is RGB and unaligned.
template <typename TyOp>
struct TpuEliminateCscOpPattern : public RewritePattern {
  TpuEliminateCscOpPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<TyOp>(op);
    LLVM_DEBUG(llvm::errs() << castOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);
    auto pixel_format = castOp.pixel_format();
    auto aligned = castOp.aligned();
    if (pixel_format == "YUV420_PLANAR" || aligned) {
      return failure();
    }

    // delete csc op
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

void tpu::CscOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuEliminateCscOpPattern<tpu::CscOp>>(context);
}
