//===- ConvertPermute.cpp - convert
//Permute----------------------------------===//
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
// This file implements the conversion of Conv2D.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_permute"

using namespace mlir;

namespace {

// For example if [1,1,256,1] => [1,1,1,256], it can convert to reshape
struct TpuPermuteToReshapePattern : public RewritePattern {
  TpuPermuteToReshapePattern(MLIRContext *context)
      : RewritePattern("tpu.permute", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto permuteOp = cast<tpu::PermuteOp>(op);
    LLVM_DEBUG(llvm::errs() << permuteOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);

    std::vector<int64_t> shape;
    int64_t input_size;
    getTensorShapeAndSize(permuteOp.input(), shape, input_size);
    bool need_opt = false;
    for (auto dim : shape) {
      if (dim == input_size) {
        need_opt = true;
        break;
      }
    }

    if (need_opt == false) {
      return matchFailure();
    }

    std::vector<Value *> operands;
    operands.push_back(op->getOperand(0));
    std::vector<NamedAttribute> attrs;
    std::string op_name = permuteOp.name().str();
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    if (permuteOp.layer_id().hasValue()) {
      attrs.push_back(rewriter.getNamedAttr("layer_id",
                                            permuteOp.layer_idAttr()));
    }
    rewriter.replaceOpWithNewOp<tpu::ReshapeOp>(
        permuteOp, permuteOp.getResult()->getType(),
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    return matchSuccess();
  }
};

} // namespace

void tpu::PermuteOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuPermuteToReshapePattern>(context);
}
