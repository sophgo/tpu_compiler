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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_permute"

using namespace mlir;

namespace {

// Permute can convert to Reshape in some situations.
// For example:
// [4,3,28,1] => [4,3,1,28]
// [4,3,1,28] => [4,1,3,28]
struct TpuPermuteToReshapePattern : public RewritePattern {
  TpuPermuteToReshapePattern(MLIRContext *context)
      : RewritePattern("tpu.permute", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto permuteOp = cast<tpu::PermuteOp>(op);
    LLVM_DEBUG(llvm::errs() << permuteOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);

    std::vector<int64_t> shape;
    int64_t input_size;
    getTensorShapeAndSize(permuteOp.input(), shape, input_size);

    int dim_size = shape.size();
    int start = 0, end = dim_size - 1;
    std::vector<int32_t> order;
    arrayAttrToVector(permuteOp.order(), order);
    while (start < dim_size && start == order[start]) {
      start++;
    }
    while (end > start && end == order[end]) {
      end--;
    }
    bool do_reshape = true;
    int64_t sum = 1;
    for (int index = start; index <= end; index++) {
      sum *= shape[index];
      if (shape[index] != 1 && sum != shape[index]) {
        do_reshape = false;
        break;
      }
    }

    if (do_reshape == false) {
      return failure();
    }

    std::vector<Value> operands;
    operands.push_back(op->getOperand(0));
    std::vector<NamedAttribute> attrs;
    std::string op_name = permuteOp.name().str();
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));

    rewriter.replaceOpWithNewOp<tpu::ReshapeOp>(
        permuteOp, permuteOp.getResult().getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    return success();
  }
};

} // namespace

void tpu::PermuteOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuPermuteToReshapePattern>(context);
}
