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

#define DEBUG_TYPE "convert_permute"

using namespace mlir;

namespace {

struct TpuPermuteToPixelShufflePattern : public RewritePattern {
  TpuPermuteToPixelShufflePattern(MLIRContext *context)
      : RewritePattern("tpu.permute", 2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto permuteOp = cast<tpu::PermuteOp>(op);
    LLVM_DEBUG(llvm::errs() << permuteOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);

    auto input_shape = getTensorShape(permuteOp.input());
    if (input_shape.size() != 6) {
      return failure();
    }
    std::vector<int32_t> ps = {0, 1, 4, 2, 5, 3};
    std::vector<int32_t> order;
    arrayAttrToVector(permuteOp.order(), order);
    if (order != ps) {
      return failure();
    }
    auto reshape_before =
        dyn_cast_or_null<tpu::ReshapeOp>(permuteOp.input().getDefiningOp());
    if (!reshape_before) {
      return failure();
    }
    auto nextOp = getNextOp(permuteOp);
    if (!nextOp) {
      return failure();
    }
    auto reshape_after = dyn_cast_or_null<tpu::ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = getTensorShape(reshape_after.output());
    int64_t upscale_factor = input_shape[2];
    int64_t on = input_shape[0];
    int64_t oc = input_shape[1];
    int64_t oh = upscale_factor * input_shape[4];
    int64_t ow = upscale_factor * input_shape[5];
    std::vector<int64_t> o_s = {on, oc, oh, ow};
    if (output_shape != o_s) {
      return failure();
    }

    std::vector<Value> operands;
    operands.push_back(reshape_before.input());
    std::vector<NamedAttribute> attrs;
    std::string op_name = reshape_after.name().str();
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("CDR")));
    attrs.push_back(rewriter.getNamedAttr(
        "upscale_factor", rewriter.getI32IntegerAttr(upscale_factor)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    rewriter.replaceOpWithNewOp<tpu::PixelShuffleOp>(
        reshape_after, reshape_after.getResult().getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    permuteOp.erase();
    reshape_before.erase();
    return success();
  }
};

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
        permuteOp, permuteOp.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return success();
  }
};

} // namespace

void tpu::PermuteOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuPermuteToPixelShufflePattern, TpuPermuteToReshapePattern>(
      context);
}
