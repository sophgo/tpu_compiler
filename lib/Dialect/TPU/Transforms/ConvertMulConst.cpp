//===- ConvertScale.cpp - convert scale -----------------------------------===//
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
// This file implements the conversion of scale.
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

#include <sstream>

#define DEBUG_TYPE "convert_mul_const"

using namespace mlir;

namespace {

struct TpuRemoveMulConstPattern : public RewritePattern {
  TpuRemoveMulConstPattern(MLIRContext *context)
      : RewritePattern("tpu.mul_const", 6, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::MulConstOp>(op);
    float const_val = castOp.const_val().convertToFloat();
    if (const_val != 1.0) {
      return failure();
    }
    rewriter.replaceOp(op, {castOp.input()});
    return success();
  }
};

// merge into conv or fc
struct TpuMergeMulConstPattern : public RewritePattern {
  TpuMergeMulConstPattern(MLIRContext *context)
      : RewritePattern("tpu.mul_const", 4, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::MulConstOp>(op);
    float const_val = castOp.const_val().convertToFloat();
    bool do_relu = castOp.do_relu();
    std::string op_name = castOp.name().str();
    TensorFile *wTF = getWeightTensorFile(op);

    auto formerOp = castOp.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    if (auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      if (convOp.do_relu() == true && const_val < 0) {
        return failure();
      }
    } else if (auto fcOp = dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
      if (fcOp.do_relu() == true && const_val < 0) {
        return failure();
      }
    } else {
      return failure();
    }
    for (auto value : formerOp->getOperands()) {
      auto weight_op =
          llvm::dyn_cast_or_null<tpu::LoadWeightOp>(value.getDefiningOp());
      if (!weight_op) {
        continue;
      }
      auto shape = getTensorShape(value);
      auto storage = weight_op.storage();
      auto weight = readAndDeleteWeightTensor<float>(value, wTF);
      for (auto &w : *weight) {
        w *= const_val;
      }
      addWeightTensorAndUpdateWeightOp(value, "_mulconst", *weight, shape,
                                       storage, wTF);
    }
    formerOp->setAttr("name", rewriter.getStringAttr(op_name));
    if (do_relu) {
      formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    }
    rewriter.replaceOp(op, {castOp.input()});
    return success();
  }
};

} // namespace

// Canonicalizer
void tpu::MulConstOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuRemoveMulConstPattern, TpuMergeMulConstPattern>(context);
}
