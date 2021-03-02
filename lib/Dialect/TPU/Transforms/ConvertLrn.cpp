//===- ConvertLrn.cpp - convert
// Lrn----------------------------------===//
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
// add LrnOne/LrnTwo/LrnThree for calc threshold
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_lrn"

using namespace mlir;

namespace {

struct TpuLrnPattern : public RewritePattern {
  TpuLrnPattern(MLIRContext *context) : RewritePattern("tpu.lrn", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lrnOp = cast<tpu::LrnOp>(op);
    LLVM_DEBUG(llvm::errs() << lrnOp.getOperationName() << ":" << getOpName(op)
                            << "\n";);
    auto sqr_lut = lrnOp.getOperand(1).getDefiningOp();
    auto power_lut = lrnOp.getOperand(2).getDefiningOp();
    auto scale = lrnOp.getOperand(3).getDefiningOp();
    if (isa<tpu::NoneOp>(sqr_lut) == false ||
        isa<tpu::NoneOp>(power_lut) == false ||
        isa<tpu::NoneOp>(scale) == false) {
      return failure();
    }
    // lrn one
    std::vector<NamedAttribute> attrs;
    std::string op_name = lrnOp.name().str() + "_one";
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    attrs.push_back(rewriter.getNamedAttr("alpha", lrnOp.alphaAttr()));
    attrs.push_back(rewriter.getNamedAttr("beta", lrnOp.betaAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("local_size", lrnOp.local_sizeAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    std::vector<Value> operands;
    operands.push_back(lrnOp.getOperand(0));
    auto lrn_one = rewriter.create<tpu::LrnOneOp>(
        op->getLoc(), lrnOp.getResult().getType(), operands, attrs);
    // lrn two
    op_name = lrnOp.name().str() + "_two";
    attrs[0] = rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name));
    operands.clear();
    operands.push_back(lrn_one);
    auto lrn_two = rewriter.create<tpu::LrnTwoOp>(
        op->getLoc(), lrnOp.getResult().getType(), operands, attrs);
    // lrn three
    op_name = lrnOp.name().str() + "_three";
    attrs[0] = rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name));
    operands.clear();
    operands.push_back(lrn_two);
    auto lrn_three = rewriter.create<tpu::LrnThreeOp>(
        op->getLoc(), lrnOp.getResult().getType(), operands, attrs);
    // lrn main
    lrnOp.setOperand(3, lrn_three);
    return success();
  }
};

} // namespace

void tpu::LrnOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<TpuLrnPattern>(context);
}
