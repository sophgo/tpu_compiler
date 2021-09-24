//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
// This file implements the TPU dialect OP Stats pass.
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

#define DEBUG_TYPE "convert_scale"

using namespace mlir;

namespace {

struct TpuConvertClipToRelu6Pattern : public RewritePattern {
  TpuConvertClipToRelu6Pattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op);
    if (!clipOp) {
      return failure();
    }

    auto min = clipOp.min().convertToFloat();
    if (min != 0) {
      // keep < 0 part
      return failure();
    }

    auto formerOp = clipOp.getOperand(0).getDefiningOp();
    if (isa<tpu::ReluOp>(formerOp)) {
      // has deal with
      return failure();
    }

    if (isa<tpu::Conv2DOp>(formerOp)){
      auto cast_op = cast<tpu::Conv2DOp>(formerOp);
      if(cast_op.param().do_relu().getValue() == true)
        return failure();
    }

    auto loc = op->getLoc();

    auto layer_name = mlir::getOpName(clipOp).str();
    std::vector<Value> newOperands;
    newOperands.push_back(clipOp.getOperand(0));

    // duplicate name for not mess up calibration table name
    std::string formerOpName = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(formerOpName)));
    attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    // insert relu before clip op
    auto reluOp = rewriter.create<tpu::ReluOp>(loc,
        clipOp.getOperand(0).getType(),
        ArrayRef<Value>{newOperands},
        ArrayRef<NamedAttribute>{attrs});

    // duplicate one
    std::vector<Value> operands;
    auto NoneOp = rewriter.create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());

    operands.push_back(reluOp.getResult());
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    auto newOp = rewriter.create<tpu::ClipOp>(loc,
        op->getResult(0).getType(),
        operands,
        op->getAttrs());

    // replace to relu->clip
    rewriter.replaceOp(op, {newOp});

    return success();
  }
};

class ConvertClipToRelu6Pass : public mlir::PassWrapper<ConvertClipToRelu6Pass, FunctionPass> {
public:
  explicit ConvertClipToRelu6Pass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertClipToRelu6Pattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};


} // namespace

std::unique_ptr<mlir::Pass> mlir::createConvertClipToRelu6Pass() {
  return std::make_unique<ConvertClipToRelu6Pass>();
}
