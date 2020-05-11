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

#define DEBUG_TYPE "convert_scale"

using namespace mlir;

namespace {

struct TpuClipAsRelu6Pattern : public RewritePattern {
  TpuClipAsRelu6Pattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op);

    if (!clipOp) {
      return matchFailure();
    }

    auto min = clipOp.min().convertToFloat();
    if (min != 0) {
      // keep < 0 part
      return matchFailure();
    }

    auto formerOp = clipOp.getOperand(0)->getDefiningOp();
    if (isa<tpu::ReluOp>(formerOp)) {
      // has deal with
      return matchFailure();
    }

    auto loc = op->getLoc();

    auto layer_name = mlir::getOpName(clipOp).str();
    std::vector<Value *> newOperands;
    newOperands.push_back(clipOp.getOperand(0));

    // duplicate name for not mess up calibration table name
    std::string formerOpName = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(formerOpName)));
    attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    // insert relu before clip op
    auto reluOp = rewriter.create<tpu::ReluOp>(loc,
        clipOp.getOperand(0)->getType(),
        ArrayRef<Value *>{newOperands},
        ArrayRef<NamedAttribute>{attrs});

    // duplicate one
    std::vector<Value *> operands;
    auto NoneOp = rewriter.create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    operands.push_back(reluOp.getResult());
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    auto _clipOp = rewriter.create<tpu::ClipOp>(loc,
        op->getResult(0)->getType(),
        operands,
        op->getAttrs());

    // replace to relu->clip
    rewriter.replaceOp(op, {_clipOp});

    return matchSuccess();
  }
};

class ClipAsRelu6Pass : public FunctionPass<ClipAsRelu6Pass> {
public:
  explicit ClipAsRelu6Pass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuClipAsRelu6Pattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};


} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createClipAsRelu6Pass() {
  return std::make_unique<ClipAsRelu6Pass>();
}

static PassRegistration<ClipAsRelu6Pass>
    pass_1("relu6-to-clip",
         "relu6 convert to clip op");
