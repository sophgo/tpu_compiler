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
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TpuFuseReluPattern : public RewritePattern {
  TpuFuseReluPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    llvm::errs() << reluOp.getOperationName() << "\n";

    // match relu Op that is following conv or eltwise Ops
    auto formerOp = op->getOperand(0)->getDefiningOp();

    if (matchPattern(formerOp, m_Op<tpu::Conv2DOp>())) {
      auto convOp = cast<tpu::Conv2DOp>(formerOp);
      assert(convOp.fused_activation_function() == "NONE");
      // set fused_activation_function for conv Op
      convOp.setAttr("fused_activation_function", rewriter.getStringAttr("RELU"));
      // remove the relu Op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseOp>())) {
      auto eltOp = cast<tpu::EltwiseOp>(formerOp);
      assert(eltOp.fused_activation_function() == "NONE");
      // set fused_activation_function for conv Op
      eltOp.setAttr("fused_activation_function", rewriter.getStringAttr("RELU"));
      // remove the relu Op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    }

    assert(0);
    return matchFailure();
  }
};

class FuseReluPass : public FunctionPass<FuseReluPass> {
public:
  explicit FuseReluPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFuseReluPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createFuseReluPass() {
  return std::make_unique<FuseReluPass>();
}

static PassRegistration<FuseReluPass>
    pass("fuse-relu",
         "Fuse relu op into previous op (conv/eltwise etc)");
