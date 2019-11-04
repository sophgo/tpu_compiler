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

struct TpuFuseReluIntoConvPattern : public RewritePattern {
  TpuFuseReluIntoConvPattern(MLIRContext *context)
      : RewritePattern("tpu.relu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    llvm::errs() << reluOp.getOperationName() << "\n";

    // match consecutive relu operations
    auto formerOp = op->getOperand(0)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::Conv2DOp>()))
      return matchFailure();
    auto convOp = cast<tpu::Conv2DOp>(formerOp);
    assert(convOp.fused_activation_function() == "NONE");

    // set fused_activation_function for conv Op
    convOp.setAttr("fused_activation_function", rewriter.getStringAttr("RELU"));

    // remove the relu Op
    rewriter.replaceOp(op, {op->getOperand(0)});

    return matchSuccess();
  }
};

class FuseReluIntoConvPass : public FunctionPass<FuseReluIntoConvPass> {
public:
  explicit FuseReluIntoConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFuseReluIntoConvPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createFuseReluIntoConvPass() {
  return std::make_unique<FuseReluIntoConvPass>();
}

static PassRegistration<FuseReluIntoConvPass>
    pass("fuse-relu-into-conv",
         "Fuse relu op into conv op");
