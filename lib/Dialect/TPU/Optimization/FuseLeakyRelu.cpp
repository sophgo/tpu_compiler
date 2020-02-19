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

template<typename NextOpTy, typename PrevOpTy>
struct TpuTgFusePattern : public RewritePattern {
  TpuTgFusePattern(MLIRContext *context)
      : RewritePattern(NextOpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto nextOp = cast<NextOpTy>(op);
    assert(nextOp);
    if (nextOp.fuse_prev()) {
      // fused already
      return matchFailure();
    }

    auto prevOpInst = op->getOperand(0)->getDefiningOp();
    if (matchPattern(prevOpInst, m_Op<PrevOpTy>())) {
      auto prevOp = cast<PrevOpTy>(prevOpInst);
      nextOp.setAttr("fuse_prev", rewriter.getBoolAttr(true));
      prevOp.setAttr("fuse_next", rewriter.getBoolAttr(true));
      return matchSuccess();
    }

    return matchFailure();
  }
};

class TgFuseLeakyReluPass : public FunctionPass<TgFuseLeakyReluPass> {
public:
  explicit TgFuseLeakyReluPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<
        TpuTgFusePattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_INT8_PT_Conv2DOp>,
        TpuTgFusePattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_INT8_PC_Conv2DOp>,
        TpuTgFusePattern<tpu::TG_BF16_LeakyReluOp, tpu::TG_BF16_Conv2DOp>
        >(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTgFuseLeakyReluPass() {
  return std::make_unique<TgFuseLeakyReluPass>();
}

static PassRegistration<TgFuseLeakyReluPass>
    pass("tg-fuse-leakyrelu",
         "Fuse leakyrelu with previous conv op, on TG Ops");
