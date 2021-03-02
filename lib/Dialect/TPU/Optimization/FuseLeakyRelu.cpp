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
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

template<typename NextOpTy, typename PrevOpTy>
struct TpuTgFusePattern : public RewritePattern {
  TpuTgFusePattern(MLIRContext *context)
      : RewritePattern(NextOpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto leakyReluOp = cast<NextOpTy>(op);
    assert(leakyReluOp);

    auto prevOpInst = op->getOperand(0).getDefiningOp();
    if (matchPattern(prevOpInst, m_Op<PrevOpTy>())) {
      auto convOp = cast<PrevOpTy>(prevOpInst);
      convOp->setAttr("do_leaky_relu", rewriter.getBoolAttr(true));
      convOp->setAttr("negative_slope", leakyReluOp.negative_slopeAttr());
      if (leakyReluOp.rshift_pos().hasValue())
        convOp->setAttr("rshift_pos", leakyReluOp.rshift_posAttr());
      if (leakyReluOp.m_i8_pos().hasValue())
        convOp->setAttr("m_i8_pos", leakyReluOp.m_i8_posAttr());
      if (leakyReluOp.rshift_neg().hasValue())
        convOp->setAttr("rshift_neg", leakyReluOp.rshift_negAttr());
      if (leakyReluOp.m_i8_neg().hasValue())
        convOp->setAttr("m_i8_neg", leakyReluOp.m_i8_negAttr());

      // remove the relu Op
      convOp->setAttr("name", leakyReluOp.nameAttr());
      rewriter.replaceOp(op, {op->getOperand(0)});
      return success();
    }

    return failure();
  }
};

class TgFuseLeakyReluPass : public mlir::PassWrapper<TgFuseLeakyReluPass, FunctionPass> {
public:
  explicit TgFuseLeakyReluPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<
        TpuTgFusePattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_INT8_PT_Conv2DOp>,
        TpuTgFusePattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_INT8_PC_Conv2DOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createTgFuseLeakyReluPass() {
  return std::make_unique<TgFuseLeakyReluPass>();
}
