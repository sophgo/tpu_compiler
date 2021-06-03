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

#define DEBUG_TYPE "fuseleakyrelu"

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

static int getUsesNrExceptReshape (Operation *owner, std::vector<Operation*> & ops) {
  int nr = 0;
  for (uint32_t i = 0; i < owner->getNumResults(); ++i) {
    for (auto &use : owner->getResult(i).getUses()) {
      Operation *owner = use.getOwner();
      if (isa<tpu::ReshapeOp>(owner)) {
        nr += getUsesNrExceptReshape(owner, ops);
      }
      else {
        ops.push_back(owner);
        nr++;
      }
    }
  }
  return nr;
}

template<typename NextOpTy>
struct TpuTgFusePrevPattern : public RewritePattern {
  TpuTgFusePrevPattern(MLIRContext *context)
      : RewritePattern(NextOpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto leakyReluOp = cast<NextOpTy>(op);
    assert(leakyReluOp);

    // try to fuse next if only one reference
    std::vector<Operation*> ops;
    int usesNumber = getUsesNrExceptReshape(op->getResult(0).getDefiningOp(), ops); 
    if (usesNumber > 1) {
      LLVM_DEBUG(llvm::errs() << "fuse next fail , over " << usesNumber << " used, cnt merge " << op->getName() << "\n";);
      return failure();
    }

    auto next = ops[0];
    // TODO: support deconv for tl case
    if (isa<tpu::TG_BF16_Conv2DOp>(next)
      //|| isa<tpu::TG_BF16_DeConv2DOp>(next)
      || isa<tpu::TG_INT8_PC_Conv2DOp>(next)) {

      next->setAttr("prev_leaky_relu", rewriter.getBoolAttr(true));
      next->setAttr("do_leaky_relu", rewriter.getBoolAttr(true));
      next->setAttr("negative_slope", leakyReluOp.negative_slopeAttr());
      if (isa<tpu::TG_INT8_PC_Conv2DOp>(next)) {
        auto convOp = cast<tpu::TG_INT8_PC_Conv2DOp>(next);
        convOp->setAttr("negative_slope", leakyReluOp.negative_slopeAttr());
        convOp->setAttr("rshift_pos", leakyReluOp.rshift_posAttr());
        convOp->setAttr("m_i8_pos", leakyReluOp.m_i8_posAttr());
        convOp->setAttr("rshift_neg", leakyReluOp.rshift_negAttr());
        convOp->setAttr("m_i8_neg", leakyReluOp.m_i8_negAttr());
      }

      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
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
        TpuTgFusePattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_INT8_PC_Conv2DOp>,
        TpuTgFusePrevPattern<tpu::TG_BF16_LeakyReluOp>,
        TpuTgFusePrevPattern<tpu::TG_INT8_LeakyReluOp>
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
