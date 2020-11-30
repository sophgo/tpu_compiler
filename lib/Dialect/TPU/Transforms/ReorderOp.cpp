//===- ReorderOp.cpp - convert cpu op ----------------------------------===//
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
// This file implements the conversion of cpu op.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_cpu_op"

using namespace mlir;

namespace {

template<typename OpTy>
struct EliminateReshapeOpPattern : public RewritePattern {
  EliminateReshapeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = llvm::cast<OpTy>(op);
    auto prevOp = castOp.getOperand().getDefiningOp();
    if (!llvm::isa<tpu::GenericCpuOp>(prevOp) &&
        !llvm::isa<tpu::InputOp>(prevOp)) {
      return failure();
    }
    if (!prevOp->getResult(0).hasOneUse()) {
      return failure();
    }
    auto type = castOp.getResult().getType();
    prevOp->getResult(0).setType(type);

    for (auto &use : castOp.getResult().getUses()) {
      auto nextOp = use.getOwner();
      for (uint32_t i = 0; i < nextOp->getNumOperands(); i++) {
        auto opd = nextOp->getOperand(i).getDefiningOp();
        if (opd == op) {
          llvm::errs() << "eliminate reshape op\n";
          nextOp->setOperand(i, castOp.getOperand());
        }
      }
    }
    return success();
  }
};

struct SinkCpuOPToReturnOpPattern : public RewritePattern {
  SinkCpuOPToReturnOpPattern(MLIRContext *context)
      : RewritePattern(ReturnOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // move all operands of ReturnOp to close to it.
    auto insertPoint = op;
    for (int i = (int)op->getNumOperands() - 1; i >= 0; --i) {
      auto opd = op->getOperand(i).getDefiningOp();
      if (!isa<tpu::GenericCpuOp>(opd) && !isa<tpu::ReshapeOp>(opd)) {
        continue;
      }
      opd->moveBefore(insertPoint);
      insertPoint = opd;
    }
    return success();
  }
};

static bool hasMoreUser(Operation *op) {
  int user = 0;
  for (auto &use : op->getResult(0).getUses()) {
    (void)use;
    user++;
  }
  return (user > 1);
}

template<typename OpTy>
struct MoveCpuOPToCloseUserPattern : public RewritePattern {
  MoveCpuOPToCloseUserPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    for (auto &use : op->getResult(0).getUses()) {
      auto nextOp = use.getOwner();
      if (!isa<tpu::GenericCpuOp>(nextOp) &&
          !isa<tpu::ReshapeOp>(nextOp) &&
          !isa<ReturnOp>(nextOp)) {
        return failure();
      }
    }
    auto insertPoint = op;
    auto opdNum = op->getNumOperands();
    for (int i = (int)opdNum - 1; i >= 0; --i) {
      auto opd = op->getOperand(i).getDefiningOp();
      if (!isa<tpu::GenericCpuOp>(opd) &&
          !isa<tpu::ReshapeOp>(opd)) {
        continue;
      }

      if (hasMoreUser(opd)) {
        continue;
      }

      opd->moveBefore(insertPoint);
      insertPoint = opd;
    }
    return success();
  }
};

static bool isUnaryOp(Operation *op) {
  int opd_num = 0;
  for (auto operand : op->getOperands()) {
    auto opd = operand.getDefiningOp();
    if ((!isa<tpu::LoadWeightOp>(opd))
        && (!isa<tpu::NoneOp>(opd))) {
      opd_num++;
    }
  }
  return (opd_num == 1);
}

static uint32_t getOpLine(Operation *op) {
  auto loc = op->getLoc().cast<FileLineColLoc>();
  return loc.getLine();
}

static void handleEltwiseOp(Operation *op) {
  auto opd0 = op->getOperand(0).getDefiningOp();
  auto opd1 = op->getOperand(1).getDefiningOp();
  auto prevOp = op->getPrevNode();
  if (prevOp == opd0 || prevOp == opd1) {
    return;
  }
  auto moveOp = (getOpLine(opd0) > getOpLine(opd1)) ? opd0 : opd1;
  uint32_t firstUseLineNo = -1;
  for (auto &use : moveOp->getResult(0).getUses()) {
    auto lineNo = getOpLine(use.getOwner());
    if (lineNo < firstUseLineNo) {
      firstUseLineNo = lineNo;
    }
  }
  if (firstUseLineNo == getOpLine(op))
    moveOp->moveBefore(op);
}

class ReorderOpPass : public mlir::PassWrapper<ReorderOpPass, FunctionPass> {
public:
  explicit ReorderOpPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    // re-order all OPs
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else if (llvm::isa<tpu::TpuTLOpCodegenInterface>(op)) {
        return;
      } else {
         auto current = op;
         if (isa<tpu::TG_INT8_EltwiseAddOp>(op) ||
             isa<tpu::TG_INT8_EltwiseMulOp>(op)) {
           handleEltwiseOp(op);
         }
         while (current->getResult(0).hasOneUse()) {
          auto use = getNextOp(current);
          if (isa<ReturnOp>(use) || !isUnaryOp(use))
            break;
          auto insertPoint = current->getNextNode();
          use->moveBefore(insertPoint);
          insertPoint = use;
          auto opdNum = use->getNumOperands();
          for (int i = (int)opdNum - 1; i >= 0; --i) {
            auto opd = use->getOperand(i).getDefiningOp();
            // opd may have multi-use, just keep it in front
            if (hasMoreUser(opd) && opd->isBeforeInBlock(use))
              continue;
            opd->moveBefore(insertPoint);
            insertPoint = opd;
          }
          current = use;
        }
      }
    });

    patterns.clear();
    patterns.insert<
        EliminateReshapeOpPattern<tpu::ReshapeOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        SinkCpuOPToReturnOpPattern
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        MoveCpuOPToCloseUserPattern<tpu::GenericCpuOp>,
        MoveCpuOPToCloseUserPattern<tpu::ReshapeOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createReorderOpPass() {
  return std::make_unique<ReorderOpPass>();
}
