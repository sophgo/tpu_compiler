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
// This file implements the convert swish to relu.
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

#define DEBUG_TYPE "convert_swish_to_relu"

using namespace mlir;

namespace {

template<typename OpTy>
struct TpuConvertSwishToReluPattern : public RewritePattern {
  TpuConvertSwishToReluPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);

    std::vector<Value> operands;
    operands.push_back(castOp.getOperand(0));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr("quant", castOp.quantAttr()));
    rewriter.replaceOpWithNewOp<tpu::ReluOp>(
        castOp, castOp.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

template <typename opTy>
struct TpuFuseSigmoidEltMulToSwishPattern : public RewritePattern {
  TpuFuseSigmoidEltMulToSwishPattern(MLIRContext *context)
      : RewritePattern(opTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<opTy>(op);
    auto opd0 = op->getOperand(0).getDefiningOp();
    auto opd1 = op->getOperand(1).getDefiningOp();

    std::vector<Value> operands;
    if (isa<tpu::SigmoidOp>(opd0)) {
      auto prevOp = opd0->getOperand(0).getDefiningOp();
      if (prevOp->getResult(0) != opd1->getResult(0)) {
        return failure();
      }
      auto NoneOp = rewriter.create<tpu::NoneOp>(op->getLoc(), rewriter.getNoneType());
      operands.push_back(castOp.getOperand(1));
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("quant", castOp.quantAttr()));
      rewriter.replaceOpWithNewOp<tpu::SwishOp>(
          castOp, castOp.getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      rewriter.eraseOp(opd0);

    } else if (isa<tpu::SigmoidOp>(opd1)) {
      auto prevOp = opd1->getOperand(0).getDefiningOp();
      if (prevOp->getResult(0) != opd0->getResult(0)) {
        return failure();
      }
      auto NoneOp = rewriter.create<tpu::NoneOp>(op->getLoc(), rewriter.getNoneType());
      operands.push_back(castOp.getOperand(0));
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("quant", castOp.quantAttr()));
      rewriter.replaceOpWithNewOp<tpu::SwishOp>(
          castOp, castOp.getResult().getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      rewriter.eraseOp(opd1);
    }
    return success();
  }
};

class ConvertSwishToReluPass : public mlir::PassWrapper<ConvertSwishToReluPass, FunctionPass> {
public:
  explicit ConvertSwishToReluPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<
        TpuConvertSwishToReluPattern<tpu::SwishOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

class ConvertSwishPass : public mlir::PassWrapper<ConvertSwishPass, FunctionPass> {
public:
  explicit ConvertSwishPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<
        TpuFuseSigmoidEltMulToSwishPattern<tpu::EltwiseMulOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createConvertSwishToReluPass() {
  return std::make_unique<ConvertSwishToReluPass>();
}

std::unique_ptr<mlir::Pass> mlir::createFuseSigmoidEltMulToSwishPass() {
  return std::make_unique<ConvertSwishPass>();
}
