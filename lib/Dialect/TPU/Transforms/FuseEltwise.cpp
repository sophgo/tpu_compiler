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

struct TpuFuseEltwisePattern : public RewritePattern {
  TpuFuseEltwisePattern(MLIRContext *context)
      : RewritePattern("tpu.eltwise", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltwiseOp = cast<tpu::EltwiseOp>(op);
    llvm::errs() << eltwiseOp.getOperationName() << "\n";

    // match eltwise Op that is following conv Ops
    // assuming second operand is the later one (the one to fuse into)
    // the first operand is the former one (the one we use its output as eltwise input)
    // TODO: need to find a way to ensure this
    assert(eltwiseOp.getNumOperands() == 2);
    auto formerOp = op->getOperand(1)->getDefiningOp();

    if (matchPattern(formerOp, m_Op<tpu::Conv2DOp>())) {
      // check the other input is conv2DOp or Eltwise
      auto input_0 = op->getOperand(0);
      assert(matchPattern(input_0->getDefiningOp(), m_Op<tpu::Conv2DOp>())
          || matchPattern(input_0->getDefiningOp(), m_Op<tpu::EltwiseOp>()));

      auto convOp = cast<tpu::Conv2DOp>(formerOp);
      assert(convOp.fused_activation_function() == "NONE");
      llvm::errs() << "  " << eltwiseOp.name() << " --> " << convOp.name() << "\n";

      // handle threshold_y or eltwise
      if (convOp.threshold_y().hasValue()) {
        float conv_threshold_y = convOp.threshold_y().getValue().convertToFloat();
        convOp.setAttr("threshold_y_before_eltwise",
            rewriter.getF32FloatAttr(conv_threshold_y));
        assert(eltwiseOp.threshold_y().hasValue());
        float eltwise_threshold_y = eltwiseOp.threshold_y().getValue().convertToFloat();
        convOp.setAttr("threshold_y",
            rewriter.getF32FloatAttr(eltwise_threshold_y));
      } else {
        llvm::errs() << "WARNING: fuse eltwise should be done after import calibration table\n";
      }

      // set Attr
      convOp.setAttr("fused_eltwise_method", rewriter.getStringAttr(eltwiseOp.method()));
      convOp.setAttr("fused_activation_function_after_eltwise",
          rewriter.getStringAttr(eltwiseOp.fused_activation_function()));

      // add eltwise input as conv input, append as a new append
      SmallVector<Value *, 4> newOperands;
      newOperands.append(formerOp->getOperands().begin(), formerOp->getOperands().end());
      newOperands.append({input_0});
      if (0) {
        // FIXME: sometimes setOperands is not allowed
        formerOp->setOperands(newOperands);
        //formerOp->setOperand(formerOp->getNumOperands(), input_0);
        // remove the eltwise Op
        rewriter.replaceOp(op, {op->getOperand(1)});
      } else {
        // replace the op with a new conv op
        auto origAttrs = convOp.getAttrs();
        std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
        rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
            eltwiseOp, convOp.getResult()->getType(),
            ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
      }

      return matchSuccess();
    }

    // assuming eltwise always follows conv Op for now
    assert(0);

    return matchFailure();
  }
};

class FuseEltwisePass : public FunctionPass<FuseEltwisePass> {
public:
  explicit FuseEltwisePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFuseEltwisePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createFuseEltwisePass() {
  return std::make_unique<FuseEltwisePass>();
}

static PassRegistration<FuseEltwisePass>
    pass("fuse-eltwise",
         "Fuse eltwise op into previous conv op");
