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
      assert(convOp.param().do_relu().getValue() == false);
      // set do_relu
      convOp.setAttr("param",
          tpu::ConvParam::get(
              convOp.param().stride_h(),
              convOp.param().stride_w(),
              convOp.param().padding(),
              convOp.param().dilation_h(),
              convOp.param().dilation_w(),
              convOp.param().group(),
              convOp.param().is_dw(),
              convOp.param().with_bias(),
              rewriter.getBoolAttr(true),
              rewriter.getContext()));
      convOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
      // remove the relu Op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseAddOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMaxOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::EltwiseMulOp>())) {
      auto eltOp = cast<tpu::EltwiseAddOp>(formerOp);
      eltOp.setAttr("do_relu", rewriter.getBoolAttr(true));
      eltOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::FullyConnectedOp>())) {
      auto fcOp = cast<tpu::FullyConnectedOp>(formerOp);
      assert(fcOp.fused_activation_function() == "NONE");
      // set fused_activation_function for fc Op
      fcOp.setAttr("fused_activation_function", rewriter.getStringAttr("RELU"));
      fcOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
      // remove the relu Op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::ConcatOp>())) {
      // Do nothing
      return matchSuccess();
    } else if (matchPattern(formerOp, m_Op<tpu::ScaleOp>())) {
      // scale is implemented by depthwise convolution in backend
      // Hence, we can fuse relu into scale
      auto scaleOp = cast<tpu::ScaleOp>(formerOp);
      assert(scaleOp.fused_activation_function() == "NONE");
      // set fused_activation_function for scale Op
      scaleOp.setAttr("fused_activation_function", rewriter.getStringAttr("RELU"));
      scaleOp.setAttr("name", rewriter.getStringAttr(reluOp.getOpName()));
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

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFuseReluPass() {
  return std::make_unique<FuseReluPass>();
}

static PassRegistration<FuseReluPass>
    pass("fuse-relu",
         "Fuse relu op into previous op (conv/eltwise etc)");
