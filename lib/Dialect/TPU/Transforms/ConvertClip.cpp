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

#define DEBUG_TYPE "convert_scale"

using namespace mlir;

namespace {

struct TpuConvertClipToRelu6Pattern : public RewritePattern {
  TpuConvertClipToRelu6Pattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op);
    if (clipOp && clipOp.min().convertToFloat() == 0){
      auto formerOp = clipOp.input().getDefiningOp();
      auto formerName = getOpName(formerOp).str();
      // check if clipOp is rewrited already
      if (clipOp.fused_relu()) {
        return failure();
      }

      if (isa<tpu::ReluOp>(formerOp)){
          clipOp->setAttr("fused_relu", rewriter.getBoolAttr(true));
          return success();
      }
      // simply consider formerOp can fuse relu
      if (isOpSupportRelu(formerOp) && formerOp->getResult(0).hasOneUse()) {
        formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
        formerOp->setAttr("name", rewriter.getStringAttr(formerName+"_relu"));
        clipOp->setAttr("fused_relu", rewriter.getBoolAttr(true));
        return success();
      }
    }
    return failure();
  }
};

class ConvertClipToRelu6Pass : public mlir::PassWrapper<ConvertClipToRelu6Pass, FunctionPass> {
public:
  explicit ConvertClipToRelu6Pass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertClipToRelu6Pattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createConvertClipToRelu6Pass() {
  return std::make_unique<ConvertClipToRelu6Pass>();
}
