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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "MachineInfo.h"
#include "SimpleAnalysis.h"

#define DEBUG_TYPE "deep-fusion-tg2tl-la"

using namespace mlir;

namespace {

struct TpuTG2TLConv2DOpPattern : public RewritePattern {
  TpuTG2TLConv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_pc_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TG_INT8_PC_Conv2DOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";
      return matchFailure();
    }

    if (1) {
      llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";

      assert(op.getNumOperands() == 3);
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));
      newOperands.push_back(op.getOperand(2));

      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));
      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));
      rewriter.replaceOpWithNewOp<tpu::TL_LA_Conv2DOp>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

class DeepFusionTG2TL_LA : public FunctionPass<DeepFusionTG2TL_LA> {
public:
  explicit DeepFusionTG2TL_LA() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuTG2TLConv2DOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionTG2TL_LA() {
  return std::make_unique<DeepFusionTG2TL_LA>();
}

static PassRegistration<DeepFusionTG2TL_LA>
    pass("deep-fusion-tg2tl-la",
         "convert Ops from TG to TL, "
         "this is a trivial conversion, yielding no improvement at all");
