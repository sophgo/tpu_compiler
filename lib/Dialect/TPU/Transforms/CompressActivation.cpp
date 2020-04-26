//===- CompressWeight- Implementation of activation compression -----------===//
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
// This file implements the activation compression.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUCompressUtil.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "../DeepFusion/MachineInfo.h"

#define DEBUG_TYPE "compress-activation"

using namespace mlir;

namespace {

template <typename OpTy>
uint64_t calcConv2DMemoryUsage(OpTy &op) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);
  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;
  // filter working size *2 for double buffer
  if (g != oc) {
    if(g != 1) { // TODO, not support group convolution now.
      return MInfo::lmem_per_lane + 1;
    }
    filterSizePerLane = MInfo::getSizePerLane(ic, oc, kh, kw, false) ;
  }

  // load bias all in once
  int bias_size = with_bias ? 9 : 5;
  uint64_t biasSizePerLane = MInfo::getSizePerLane(1, oc, 1, bias_size, false);

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane +
                          filterSizePerLane + biasSizePerLane;

  return totalPerLane;
}

bool canStoreCompressedActivation(Operation *op) {
  bool storeCompressed= false;

  // Check if all user operation does not need to do tiling.
  // Only support conv->conv now.
  for (auto &use : op->getResult(0)->getUses()) {
    uint64_t totalPerLane = MInfo::lmem_per_lane + 1;
    auto useOp = use.getOwner();
    if (auto useTpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
      totalPerLane =
          calcConv2DMemoryUsage<tpu::TG_INT8_PC_Conv2DOp>(useTpuOp);
    } else {
      storeCompressed = false;
      break;
    }

    if (totalPerLane <= MInfo::lmem_per_lane)
      storeCompressed = true;
    else {
      storeCompressed = false;
      break;
    }
  }

  return storeCompressed;
}

bool canLoadCompressedActivation(Operation *op) {

  // Check if input operation store compressed activation.
  // Only support conv->conv now.
  for (auto operand : op->getOperands()) {
    auto operandOp = operand->getDefiningOp();
    if (auto operandTpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
      if (operandTpuOp.store_compr_act().hasValue())
        return true;
    }
  }

  return false;
}

template <typename OpTy>
class StoreCompressedConvolutionActivationPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StoreCompressedConvolutionActivationPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  PatternMatchResult matchAndRewrite(OpTy convOp,
                                     PatternRewriter &rewriter) const override {
    // Already marked.
    if (convOp.store_compr_act().hasValue())
      return Pattern::matchFailure();

    uint64_t totalPerLane = calcConv2DMemoryUsage<OpTy>(convOp);
    if (totalPerLane > MInfo::lmem_per_lane)
      return Pattern::matchFailure();

    // A operation needs to generate the compressed activation first then
    // the user operation has to load the compressed activation.
    auto op = convOp.getOperation();
    if (!canStoreCompressedActivation(op))
      return Pattern::matchFailure();

    convOp.setAttr("store_compr_act",
                   Builder(op->getContext()).getBoolAttr(true));

    LLVM_DEBUG(llvm::dbgs()
               << "StoreCompressedConvolutionActivationPattern: op "
               << convOp.name()
               << ", layer ID " << convOp.layer_id()
               << ", store compressed activation\n");

    return Pattern::matchSuccess();
  }
};

template <typename OpTy>
class LoadCompressedConvolutionActivationPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LoadCompressedConvolutionActivationPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  PatternMatchResult matchAndRewrite(OpTy convOp,
                                     PatternRewriter &rewriter) const override {
    // Already marked.
    if (convOp.load_compr_act().hasValue())
      return Pattern::matchFailure();

    auto op = convOp.getOperation();
    if (!canLoadCompressedActivation(op))
      return Pattern::matchFailure();

    convOp.setAttr("load_compr_act",
                   Builder(op->getContext()).getBoolAttr(true));

    LLVM_DEBUG(llvm::dbgs()
               << "LoadCompressedConvolutionActivationPattern: op "
               << convOp.name()
               << ", layer ID " << convOp.layer_id()
               << ", load compressed activation\n");

    return Pattern::matchSuccess();
  }
};

struct CompressActivationPass : public FunctionPass<CompressActivationPass> {
  void runOnFunction() override;
};

} // anonymous namespace

void CompressActivationPass::runOnFunction() {
  OwningRewritePatternList patterns;

  // Determine whether the operation can store compressed activation.
  patterns.insert<
      StoreCompressedConvolutionActivationPattern<tpu::TG_INT8_PC_Conv2DOp>
      >(&getContext());
  applyPatternsGreedily(getFunction(), patterns);

  // Determine whether the operation can load compressed activation.
  patterns.clear();

  patterns.insert<
      LoadCompressedConvolutionActivationPattern<tpu::TG_INT8_PC_Conv2DOp>
      >(&getContext());
  applyPatternsGreedily(getFunction(), patterns);

}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createCompressActivationPass() {
  return std::make_unique<CompressActivationPass>();
}

static PassRegistration<CompressActivationPass>
    pass("compress-activation", "Compress activation");
