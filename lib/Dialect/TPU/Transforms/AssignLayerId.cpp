//===- AssignLayerId - Implementation of Layer id assignment --------------===//
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
// This file implements the TPU layer id pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
template<typename T>
struct TpuLayerIdPattern : public RewritePattern {
  TpuLayerIdPattern(MLIRContext *context, StringRef opName, uint32_t *layer_id)
      : RewritePattern(opName, 1, context),
        layer_id_(layer_id) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
    auto castOp = cast<T>(op);
    if (castOp.layer_id().hasValue()) {
      // already assigned
      return matchFailure();
    }

    castOp.setAttr("layer_id", rewriter.getI32IntegerAttr(++*layer_id_));

    return matchSuccess();
  }

  uint32_t *layer_id_;
};

class AssignLayerIdPass : public FunctionPass<AssignLayerIdPass> {
public:
  explicit AssignLayerIdPass() {}

  void runOnFunction() override {
    auto fn = getFunction();

    uint32_t layer_id = 0;
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuLayerIdPattern<tpu::Conv2DOp> >(
        context, "tpu.conv_2d", &layer_id);
    patterns.insert<TpuLayerIdPattern<tpu::FullyConnectedOp> >(
        context, "tpu.fully_connected", &layer_id);
    patterns.insert<TpuLayerIdPattern<tpu::Pool2DOp> >(
        context, "tpu.pool_2d", &layer_id);
    patterns.insert<TpuLayerIdPattern<tpu::EltwiseOp> > (
        context, "tpu.eltwise", &layer_id);
    patterns.insert<TpuLayerIdPattern<tpu::PReluOp>>(context, "tpu.prelu",
                                                       &layer_id);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignLayerIdPass() {
  return std::make_unique<AssignLayerIdPass>();
}

static PassRegistration<AssignLayerIdPass>
    pass("assign-layer-id",
         "Assign layer id to each tpu op");
