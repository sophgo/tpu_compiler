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

#define DEBUG_TYPE "deep-fusion-tl-la2lw"

using namespace mlir;

// TODO: move to backend
static const struct MachineInfo {
  const int lane_num = 32;
  const int eu_num = 16;
  const uint64_t lmem_per_lane = 32 * 1024;
} mInfo;

static uint64_t getSizePerLane(int n, int c, int h, int w, bool eu_align) {
  uint64_t channelPerLane = llvm::alignTo(c, mInfo.lane_num) / mInfo.lane_num;
  uint64_t bytesPerChannel = h * w;
  if (eu_align) {
    bytesPerChannel = llvm::alignTo(bytesPerChannel, mInfo.eu_num);
  }
  // total number align to eu_num is mandatory
  return llvm::alignTo(n * channelPerLane * bytesPerChannel, mInfo.eu_num);
}


namespace {

struct TpuTL_LA_Conv2DOpPattern : public RewritePattern {
  TpuTL_LA_Conv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tl_la_conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TL_LA_Conv2DOp>(opInst);
    //auto loc = op->getLoc();

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    if (1) {
      llvm::errs() << "TL_LA2LW: layer ID " << op.layer_id() << "\n";
      return matchFailure();
    }
  }
};

class DeepFusionTL_LA2LW : public FunctionPass<DeepFusionTL_LA2LW> {
public:
  explicit DeepFusionTL_LA2LW() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuTL_LA_Conv2DOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionTL_LA2LW() {
  return std::make_unique<DeepFusionTL_LA2LW>();
}

static PassRegistration<DeepFusionTL_LA2LW>
    pass("deep-fusion-tl-la2lw",
         "convert TL Conv Ops from LA to LW.");
