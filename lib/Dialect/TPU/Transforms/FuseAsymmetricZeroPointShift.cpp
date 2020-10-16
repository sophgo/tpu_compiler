//==================- FuseAsymetricZeroPointShift.cpp ------------------------------===//
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
// Now Asymmetric Quantization use depthwise convolution add bias(zero point)
// With former layer and next layer, we can fused the opsite bias value convolution
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse_asymmetric_zero_point"

using namespace mlir;
namespace {
struct TpuFuseAsymmetricZeroPointPattern : public RewritePattern {
  TpuFuseAsymmetricZeroPointPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    auto forop = op->getOperand(0)->getDefiningOp();
    auto formerOp = llvm::dyn_cast_or_null<tpu::Conv2DOp>(
        op->getOperand(0)->getDefiningOp());
    if (!formerOp) {
      return matchFailure();
    }

    TensorFile *wTF = getWeightTensorFile(op);
    // we use depthwise conv do zero point,
    // and set only rshift is 0,
    // if not rshift_only, pass
    if (getOpQuantParamType(op) != "RSHIFT_ONLY" ||
        getOpQuantParamType(op->getOperand(0)->getDefiningOp()) !=
            "RSHIFT_ONLY") {
      return matchFailure();
    }

    std::unique_ptr<std::vector<float>> conv_zero_point;
    std::unique_ptr<std::vector<float>> former_zero_point;

    // zero point save in bias
    auto conv_zero_point_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        convOp.getOperand(2)->getDefiningOp());
    auto conv_zero_point_name = conv_zero_point_op.name();
    auto type = conv_zero_point_op.getResult()->getType().cast<TensorType>();
    conv_zero_point = wTF->readTensor<float>(conv_zero_point_name, type);

    auto former_zero_point_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        formerOp.getOperand(2)->getDefiningOp());
    auto former_zero_point_name = former_zero_point_op.name();
    former_zero_point = wTF->readTensor<float>(former_zero_point_name, type);

    if (!std::equal(conv_zero_point->begin() + 1, conv_zero_point->end(),
                    conv_zero_point->begin()) ||
        !std::equal(former_zero_point->begin() + 1, former_zero_point->end(),
                    former_zero_point->begin())) {
      // zero point must all equal
      return matchFailure();
    }

    if (conv_zero_point->at(0) != -former_zero_point->at(0)) {
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs()
                   << "Fuse asymmetric zero point, zero point value: "
                   << conv_zero_point->at(0) << "\n\t"
                   << formerOp.getOperationName() << " : "
                   << getOpName(formerOp) << "\n\t" << convOp.getOperationName()
                   << " : " << getOpName(op) << "\n";);
    rewriter.replaceOp(op, {forop->getOperand(0)});

    return matchSuccess();
  };
};

class FuseAsymmetricZeroPointPass
    : public FunctionPass<FuseAsymmetricZeroPointPass> {
public:
  explicit FuseAsymmetricZeroPointPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<TpuFuseAsymmetricZeroPointPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFuseAsymmetricZeroPointPass() {
  return std::make_unique<FuseAsymmetricZeroPointPass>();
}

static PassRegistration<FuseAsymmetricZeroPointPass>
    pass("fuse-asymmetric-zero-point", "Fuse asymmetric zero point");
