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
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TpuBatchNormOpPattern : public OpRewritePattern<tpu::BatchNormOp> {
  using OpRewritePattern<tpu::BatchNormOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::BatchNormOp op,
                                     PatternRewriter &rewriter) const {
    llvm::errs() << "match " << op.getOperationName() << "\n";
    // rewrite all.
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        op, op.getResult()->getType(),
        ArrayRef<Value *>{op.getOperand(0), op.getOperand(1), op.getOperand(2)},
        ArrayRef<NamedAttribute>{});

    return matchSuccess();
    //return matchFailure();
  }
};

class ConvertBnToScalePass : public FunctionPass<ConvertBnToScalePass> {
public:
  explicit ConvertBnToScalePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    //OpBuilder b(fn.getBody());
    //fn.walk<mlir::tpu::BatchNormOp>([&](mlir::tpu::BatchNormOp op) {
    //  os << " > " << op.getOperationName() << "\n";
    //});

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuBatchNormOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createConvertBnToScalePass() {
  return std::make_unique<ConvertBnToScalePass>();
}

static PassRegistration<ConvertBnToScalePass>
    pass("convert-bn-to-scale",
         "Convert a BN operation to Scale operation");
