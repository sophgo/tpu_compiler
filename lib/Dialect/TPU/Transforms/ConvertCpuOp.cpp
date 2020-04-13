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
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"

#define DEBUG_TYPE "convert_cpu_op"

using namespace mlir;

namespace {

template <typename OpTy>
struct ConvertCpuOpDefaultPattern : public RewritePattern {
  ConvertCpuOpDefaultPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    LLVM_DEBUG(llvm::errs() << castOp.getOperationName() << ":" << getOpName(castOp)<< "\n";);

    auto builder = Builder(op->getContext());
    std::vector<NamedAttribute> param;
    for (auto& attr : op->getAttrs()) {
      param.push_back(attr);
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(castOp.name())));
    attrs.push_back(builder.getNamedAttr("operation_name", builder.getStringAttr(castOp.getOperationName())));
    attrs.push_back(builder.getNamedAttr("quantifiable", builder.getBoolAttr(false)));
    attrs.push_back(builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
    attrs.push_back(builder.getNamedAttr("param", builder.getDictionaryAttr(param)));
    auto gaddrAttr = op->getAttr("gaddr").dyn_cast_or_null<IntegerAttr>();
    if (gaddrAttr) {
      int64_t gaddr = gaddrAttr.getValue().getSExtValue();
      attrs.push_back(builder.getNamedAttr("gaddr", builder.getI64IntegerAttr(gaddr)));
    }
    if (castOp.layer_id().hasValue()) {
      int32_t layer_id = castOp.layer_id().getValue().getSExtValue();
      attrs.push_back(builder.getNamedAttr("layer_id", builder.getI32IntegerAttr(layer_id)));
    }

    std::vector<Value *> operands(op->getOperands().begin(), op->getOperands().end());

    auto newOp = OpBuilder(op).create<tpu::GenericCpuOp>(op->getLoc(),
        castOp.getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    auto result = newOp.getResult();
    rewriter.replaceOp(op, {result});

    return matchFailure();
  }
};

class ConvertCpuOpPass : public FunctionPass<ConvertCpuOpPass> {
public:
  explicit ConvertCpuOpPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<
        ConvertCpuOpDefaultPattern<tpu::SoftmaxOp>,
        ConvertCpuOpDefaultPattern<tpu::QuantOp>,
        ConvertCpuOpDefaultPattern<tpu::DetectionOutputOp>,
        ConvertCpuOpDefaultPattern<tpu::RetinaFaceDetectionOp>,
        ConvertCpuOpDefaultPattern<tpu::PreprocessOp>,
        ConvertCpuOpDefaultPattern<tpu::TransposeOp>,
        ConvertCpuOpDefaultPattern<tpu::YoloDetectionOp>
      >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> createConvertCpuOpPass() {
  return std::make_unique<ConvertCpuOpPass>();
}

static PassRegistration<ConvertCpuOpPass>
    pass("convert-cpu-op",
         "Convert CPU op to GenericCpuOp");
