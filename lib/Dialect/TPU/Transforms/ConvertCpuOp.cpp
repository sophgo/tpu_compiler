//===- ConvertCpuOp.cpp - convert cpu op ----------------------------------===//
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
// This file implements the conversion of cpu op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
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
    LLVM_DEBUG(llvm::errs() << castOp.getOperationName() << ":"
                            << getOpName(castOp)<< "\n";);

    auto builder = Builder(op->getContext());

    if (isa<tpu::QuantOp>(op)) {
      auto from = op->getAttr("from").cast<StringAttr>().getValue().str();
      auto to = op->getAttr("to").cast<StringAttr>().getValue().str();

      if (from == "NONE" || to == "NONE") {
        // pass to cpu
      }
      else {
        // pass to tpu
        LLVM_DEBUG(llvm::errs() << "mix: " << castOp.getOperationName() << ":" << getOpName(castOp)<< "\n";);
        return matchFailure();
      }
    }

    std::vector<NamedAttribute> param;
    tpu::QuantParam quantAttr = getDefaultQuantParam(builder);
    for (auto& attr : op->getAttrs()) {
      if (attr.first == "quant") {
        quantAttr = attr.second.cast<tpu::QuantParam>();
      } else {
        param.push_back(attr);
      }
    }

    std::vector<NamedAttribute> attrs;
    auto nameAttr = builder.getStringAttr(castOp.name());
    auto operationAttr = builder.getStringAttr(castOp.getOperationName());
    auto quantifiableAttr = builder.getBoolAttr(false);
    auto paramAttr = builder.getDictionaryAttr(param);

    attrs.push_back(builder.getNamedAttr("name", nameAttr));
    attrs.push_back(builder.getNamedAttr("operation_name", operationAttr));
    attrs.push_back(builder.getNamedAttr("quantifiable", quantifiableAttr));
    attrs.push_back(builder.getNamedAttr("quant", quantAttr));
    attrs.push_back(builder.getNamedAttr("param", paramAttr));
    auto gaddrAttr = op->getAttr("gaddr").dyn_cast_or_null<IntegerAttr>();
    if (gaddrAttr) {
      int64_t gaddr = gaddrAttr.getValue().getSExtValue();
      attrs.push_back(builder.getNamedAttr("gaddr",
          builder.getI64IntegerAttr(gaddr)));
    }
    if (castOp.layer_id().hasValue()) {
      int32_t layer_id = castOp.layer_id().getValue().getSExtValue();
      attrs.push_back(builder.getNamedAttr("layer_id",
          builder.getI32IntegerAttr(layer_id)));
    }

    std::vector<Value *> operands(op->getOperands().begin(),
                                  op->getOperands().end());

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
