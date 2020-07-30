//===- ConvertQuantOp.cpp - convert quant op ----------------------------------===//
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
// This file implements the conversion of quant op.
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

#define DEBUG_TYPE "convert_quant_op"

using namespace mlir;

namespace {

struct ConvertQuantOpPattern : public RewritePattern {
  ConvertQuantOpPattern(MLIRContext *context)
      : RewritePattern("tpu.quant", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto quantOp = cast<tpu::QuantOp>(op);
    LLVM_DEBUG(llvm::errs() << quantOp.getOperationName() << ":"
                            << getOpName(quantOp)<< "\n";);

    auto builder = Builder(op->getContext());
    auto elementType = mlir::FloatType::getBF16(builder.getContext());
    llvm::ArrayRef<int64_t> shape = quantOp.getOperand()->getType().dyn_cast<mlir::TensorType>().getShape();
    auto result_type = RankedTensorType::get(shape, elementType);

    if (quantOp.from() == "NONE" && quantOp.to() == "INT8") {
      // input (fp32) -> quant (int8) ==> input (fp32) -> cast (bf16) -> quant (int8)
      auto preOp = quantOp.getOperand()->getDefiningOp();
      auto name = preOp->getAttrOfType<StringAttr>("name").getValue().str()
                       + "_cast";
      std::vector<NamedAttribute> cast_attrs;
      cast_attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
      cast_attrs.push_back(builder.getNamedAttr("from", builder.getStringAttr("FP32")));
      cast_attrs.push_back(builder.getNamedAttr("to", builder.getStringAttr("BF16")));
      cast_attrs.push_back(builder.getNamedAttr("layer_id", quantOp.layer_idAttr()));
      std::vector<Value *> operands(op->getOperands().begin(),
                                  op->getOperands().end());

      auto castOp = OpBuilder(op).create<tpu::CastOp>(op->getLoc(),
        result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{cast_attrs});
      auto result = castOp.getResult();

      std::vector<NamedAttribute> quant_attrs;
      quant_attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name + "_quant")));
      quant_attrs.push_back(builder.getNamedAttr("threshold", quantOp.thresholdAttr()));
      quant_attrs.push_back(builder.getNamedAttr("from", builder.getStringAttr("BF16")));
      quant_attrs.push_back(builder.getNamedAttr("to", builder.getStringAttr("INT8")));
      quant_attrs.push_back(builder.getNamedAttr("layer_id", quantOp.layer_idAttr()));
      std::vector<Value *> opds;
      opds.push_back(result);

      auto newOp = OpBuilder(op).create<tpu::QuantOp>(op->getLoc(),
        quantOp.getResult()->getType(), ArrayRef<Value *>{opds},
        ArrayRef<NamedAttribute>{quant_attrs});

      rewriter.replaceOp(op, {newOp});
    } else if (quantOp.from() == "INT8" && quantOp.to() == "NONE") {
      // op (int8) -> dequant (fp32) ==> op (int8) -> dequant (bf16) -> cast (fp32)
      auto name = quantOp.getAttrOfType<StringAttr>("name").getValue().str()
                       + "_cast";
      std::vector<NamedAttribute> quant_attrs;
      quant_attrs.push_back(builder.getNamedAttr("name", quantOp.nameAttr()));
      quant_attrs.push_back(builder.getNamedAttr("threshold", quantOp.thresholdAttr()));
      quant_attrs.push_back(builder.getNamedAttr("from", builder.getStringAttr("INT8")));
      quant_attrs.push_back(builder.getNamedAttr("to", builder.getStringAttr("BF16")));
      quant_attrs.push_back(builder.getNamedAttr("layer_id", quantOp.layer_idAttr()));
      std::vector<Value *> operands(op->getOperands().begin(),
                                  op->getOperands().end());

      auto newOp = OpBuilder(op).create<tpu::QuantOp>(op->getLoc(),
        result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{quant_attrs});
      auto result = newOp.getResult();

      std::vector<NamedAttribute> cast_attrs;
      cast_attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
      cast_attrs.push_back(builder.getNamedAttr("from", builder.getStringAttr("BF16")));
      cast_attrs.push_back(builder.getNamedAttr("to", builder.getStringAttr("FP32")));
      cast_attrs.push_back(builder.getNamedAttr("layer_id", quantOp.layer_idAttr()));
      std::vector<Value *> opds;
      opds.push_back(result);

      auto castOp = OpBuilder(op).create<tpu::CastOp>(op->getLoc(),
        quantOp.getResult()->getType(), ArrayRef<Value *>{opds},
        ArrayRef<NamedAttribute>{cast_attrs});

      rewriter.replaceOp(op, {castOp});
    } else {
      LLVM_DEBUG(llvm::errs() << "No need to convert from int8/bf16 to bf16/int8" << "\n";);
    }

    return matchSuccess();
  }
};

class ConvertQuantOpPass : public FunctionPass<ConvertQuantOpPass> {
public:
  explicit ConvertQuantOpPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();
    patterns.insert<ConvertQuantOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> createConvertQuantOpPass() {
  return std::make_unique<ConvertQuantOpPass>();
}

static PassRegistration<ConvertQuantOpPass>
    pass("convert-quant-op",
         "Convert cpu quant to tpu quant");