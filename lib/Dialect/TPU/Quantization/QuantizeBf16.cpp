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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "quantize_bf16"

using namespace mlir;

namespace mlir {

///
/// Conv Ops quantization method
///
template<typename OpTy>
LogicalResult quantizeBf16ConvOps(Operation *op) {
  auto convOp = cast<OpTy>(op);
  TensorFile *wTF = getWeightTensorFile(op);

  // get filter tensor
  auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
  std::vector<int64_t> filterShape;
  int64_t filterSize;
  getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
  assert(filterSize == (int64_t)filter->size());

  // get oc and isz
  int64_t oc = 0;
  if (filterShape.size() == 4) {
    oc = filterShape[0];
  } else if (filterShape.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = filterShape[0] * filterShape[1];
  } else {
    assert(0);
  }
  assert(filterSize % oc == 0);
  //int64_t isz = filterSize / oc;

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(convOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
    getTensorShapeAndSize(convOp.bias(), biasShape, biasSize);
    assert(biasSize == oc);
    assert(biasSize == (int64_t)bias->size());
  }

  // create new tensors
  auto new_filter = std::make_unique<std::vector<bfloat16> >(filterSize);
  std::unique_ptr<std::vector<bfloat16> > new_bias = nullptr;
  if (bias) {
    new_bias = std::make_unique<std::vector<bfloat16> >(biasSize);
  }

  // quantization
  FloatToBFloat16(filter->data(), new_filter->data(), filterSize);
  if (bias) {
    FloatToBFloat16(bias->data(), new_bias->data(), biasSize);
  }

  // update op
  addWeightTensorAndUpdateWeightOp<bfloat16>(convOp.getOperand(1),
      "quant", *new_filter, filterShape, "BF16", wTF);
  if (bias) {
    addWeightTensorAndUpdateWeightOp<bfloat16>(convOp.getOperand(2),
        "quant", *new_bias, biasShape, "BF16", wTF);
  }
  setOpQuant(op, "BF16");
  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// FC Ops quantization method
///
LogicalResult quantizeBf16FullyConnectedOps(Operation *op) {
  auto fcOp = cast<tpu::FullyConnectedOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);

  // get filter tensor
  auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
  std::vector<int64_t> filterShape;
  int64_t filterSize;
  getTensorShapeAndSize(fcOp.filter(), filterShape, filterSize);

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(fcOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
    getTensorShapeAndSize(fcOp.bias(), biasShape, biasSize);
  }

  // create new tensors
  auto new_filter = std::make_unique<std::vector<bfloat16> >(filterSize);
  std::unique_ptr<std::vector<bfloat16> > new_bias = nullptr;
  if (bias) {
    new_bias = std::make_unique<std::vector<bfloat16> >(biasSize);
  }

  // quantization
  FloatToBFloat16(filter->data(), new_filter->data(), filterSize);
  if (bias) {
    FloatToBFloat16(bias->data(), new_bias->data(), biasSize);
  }

  // update op
  addWeightTensorAndUpdateWeightOp<bfloat16>(fcOp.getOperand(1),
      "quant", *new_filter, filterShape, "BF16", wTF);
  if (bias) {
    addWeightTensorAndUpdateWeightOp<bfloat16>(fcOp.getOperand(2),
        "quant", *new_bias, biasShape, "BF16", wTF);
  }
  setOpQuant(op, "BF16");
  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// LeakyRelu Ops quantization method
///
LogicalResult quantizeBf16LeakyReluOps(Operation *op) {
  auto lreluOp = cast<tpu::LeakyReluOp>(op);
  auto builder = Builder(op->getContext());

  float negative_slope = lreluOp.negative_slope().convertToFloat();
  LLVM_DEBUG(llvm::errs() << "  negative_slope: "
                          << std::to_string(negative_slope) << "\n";);
  uint16_t bf16_quant_negative_slope;
  float quant_negative_slope;
  FloatToBFloat16(&negative_slope, &bf16_quant_negative_slope, 1);
  BFloat16ToFloat(&bf16_quant_negative_slope, &quant_negative_slope, 1);
  lreluOp.setAttr("negative_slope", builder.getF32FloatAttr(quant_negative_slope));

  setOpQuant(op, "BF16");
  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// bypass Ops quantization method
///
LogicalResult quantizeBf16BypassOps(Operation *op) {
  setOpQuant(op, "BF16");
  setOpResultType(op, StandardTypes::BF16);

  return success();
}

//===----------------------------------------------------------------------===//
// quantizeBf16 API
//===----------------------------------------------------------------------===//

#define DECLARE_QUANTIZE_BF16_BYPASS_METHOD(OP) \
  LogicalResult OP::quantizeBf16() { \
    llvm::errs() << "quantizeBf16: " << getOperationName() \
                 << " [" << getOpName() << "]\n"; \
    Operation *op = this->getOperation(); \
    return quantizeBf16BypassOps(op); \
  }

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastMulOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ConcatOp)

LogicalResult tpu::Conv2DOp::quantizeBf16() {
  llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv2DOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CropOp)

LogicalResult tpu::DeConv2DOp::quantizeBf16() {
  llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::DeConv2DOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseAddOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseMaxOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseMulOp)

LogicalResult tpu::FullyConnectedOp::quantizeBf16() {
  llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeBf16FullyConnectedOps(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InputOp)

LogicalResult tpu::LeakyReluOp::quantizeBf16() {
  llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeBf16LeakyReluOps(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PermuteOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PixelShuffleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolAvg2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PowerOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReciprocalOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SigmoidOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SliceOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SqrtOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TanHOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::UpsampleOp)

#define DECLARE_QUANTIZE_BF16_DISABLED_METHOD(OP) \
  LogicalResult OP::quantizeBf16() { \
    llvm::errs() << "quantizeBf16: " << getOperationName() \
                 << " [" << getOpName() << ", disabled]\n"; \
    assert(false); \
    return failure(); \
  }
/// This Ops does not support quantizie
/// their quant interface are kept for holding threshold only
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::ReshapeOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::SoftmaxOp)

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

template<typename OpTy>
struct QuantizeBf16Pattern : public RewritePattern {
  QuantizeBf16Pattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs() << " < " << getOpName(op)
                              << ", quantized already\n";);
      return matchFailure();
    }
    auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op);
    if (!quantOp) {
      assert(false);
      return matchFailure();
    }
    auto ret = quantOp.quantizeBf16();
    if (failed(ret)) {
      assert(false);
      return matchFailure();
    }
    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddBf16QuantOpBeforeOpPattern : public RewritePattern {
  TpuAddBf16QuantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getOperand(0)->getDefiningOp()
        && isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    auto type = op->getResult(0)->getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("from",
        rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("to",
        rewriter.getStringAttr("BF16")));
    attrs.push_back(rewriter.getNamedAttr("name",
        rewriter.getStringAttr(getOpName(op).str() + "_quant")));
    attrs.push_back(rewriter.getNamedAttr("layer_id",
        rewriter.getI32IntegerAttr(getOpLayerId(op))));
    auto quantOp = rewriter.create<tpu::QuantOp>(op->getLoc(), type,
        ArrayRef<Value *>{op->getOperand(0)}, ArrayRef<NamedAttribute>{attrs});

    op->setOperand(0, quantOp.getResult());

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddBf16DequantOpBeforeOpPattern : public RewritePattern {
  TpuAddBf16DequantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto prev_op = op->getOperand(i)->getDefiningOp();
      if (getOpQuant(prev_op) != "BF16") {
        continue;
      }
      auto type = op->getOperand(i)->getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("from",
          rewriter.getStringAttr("BF16")));
      attrs.push_back(rewriter.getNamedAttr("to",
          rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("threshold",
          rewriter.getF32FloatAttr(getOpThreshold(prev_op))));
      attrs.push_back(rewriter.getNamedAttr("name",
          rewriter.getStringAttr(getOpName(prev_op).str() + "_dequant")));
      attrs.push_back(rewriter.getNamedAttr("layer_id",
          rewriter.getI32IntegerAttr(getOpLayerId(prev_op))));
      auto quantOp = rewriter.create<tpu::QuantOp>(prev_op->getLoc(), type,
          ArrayRef<Value *>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
      setOpResultType(quantOp.getOperation(), StandardTypes::F32);
      op->setOperand(i, quantOp.getResult());
    }

    return matchSuccess();
  }
};

class QuantizeBf16Pass : public FunctionPass<QuantizeBf16Pass> {
public:
  explicit QuantizeBf16Pass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();

#if 0
    OwningRewritePatternList patterns_q;
    patterns.insert<
        QuantizeBf16Pattern<tpu::BroadcastMulOp>,
        QuantizeBf16Pattern<tpu::ConcatOp>,
        QuantizeBf16Pattern<tpu::Conv2DOp>,
        QuantizeBf16Pattern<tpu::CropOp>,
        QuantizeBf16Pattern<tpu::DeConv2DOp>,
        QuantizeBf16Pattern<tpu::EltwiseAddOp>,
        QuantizeBf16Pattern<tpu::EltwiseMaxOp>,
        QuantizeBf16Pattern<tpu::EltwiseMulOp>,
        QuantizeBf16Pattern<tpu::FullyConnectedOp>,
        QuantizeBf16Pattern<tpu::InputOp>,
        QuantizeBf16Pattern<tpu::LeakyReluOp>,
        QuantizeBf16Pattern<tpu::PermuteOp>,
        QuantizeBf16Pattern<tpu::PixelShuffleOp>,
        QuantizeBf16Pattern<tpu::PoolAvg2DOp>,
        QuantizeBf16Pattern<tpu::PoolMax2DOp>,
        QuantizeBf16Pattern<tpu::PReluOp>,
        QuantizeBf16Pattern<tpu::ReciprocalOp>,
        QuantizeBf16Pattern<tpu::ReluOp>,
        QuantizeBf16Pattern<tpu::ShuffleChannelOp>,
        QuantizeBf16Pattern<tpu::SigmoidOp>,
        QuantizeBf16Pattern<tpu::SliceOp>,
        QuantizeBf16Pattern<tpu::SqrtOp>,
        QuantizeBf16Pattern<tpu::UpsampleOp>
        >(context);
    applyPatternsGreedily(fn, patterns_q);
#endif

    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else if (isa<tpu::ReshapeOp>(op)
                 || isa<tpu::SoftmaxOp>(op)) {
        // no need to quant
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        auto ret = quantOp.quantizeBf16();
        if (failed(ret)) {
          assert(false);
        }
      } else if (isa<tpu::DetectionOutputOp>(op)
                 || isa<tpu::PriorBoxOp>(op)) {
        // cpu Ops that has no quant support
      } else {
        llvm::errs() << "lower didn't handle " << op->getName() << "\n";
        assert(false);
      }
    });

    OwningRewritePatternList patterns;
    patterns.insert<
        TpuAddBf16QuantOpBeforeOpPattern<tpu::InputOp>,
        TpuAddBf16DequantOpBeforeOpPattern<tpu::DetectionOutputOp>,
        TpuAddBf16DequantOpBeforeOpPattern<tpu::SoftmaxOp>,
        TpuAddBf16DequantOpBeforeOpPattern<ReturnOp>
        >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace mlir

std::unique_ptr<OpPassBase<FuncOp>> mlir::createQuantizeBf16Pass() {
  return std::make_unique<QuantizeBf16Pass>();
}

static PassRegistration<QuantizeBf16Pass>
    pass("quant-bf16",
         "Quantization to bf16");
