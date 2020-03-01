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

namespace {

template<typename OpTy>
struct TpuQuantBf16Conv2DOpPattern : public RewritePattern {
  TpuQuantBf16Conv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs() << getOpName(op) << " quantized already\n";);
      return matchFailure();
    }
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

    return matchSuccess();
  }
};

struct TpuQuantBf16FullyConnectedOpPattern : public RewritePattern {
  TpuQuantBf16FullyConnectedOpPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs() << getOpName(op) << " quantized already\n";);
      return matchFailure();
    }
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

    return matchSuccess();
  }
};

struct TpuQuantBf16LeakyReluOpOpPattern : public RewritePattern {
  TpuQuantBf16LeakyReluOpOpPattern(MLIRContext *context)
      : RewritePattern("tpu.leaky_relu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lreluOp = cast<tpu::LeakyReluOp>(op);
    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs() << getOpName(op) << " quantized already\n";);
      return matchFailure();
    }
    setOpQuant(op, "BF16");
    float negative_slope = lreluOp.negative_slope().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  negative_slope: "
                            << std::to_string(negative_slope) << "\n";);
    uint16_t bf16_quant_negative_slope;
    float quant_negative_slope;
    FloatToBFloat16(&negative_slope, &bf16_quant_negative_slope, 1);
    BFloat16ToFloat(&bf16_quant_negative_slope, &quant_negative_slope, 1);
    lreluOp.setAttr("negative_slope", rewriter.getF32FloatAttr(quant_negative_slope));

    return matchSuccess();
  }
};

// default quantize pattern, for no weight operations
template<typename OpTy>
struct TpuQuantBf16DefaultPattern : public RewritePattern {
  TpuQuantBf16DefaultPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (getOpQuant(op) != "NONE") {
      LLVM_DEBUG(llvm::errs() << getOpName(op) << " quantized already\n";);
      return matchFailure();
    }
    setOpQuant(op, "BF16");

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddQuantizeOpBeforeOpPattern : public RewritePattern {
  TpuAddQuantizeOpBeforeOpPattern(MLIRContext *context)
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
struct TpuAddDequantizeOpBeforeOpPattern : public RewritePattern {
  TpuAddDequantizeOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    for (auto i = 0; i < op->getNumOperands(); i++) {
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

    OwningRewritePatternList patterns;
    patterns.insert<
        TpuQuantBf16DefaultPattern<tpu::BroadcastMulOp>,
        TpuQuantBf16DefaultPattern<tpu::ConcatOp>,
        TpuQuantBf16Conv2DOpPattern<tpu::Conv2DOp>,
        TpuQuantBf16DefaultPattern<tpu::CropOp>,
        TpuQuantBf16Conv2DOpPattern<tpu::DeConv2DOp>,
        TpuQuantBf16DefaultPattern<tpu::DivOp>,
        TpuQuantBf16DefaultPattern<tpu::EltwiseAddOp>,
        TpuQuantBf16DefaultPattern<tpu::EltwiseMaxOp>,
        TpuQuantBf16DefaultPattern<tpu::EltwiseMulOp>,
        TpuQuantBf16FullyConnectedOpPattern,
        TpuQuantBf16DefaultPattern<tpu::InputOp>,
        TpuQuantBf16LeakyReluOpOpPattern,
        TpuQuantBf16DefaultPattern<tpu::PermuteOp>,
        TpuQuantBf16DefaultPattern<tpu::PoolAvg2DOp>,
        TpuQuantBf16DefaultPattern<tpu::PoolMax2DOp>,
        TpuQuantBf16DefaultPattern<tpu::PReluOp>,
        TpuQuantBf16DefaultPattern<tpu::ReluOp>,
        TpuQuantBf16DefaultPattern<tpu::ShuffleChannelOp>,
        TpuQuantBf16DefaultPattern<tpu::SigmoidOp>,
        TpuQuantBf16DefaultPattern<tpu::SliceOp>,
        TpuQuantBf16DefaultPattern<tpu::SqrtOp>,
        TpuQuantBf16DefaultPattern<tpu::UpsampleOp>
        >(context);
    applyPatternsGreedily(fn, patterns);

    patterns.clear();
    patterns.insert<
        TpuAddQuantizeOpBeforeOpPattern<tpu::InputOp>,
        TpuAddDequantizeOpBeforeOpPattern<tpu::DetectionOutputOp>,
        TpuAddDequantizeOpBeforeOpPattern<tpu::SoftmaxOp>,
        TpuAddDequantizeOpBeforeOpPattern<ReturnOp>
        >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createQuantizeBf16Pass() {
  return std::make_unique<QuantizeBf16Pass>();
}

static PassRegistration<QuantizeBf16Pass>
    pass("quant-bf16",
         "Quantization to bf16");
