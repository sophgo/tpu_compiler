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
  assert(getOpQuant(op) == "BF16");

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

  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// FC Ops quantization method
///
LogicalResult quantizeBf16FullyConnectedOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

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

  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// LeakyRelu Ops quantization method
///
LogicalResult quantizeBf16LeakyReluOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

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

  setOpResultType(op, StandardTypes::BF16);

  return success();
}

///
/// bypass Ops quantization method
///
LogicalResult quantizeBf16BypassOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

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
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
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

} // namespace mlir
