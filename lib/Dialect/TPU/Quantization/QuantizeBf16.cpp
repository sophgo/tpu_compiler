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
#include "mlir/Dialect/TPU/NativeCpuImplementation.h"
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

extern const int BF16_TABLE_START = -8;
extern const int BF16_TABLE_END = 8;


double sigmoid(double x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

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
/// Lut Ops quantization method
///
template <typename OpTy>
LogicalResult quantizeBF16LutOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use LUT
  setOpQuantParamType(op, "LUT_BF16");

  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);
  auto lutOp = cast<OpTy>(op);

  // quantization
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                          << ", threshold_y = " << std::to_string(threshold_y)
                          << ", threshold_x = " << std::to_string(threshold_x)
                          << "\n";);
  int npu_num = 32; //<! 1880v2 hardcode

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  std::vector<float> y0_table;
  std::vector<float> y0_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_table.resize(tbl_shape);
  y0_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);

  // use function pointer
  double (*activate_func)(double);
  if (OpTy::getOperationName() == "tpu.sigmoid") {
    activate_func = sigmoid;
  }

  gen_bf16_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                 y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                       y0_fp32_table.data(), y0_fp32_slope_table.data(),
                       activate_func);

  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(),
                  y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(),
                  y0_bf16_slope_table.data(), table_hw);

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i){
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(),
              y0_bf16_slope_table.data() + table_hw,
              y0_slope_table.data() + i * table_hw);
  }

  // update op
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  StringRef storageType = "BF16";
  auto y0_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "y0_table", y0_table, shape, storageType, wTF, wfV);
  auto mantissa_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "mantissa_table", y0_slope_table, shape, storageType, wTF, wfV);
  lutOp.setOperand(1, y0_table_op);
  lutOp.setOperand(2, mantissa_table_op);

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
/// Gru Ops quantization method
///
LogicalResult quantizeBf16GruOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

  auto gruOp = cast<tpu::GruOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);

  // get weight tensor
  auto weight = readAndDeleteWeightTensor<float>(gruOp.weight(), wTF);
  std::vector<int64_t> weightShape;
  int64_t weightSize;
  getTensorShapeAndSize(gruOp.weight(), weightShape, weightSize);

  // get recurrence tensor
  auto recurrence = readAndDeleteWeightTensor<float>(gruOp.recurrence(), wTF);
  std::vector<int64_t> recurrenceShape;
  int64_t recurrenceSize;
  getTensorShapeAndSize(gruOp.recurrence(), recurrenceShape, recurrenceSize);

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(gruOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(gruOp.bias(), wTF);
    getTensorShapeAndSize(gruOp.bias(), biasShape, biasSize);
  }

  // get initial_h tensor
  std::unique_ptr<std::vector<float> > initial_h = nullptr;
  std::vector<int64_t> initial_hShape;
  int64_t initial_hSize = 0;
  if ( !isTensorNone(gruOp.initial_h()) ) {
    initial_h = readAndDeleteWeightTensor<float>(gruOp.initial_h(), wTF);
    getTensorShapeAndSize(gruOp.initial_h(), initial_hShape, initial_hSize);
  }

  // create new tensors
  auto new_weight = std::make_unique<std::vector<bfloat16> >(weightSize);
  auto new_recurrence = std::make_unique<std::vector<bfloat16> >(recurrenceSize);
  std::unique_ptr<std::vector<bfloat16> > new_bias = nullptr;
  std::unique_ptr<std::vector<bfloat16> > new_initial_h = nullptr;

  if (bias) {
    new_bias = std::make_unique<std::vector<bfloat16> >(biasSize);
  }

  if (initial_h) {
    new_initial_h = std::make_unique<std::vector<bfloat16> >(initial_hSize);
  }

  // quantization
  FloatToBFloat16(weight->data(), new_weight->data(), weightSize);
  FloatToBFloat16(recurrence->data(), new_recurrence->data(), recurrenceSize);

  if (bias) {
    FloatToBFloat16(bias->data(), new_bias->data(), biasSize);
  }

  if (initial_h) {
    FloatToBFloat16(initial_h->data(), new_initial_h->data(), initial_hSize);
  }

  // update op
  addWeightTensorAndUpdateWeightOp<bfloat16>(gruOp.getOperand(1),
      "quant", *new_weight, weightShape, "BF16", wTF);
  addWeightTensorAndUpdateWeightOp<bfloat16>(gruOp.getOperand(2),
      "quant", *new_recurrence, recurrenceShape, "BF16", wTF);

  if (bias) {
    addWeightTensorAndUpdateWeightOp<bfloat16>(gruOp.getOperand(3),
        "quant", *new_bias, biasShape, "BF16", wTF);
  }

  if (initial_h) {
    addWeightTensorAndUpdateWeightOp<bfloat16>(gruOp.getOperand(4),
        "quant", *new_initial_h, initial_hShape, "BF16", wTF);
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
    LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() \
                 << " [" << getOpName() << "]\n";); \
    Operation *op = this->getOperation(); \
    return quantizeBf16BypassOps(op); \
  }

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastMulOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ConcatOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ClipOp)

LogicalResult tpu::Conv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv2DOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CustomOp)

LogicalResult tpu::DeConv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::DeConv2DOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseAddOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseMaxOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseMinOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::EltwiseMulOp)

LogicalResult tpu::FullyConnectedOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16FullyConnectedOps(op);
}

LogicalResult tpu::GruOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16GruOps(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InputOp)

LogicalResult tpu::LeakyReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LeakyReluOps(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::LrnOneOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::LrnTwoOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::LrnThreeOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::LrnOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PadOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PermuteOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PixelShuffleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolAvg2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PowerOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PreprocessOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReciprocalOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)

LogicalResult tpu::SigmoidOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SigmoidOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SliceOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SqrtOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TanHOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::UpsampleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileInterpOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InterpOp)

#define DECLARE_QUANTIZE_BF16_DISABLED_METHOD(OP) \
  LogicalResult OP::quantizeBf16() { \
    LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() \
                 << " [" << getOpName() << ", disabled]\n";); \
    assert(false); \
    return failure(); \
  }
/// This Ops does not support quantizie
/// their quant interface are kept for holding threshold only
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::ReshapeOp)

} // namespace mlir
