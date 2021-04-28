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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/Passes.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <math.h>
#include <sstream>

#define DEBUG_TYPE "quantize_bf16"

float BF16_TABLE_START = -8.0f;
float BF16_TABLE_END = 8.0f;
static float revert_threshold;
static double sigmoid(double x) { return 0.5 * tanh(0.5 * x) + 0.5; }
static double mish(double x) { return my_mish_caffe(x, revert_threshold); }
static double softplus(double x) {
  return softplus_activate(x, revert_threshold);
}

using namespace mlir;

namespace mlir {

///
/// bypass Ops quantization method
///
LogicalResult quantizeBf16BypassOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

//
// weight fp32->bf16->fp32
//
static void quantizeBf16WeightOp(Value op, TensorFile *wTF) {
  if (isTensorNone(op)) {
    return;
  }
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op.getDefiningOp());
  if (weightOp == nullptr) {
    return;
  }
  auto data = readAndDeleteWeightTensor<float>(op, wTF);
  auto shape = getTensorShape(op);
  auto size = getTensorSize(op);
  std::vector<bfloat16> data_bf16(size);
  FloatToBFloat16(data->data(), data_bf16.data(), size);
  BFloat16ToFloat(data_bf16.data(), data->data(), size);
  addWeightTensorAndUpdateWeightOp<float>(op, "quant", *data, shape, "BF16",
                                          wTF);
}

//
// lut
//
typedef enum lut_type {
  // mantissa
  RECIPROCAL = 0,
  SQRT,
  RECIPROCAL_SQRT,
  POWER,
  // slope
  SIGMOID = 1024,
  TANH,
  EXP,
  MISH,
  SOFTPLUS,
} lut_type_t;

// insert bf16 table to operands
static void insertBf16LutOp(Operation *op, int tableIndex, int mantissaIndex,
                            lut_type_t type, float param = 0.0f) {
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;

  bool is_slope = (type >= 1024); // slope or mantissa
  std::vector<float> table_fp32(tbl_shape);
  std::vector<float> mantissa_fp32(tbl_shape);
  std::vector<bfloat16> table_bf16(table_hw);
  std::vector<bfloat16> mantissa_bf16(table_hw);

  const int start = is_slope ? BF16_TABLE_START : -62;
  const int end = is_slope ? BF16_TABLE_END : 63;
  std::string type_name;
  switch (type) {
  case RECIPROCAL:
    type_name = "reciprocal";
    bf16_gen_reciprocal(start, end, table_hw, table_bf16.data());
    bf16_gen_reciprocal_mantissa(start, end, table_hw, mantissa_bf16.data());
    break;
  case SQRT:
    type_name = "sqrt";
    bf16_gen_sqrt(start, table_hw, table_bf16.data());
    bf16_gen_sqrt_mantissa(table_hw, mantissa_bf16.data());
    break;
  case RECIPROCAL_SQRT:
    type_name = "reciprocal_sqrt";
    bf16_gen_reciprocal_sqrt(start, table_hw, table_bf16.data());
    bf16_gen_reciprocal_sqrt_mantissa(table_hw, mantissa_bf16.data());
    break;
  case POWER:
    type_name = "power";
    bf16_gen_power_exp_table(table_bf16.data(), param, start, table_hw);
    bf16_gen_power_mantissa_table(mantissa_bf16.data(), param, table_hw);
    break;
  case SIGMOID:
    type_name = "sigmoid";
    gen_bf16_table(start, end, table_hw, table_fp32.data(), sigmoid);
    gen_bf16_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                         table_fp32.data(), mantissa_fp32.data(), sigmoid);
    break;
  case TANH:
    type_name = "tanh";
    gen_bf16_table(start, end, table_hw, table_fp32.data(), tanh);
    gen_bf16_slope_table(start, end, table_hw, table_fp32.data(),
                         mantissa_fp32.data(), tanh);
    break;
  case EXP:
    type_name = "exp";
    gen_bf16_table(start, end, table_hw, table_fp32.data(), exp);
    gen_bf16_slope_table(start, end, table_hw, table_fp32.data(),
                         mantissa_fp32.data(), exp);
    if (start <= -15) {
      table_fp32[128] = 0; // Make lut exp(x) = 0 when x <= START
    }
    break;
  case MISH:
    type_name = "miss";
    gen_bf16_table(start, end, table_hw, table_fp32.data(), mish);
    gen_bf16_slope_table(start, end, table_hw, table_fp32.data(),
                         mantissa_fp32.data(), mish);
    break;
  case SOFTPLUS:
    type_name = "softplus";
    gen_bf16_table(start, end, table_hw, table_fp32.data(), softplus);
    gen_bf16_slope_table(start, end, table_hw, table_fp32.data(),
                         mantissa_fp32.data(), softplus);
    break;
  }
  std::string suffix_mantissa;
  if (is_slope) {
    suffix_mantissa = type_name + "_slope_table";
    FloatToBFloat16(table_fp32.data(), table_bf16.data(), table_hw);
    FloatToBFloat16(mantissa_fp32.data(), mantissa_bf16.data(), table_hw);
  } else {
    suffix_mantissa = type_name + "_mantissa_table";
  }
  BFloat16ToFloat(table_bf16.data(), table_fp32.data(), table_hw);
  BFloat16ToFloat(mantissa_bf16.data(), mantissa_fp32.data(), table_hw);
  for (int i = 1; i < npu_num; i++) {
    std::copy(table_fp32.data(), table_fp32.data() + table_hw,
              table_fp32.data() + i * table_hw);
    std::copy(mantissa_fp32.data(), mantissa_fp32.data() + table_hw,
              mantissa_fp32.data() + i * table_hw);
  }

  // update op
  StringRef storageType = "BF16";
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto table_op = addWeightTensorAndCreateWeightOp<float>(
      op, type_name + "_table", table_fp32, shape, storageType, wTF, wfV);
  auto table_mantissa_op = addWeightTensorAndCreateWeightOp<float>(
      op, suffix_mantissa, mantissa_fp32, shape, storageType, wTF, wfV);

  op->setOperand(tableIndex, table_op);
  op->setOperand(mantissaIndex, table_mantissa_op);
}

///
/// Conv Ops quantization method
///
template <typename OpTy>
LogicalResult quantizeBf16ConvOps(Operation *op, int spatial_dims) {
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
  } else if (filterShape.size() == 5 && spatial_dims == 2) {
    // g, oc/g, ic/g, kh, kw
    oc = filterShape[0] * filterShape[1];
  } else if (filterShape.size() == 5 && spatial_dims == 3) {
    // oc, ic, kd, kh, kw
    oc = filterShape[0];
  } else {
    assert(0);
  }
  assert(filterSize % oc == 0);

  // create new tensors
  auto filter_bf16 = std::make_unique<std::vector<bfloat16>>(filterSize);

  // quantization
  FloatToBFloat16(filter->data(), filter_bf16->data(), filterSize);
  BFloat16ToFloat(filter_bf16->data(), filter->data(), filterSize);

  // update op
  addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1), "quant",
                                          *filter, filterShape, "BF16", wTF);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

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

  auto lutOp = cast<OpTy>(op);
  auto _start = BF16_TABLE_START;
  auto _end = BF16_TABLE_END;

  BF16_TABLE_START = lutOp.min_range().convertToFloat();
  BF16_TABLE_END = lutOp.max_range().convertToFloat();

  if (OpTy::getOperationName() == "tpu.sigmoid") {
    insertBf16LutOp(op, 1, 2, SIGMOID);
  } else if (OpTy::getOperationName() == "tpu.tanh") {
    insertBf16LutOp(op, 1, 2, TANH);
  } else if (OpTy::getOperationName() == "tpu.exp") {
    insertBf16LutOp(op, 1, 2, EXP);
  } else if (OpTy::getOperationName() == "tpu.mish") {
    auto castOp = dyn_cast<tpu::MishOp>(op);
    revert_threshold = castOp.mish_threshold().convertToFloat();
    insertBf16LutOp(op, 1, 2, MISH);
  } else if (OpTy::getOperationName() == "tpu.softplus") {
    auto castOp = dyn_cast<tpu::SoftPlusOp>(op);
    revert_threshold = castOp.threshold().convertToFloat();
    insertBf16LutOp(op, 1, 2, SOFTPLUS);
  } else {
    llvm_unreachable("not support now");
  }
  BF16_TABLE_START = _start;
  BF16_TABLE_END = _end;
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

//
// Eltwise Const
//
template <typename OpTy>
LogicalResult quantizeBf16EltwiseOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  uint32_t num_input = op->getNumOperands();
  for (uint32_t i = 0; i < num_input; i++) {
    quantizeBf16WeightOp(op->getOperand(i), wTF);
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
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
  lreluOp->setAttr("negative_slope",
                   builder.getF32FloatAttr(quant_negative_slope));

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// quantizeBf16 API
//===----------------------------------------------------------------------===//

LogicalResult tpu::ArgMaxOp::quantizeBf16() {
  Operation *op = this->getOperation();
  setOpResultType(op->getResult(0), FloatType::getF32(getContext()));
  return success();
}

LogicalResult tpu::Conv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv2DOp>(op, 2);
}

LogicalResult tpu::DeConv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::DeConv2DOp>(op, 2);
}

LogicalResult tpu::Conv3DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv3DOp>(op, 3);
}

LogicalResult tpu::EmbeddingOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(table(), wTF);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::EltwiseAddOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16EltwiseOps<tpu::EltwiseAddOp>(op);
}

LogicalResult tpu::EltwiseMaxOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16EltwiseOps<tpu::EltwiseMaxOp>(op);
}

LogicalResult tpu::EltwiseMinOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16EltwiseOps<tpu::EltwiseMinOp>(op);
}

LogicalResult tpu::EltwiseMulOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16EltwiseOps<tpu::EltwiseMulOp>(op);
}

LogicalResult tpu::FullyConnectedOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(filter(), wTF);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::GruOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(recurrence(), wTF);
  quantizeBf16WeightOp(initial_h(), wTF);
  insertBf16LutOp(op, 4, 5, SIGMOID);
  insertBf16LutOp(op, 6, 7, TANH);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::InterpOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto interpOp = cast<tpu::InterpOp>(op);
  llvm::StringRef type = "NONE";
  interpOp.setOpQuantMode(type);
  return success();
}

LogicalResult tpu::LayerNormOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(scale(), wTF);
  quantizeBf16WeightOp(bias(), wTF);
  insertBf16LutOp(op, 3, 4, RECIPROCAL_SQRT);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::LeakyReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LeakyReluOps(op);
}

LogicalResult tpu::PReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n");
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(filter(), wTF);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::LrnOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n");
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  auto lrnOp = cast<tpu::LrnOp>(op);
  float beta = lrnOp.beta().convertToFloat();
  insertBf16LutOp(op, 1, 2, POWER, beta);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::LstmOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16WeightOp(recurrence(), wTF);
  quantizeBf16WeightOp(initial_h(), wTF);
  quantizeBf16WeightOp(initial_c(), wTF);
  insertBf16LutOp(op, 5, 6, SIGMOID);
  insertBf16LutOp(op, 7, 8, TANH);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::MishOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::MishOp>(op);
}

LogicalResult tpu::QuadraticSumOp::quantizeBf16() {
  Operation *op = this->getOperation();
  // for high precision
  if (high_precision()) {
    setOpResultType(op->getResult(0), FloatType::getF32(getContext()));
  }
  return success();
}

LogicalResult tpu::SqrtOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  insertBf16LutOp(op, 1, 2, SQRT);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::ReciprocalOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  insertBf16LutOp(op, 1, 2, RECIPROCAL);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SigmoidOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SigmoidOp>(op);
}

LogicalResult tpu::TanHOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::TanHOp>(op);
}

LogicalResult tpu::ExpOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::ExpOp>(op);
}

LogicalResult tpu::SoftmaxOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  BF16_TABLE_START = -15;
  BF16_TABLE_END = 1;
  insertBf16LutOp(op, 1, 2, EXP);
  insertBf16LutOp(op, 3, 4, RECIPROCAL);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SoftPlusOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SoftPlusOp>(op);
}

//
// quantization bypass
//
#define DECLARE_QUANTIZE_BF16_BYPASS_METHOD(OP)                                \
  LogicalResult OP::quantizeBf16() {                                           \
    LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["  \
                            << getOpName() << "]\n";);                         \
    Operation *op = this->getOperation();                                      \
    return quantizeBf16BypassOps(op);                                          \
  }

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::AbsOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastMulOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastAddOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastSubOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ConcatOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ClipOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CscOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CustomOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::DilateOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InputOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InstanceNormOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::MatMulOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PadOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PermuteOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PixelShuffleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolAvg2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax3DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMaskOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PowerOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceL2Op)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMeanOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMaxOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ScaleLutOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SliceOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SquareOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileInterpOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::UpsampleOp)

//
// quantization disabled
//
#define DECLARE_QUANTIZE_BF16_DISABLED_METHOD(OP)                              \
  LogicalResult OP::quantizeBf16() {                                           \
    LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["  \
                            << getOpName() << ", disabled]\n";);               \
    assert(false);                                                             \
    return failure();                                                          \
  }
/// This Ops does not support quantizie
/// their quant interface are kept for holding threshold only
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::ReshapeOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnOneOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnTwoOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnThreeOp)
} // namespace mlir
