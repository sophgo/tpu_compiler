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
#include "tpuc/Interpreter/cpu/lut_func.hpp"
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
  BF16(data->data(), data->data(), size);
  addWeightTensorAndUpdateWeightOp<float>(op, "quant", *data, shape, "BF16",
                                          wTF);
}

static void quantizeBf16LayerNormWeightOp(Value op, TensorFile *wTF) {
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
  int npu_num = MInfo::lane_num;
  BF16(data->data(), data->data(), size);
  std::vector<float> data_fp32(size * npu_num);
  for (int i = 0; i < npu_num; i++) {
    std::copy(data->begin(), data->end(), data_fp32.data() + i * size);
  }
  std::vector<int64_t> shape_fp32;
  if (shape.size() == 2) {
    shape_fp32 = {1, npu_num, shape[0], shape[1]};
  } else {
    shape_fp32 = {1, npu_num, size};
  }
  addWeightTensorAndUpdateWeightOp<float>(op, "quant", data_fp32, shape_fp32,
                                          "BF16", wTF);
}

//
// lut
//

// insert bf16 table to operands
static void insertBf16LutOp(Operation *op, const std::string &type_name, const std::string &method,
                                         int tableIndex, int mantissaIndex, float param0 = 0.0, float param1 = 0.0) {
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  int table_shape = npu_num * table_hw;

  std::vector<float> table(table_hw);
  std::vector<float> mantissa(table_hw);
  std::vector<float> table_all_lane(table_shape);
  std::vector<float> mantissa_all_lane(table_shape);
  float range_start = -62;
  float range_end = 63;

  std::string suffix_mantissa;
  if (method == "mantissa") {
    bf16_gen_exponent_mantissa_table(type_name, table.data(), mantissa.data(),
                                     param0, param1);
    suffix_mantissa = type_name + "_mantissa_table";
  } else {
    bf16_gen_base_slope_table(type_name, table.data(), mantissa.data(),
                              range_start, range_end, param0, param1);
    suffix_mantissa = type_name + "_slope_table";
  }
  for (int i = 0; i < npu_num; i++) {
    std::copy(table.data(), table.data() + table_hw,
              table_all_lane.data() + i * table_hw);
    std::copy(mantissa.data(), mantissa.data() + table_hw,
              mantissa_all_lane.data() + i * table_hw);
  }

  // update op
  StringRef storageType = "BF16";
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto table_op = addWeightTensorAndCreateWeightOp<float>(
      op, type_name + "_table", table_all_lane, shape, storageType, wTF, wfV);
  auto table_mantissa_op = addWeightTensorAndCreateWeightOp<float>(
      op, suffix_mantissa, mantissa_all_lane, shape, storageType, wTF, wfV);

  auto builder = Builder(op->getContext());
  op->setAttr("min_range", builder.getF32FloatAttr(range_start));
  op->setAttr("max_range", builder.getF32FloatAttr(range_end));
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

  // quantization
  BF16(filter->data(), filter->data(), filterSize);

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

  if (isa<tpu::SigmoidOp>(op)) {
    auto sigmoidOp = cast<tpu::SigmoidOp>(op);
    float scale = sigmoidOp.scale().convertToFloat();
    float bias = sigmoidOp.bias().convertToFloat();
    insertBf16LutOp(op, "sigmoid", "slope", 1, 2, scale, bias);
  } else if (isa<tpu::SwishOp>(op)) {
    insertBf16LutOp(op, "swish", "slope", 1, 2);
  } else if (isa<tpu::TanHOp>(op)) {
    insertBf16LutOp(op, "tanh", "slope", 1, 2);
  } else if (isa<tpu::LogOp>(op)) {
    insertBf16LutOp(op, "log", "slope", 1, 2);
  } else if (isa<tpu::ExpOp>(op)) {
    auto expOp = cast<tpu::ExpOp>(op);
    float scale = expOp.scale().convertToFloat();
    float bias = expOp.bias().convertToFloat();
    insertBf16LutOp(op, "exp", "slope", 1, 2, scale, bias);
  } else if (isa<tpu::EluOp>(op)) {
    insertBf16LutOp(op, "elu", "slope", 1, 2);
  } else if (isa<tpu::MishOp>(op)) {
    insertBf16LutOp(op, "mish", "slope", 1, 2);
  } else if (isa<tpu::SoftPlusOp>(op)) {
    auto spOp = cast<tpu::SoftPlusOp>(op);
    float scale = spOp.scale().convertToFloat();
    float bias = spOp.bias().convertToFloat();
    insertBf16LutOp(op, "softplus", "slope", 1, 2, scale, bias);
  } else {
    llvm_unreachable("not support now");
  }

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
  negative_slope = BF16(negative_slope);
  lreluOp->setAttr("negative_slope", builder.getF32FloatAttr(negative_slope));

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
  insertBf16LutOp(op, "sigmoid", "slope", 4, 5, 1.0, 0.0);
  insertBf16LutOp(op, "tanh", "slope", 6, 7);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::InterpOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto interpOp = cast<tpu::InterpOp>(op);
  llvm::StringRef type = "NONE";
  if (interpOp.coordinate_transformation_mode().startswith("nearest")) {
    type = "BF16";
    setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  }
  interpOp.setOpQuantMode(type);
  return success();
}

LogicalResult tpu::LayerNormOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  quantizeBf16LayerNormWeightOp(scale(), wTF);
  quantizeBf16LayerNormWeightOp(bias(), wTF);
  insertBf16LutOp(op, "reciprocal_sqrt", "mantissa", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::StdOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "sqrt", "mantissa", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::LeakyReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LeakyReluOps(op);
}

LogicalResult tpu::LogOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::LogOp>(op);
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
  insertBf16LutOp(op, "power", "mantissa", 1, 2, beta);
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
  quantizeBf16WeightOp(cont(), wTF);
  insertBf16LutOp(op, "sigmoid", "slope", 6, 7, 1.0, 0.0);
  insertBf16LutOp(op, "tanh", "slope", 8, 9);
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
  insertBf16LutOp(op, "sqrt", "mantissa", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::ReciprocalOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  insertBf16LutOp(op, "reciprocal", "mantissa", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SigmoidOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SigmoidOp>(op);
}

LogicalResult tpu::SwishOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SwishOp>(op);
}

LogicalResult tpu::TanHOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::TanHOp>(op);
}

LogicalResult tpu::EluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::EluOp>(op);
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
  insertBf16LutOp(op, "exp", "slope", 1, 2, 1.0, 0.0);
  insertBf16LutOp(op, "reciprocal", "mantissa", 3, 4);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SoftPlusOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SoftPlusOp>(op);
}

LogicalResult tpu::ReflectionPadOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  std::vector<int> pads;
  arrayAttrToVector(this->pads(), pads);
  int pad_idx = 0;
  for (int K : pads) {
    std::vector<float> select(K * K, 0.0f);
    for (int i = 0; i < K; i++) {
      int last = K - i - 1;
      select[i * K + last] = 1.0f;
    }
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    auto shape = std::vector<int64_t>{K, K};
    auto select_op = addWeightTensorAndCreateWeightOp<float>(
        op, "_select_" + std::to_string(pad_idx), select, shape, "BF16", wTF,
        wfV);
    op->setOperand(pad_idx + 1, select_op);
    pad_idx++;
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::WhereOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  assert(getOpQuant() == "BF16");
  assert (isTensorNone(x()) && "only support x as none type(masked_fill case)");

  // mask saved as 0/1 that not convert it
  condition().getDefiningOp()->setAttr("storage", builder.getStringAttr("BF16"));

  op->setAttr("fill_constant",
          builder.getF32FloatAttr(BF16(fill_constant().convertToFloat())));
  return success();
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
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMinOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ScaleLutOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SliceOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SquareOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileOp)
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
