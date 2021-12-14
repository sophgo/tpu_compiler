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

inline static float INT8(float data) {
  data = std::floor(data + 0.5);
  return std::max(std::min(data, 127.0f), -128.0f);
}

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
  if (isa<tpu::LoadWeightOp>(op.getDefiningOp()) == false) {
    return;
  }
  auto data = readAndDeleteWeightTensor<float>(op, wTF);
  auto shape = getTensorShape(op);
  auto size = getTensorSize(op);
  BF16(data->data(), data->data(), size);
  addWeightTensorAndUpdateWeightOp<float>(op, "quant", *data, shape, "BF16",
                                          wTF);
}

static bool check_mix_bf16(Operation *op, Value weight, int axis, int max = 0) {
  if (mlir::getOpQuantParamType(op) != "MIX_BF16") {
    return false;
  }
  if (false == isa<tpu::LoadWeightOp>(weight.getDefiningOp())) {
    return false;
  }
  auto shape = getTensorShape(weight);
  int num_dims = shape.size();
  if (axis < 0) {
    axis += num_dims;
  }
  assert(axis < num_dims && axis >= 0);
  if (max > 0 && shape[axis] > max) {
    return false;
  }
  return true;
}

static void quantizeWeightInt8(const std::vector<float> &weight_fp32,
                               int64_t outer_size, int64_t axis_size,
                               int64_t inner_size,
                               std::vector<float> &weight_int8,
                               std::vector<float> &quant_scale,
                               std::vector<float> &quant_zeropoint) {
  int64_t size = weight_fp32.size();
  assert(size == outer_size * inner_size * axis_size);
  weight_int8.resize(size);
  quant_scale.resize(axis_size);
  quant_zeropoint.resize(axis_size);
  float max, min, scale, zeropoint;
  for (int k = 0; k < axis_size; k++) {
    max = weight_fp32[k * inner_size];
    min = weight_fp32[k * inner_size];
    for (int o = 0; o < outer_size; o++) {
      for (int i = 0; i < inner_size; i++) {
        int index = o * axis_size * inner_size + k * inner_size + i;
        float data = weight_fp32[index];
        if (data > max) {
          max = data;
        }
        if (data < min) {
          min = data;
        }
      }
    }
    if (max == min) {
      for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
          int index = o * axis_size * inner_size + k * inner_size + i;
          weight_int8[index] = INT8(1.0f);
        }
      }
      quant_scale[k] = BF16(max);
      quant_zeropoint[k] = BF16(0.0f);
    } else {
      zeropoint = (max + min) / 2;
      scale = 128.0 / (max - zeropoint);
      for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
          int index = o * axis_size * inner_size + k * inner_size + i;
          weight_int8[index] = INT8((weight_fp32[index] - zeropoint) * scale);
        }
      }
      quant_scale[k] = BF16((max - zeropoint) / 128.0f);
      quant_zeropoint[k] = BF16(zeropoint);
    }
  }
}

static void quantizeWeightInt8Op(Operation *op, Value v, TensorFile *wTF,
                                 int axis, int scale_index,
                                 int zeropoint_index) {
  if (isa<tpu::LoadWeightOp>(v.getDefiningOp()) == false) {
    llvm_unreachable("must be weight op");
  }
  int64_t size;
  std::vector<int64_t> shape;
  getTensorShapeAndSize(v, shape, size);
  int num_dims = shape.size();
  if (axis < 0) {
    axis += num_dims;
  }
  assert(axis < num_dims);
  auto weight = readAndDeleteWeightTensor<float>(v, wTF);
  auto new_weight = std::make_unique<std::vector<float>>();
  auto axis_size = shape[axis];
  auto inner_size = std::accumulate(shape.begin() + axis + 1, shape.end(), 1,
                                    std::multiplies<int64_t>());
  auto outer_size = std::accumulate(shape.begin(), shape.begin() + axis, 1,
                                    std::multiplies<int64_t>());
  auto quant_scale = std::make_unique<std::vector<float>>();
  auto quant_zeropoint = std::make_unique<std::vector<float>>();
  quantizeWeightInt8(*weight, outer_size, axis_size, inner_size, *new_weight,
                     *quant_scale, *quant_zeropoint);
  addWeightTensorAndUpdateWeightOp<float>(v, "quant", *new_weight, shape,
                                          "INT8", wTF);
  Value wfV = getWeightFileValue(op);
  std::vector<int64_t> qshape(1, axis_size);
  auto scale_op = addWeightTensorAndCreateWeightOp<float>(
      op, "quant_scale", *quant_scale, qshape, "BF16", wTF, wfV);
  auto zeropoint_op = addWeightTensorAndCreateWeightOp<float>(
      op, "quant_zeropoint", *quant_zeropoint, qshape, "BF16", wTF, wfV);
  op->setOperand(scale_index, scale_op);
  op->setOperand(zeropoint_index, zeropoint_op);
}

static void quantizeBf16LayerNormWeightOp(Value op, TensorFile *wTF) {
  if (isa<tpu::LoadWeightOp>(op.getDefiningOp()) == false) {
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
static void insertBf16LutOp(Operation *op, const std::string &type_name,
                            int tableIndex, int mantissaIndex,
                            float param0 = 0.0, float param1 = 0.0) {
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
  if (type_name == "pow") {
    // only pow do mantissa
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
  bool is_mix_bf16 = (getOpQuantParamType(op) == "MIX_BF16");

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
  int num_dims = filterShape.size();
  if (num_dims == 4) {
    oc = filterShape[0];
  } else if (num_dims == 5 && spatial_dims == 2) {
    // g, oc/g, ic/g, kh, kw
    oc = filterShape[0] * filterShape[1];
  } else if (num_dims == 5 && spatial_dims == 3) {
    // oc, ic, kd, kh, kw
    oc = filterShape[0];
  } else {
    assert(0);
  }
  assert(filterSize % oc == 0);
  int64_t inner_size = filterSize/oc;
  if (is_mix_bf16) {
    // TODO(charle.hu): not good for conv
    //if (inner_size < 8 || spatial_dims > 2) {
      // no need to do weight int8
      setOpQuantParamType(op, "NONE");
      is_mix_bf16 = false;
    //}
  }

  if (!is_mix_bf16) {
    BF16(filter->data(), filter->data(), filterSize);
    addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1), "quant",
                                            *filter, filterShape, "BF16", wTF);
  } else {
    std::vector<float> weight_int8;
    std::vector<float> quant_scale;
    std::vector<float> quant_zeropoint;
    quantizeWeightInt8(*filter, 1, oc, inner_size,
                       weight_int8, quant_scale, quant_zeropoint);
    Value wfV = getWeightFileValue(op);
    std::vector<int64_t> qshape(1, oc);
    auto scale_op = addWeightTensorAndCreateWeightOp<float>(
        op, "quant_scale", quant_scale, qshape, "BF16", wTF, wfV);
    auto zeropoint_op = addWeightTensorAndCreateWeightOp<float>(
        op, "quant_zeropoint", quant_zeropoint, qshape, "BF16", wTF, wfV);
    addWeightTensorAndUpdateWeightOp<float>(
        convOp.filter(), "quant", weight_int8, filterShape, "INT8", wTF);
    op->setOperand(3, scale_op);
    op->setOperand(4, zeropoint_op);
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

LogicalResult tpu::ConcatOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  for (auto input:inputs()) {
    quantizeBf16WeightOp(input, wTF);
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
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

LogicalResult tpu::ConvFcOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  if (check_mix_bf16(op, filter(), -1)) {
    quantizeWeightInt8Op(op, filter(), wTF, -1, 2, 3);
  } else {
    quantizeBf16WeightOp(filter(), wTF);
    mlir::setOpQuantParamType(op, "NONE");
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::EmbeddingOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  TensorFile *wTF = getWeightTensorFile(op);
  if (check_mix_bf16(op, table(), -1)) {
    quantizeWeightInt8Op(op, table(), wTF, -1, 2, 3);
  } else {
    quantizeBf16WeightOp(table(), wTF);
    mlir::setOpQuantParamType(op, "NONE");
  }
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

  if (check_mix_bf16(op, filter(), -2)) {
    quantizeWeightInt8Op(op, filter(), wTF, -2, 3, 4);
  } else {
    quantizeBf16WeightOp(filter(), wTF);
    mlir::setOpQuantParamType(op, "NONE");
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::BroadcastMulOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  for (auto input: inputs()) {
    quantizeBf16WeightOp(input, wTF);
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::BroadcastAddOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  for (auto input: inputs()) {
    quantizeBf16WeightOp(input, wTF);
  }
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
  insertBf16LutOp(op, "sigmoid", 4, 5, 1.0, 0.0);
  insertBf16LutOp(op, "tanh", 6, 7);
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
  insertBf16LutOp(op, "pow", 1, 2, -0.5f);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::StdOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "pow", 1, 2, 0.5f);
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
  insertBf16LutOp(op, "log", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
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
  insertBf16LutOp(op, "pow", 1, 2, -1 * beta().convertToFloat());
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
  insertBf16LutOp(op, "sigmoid", 6, 7, 1.0, 0.0);
  insertBf16LutOp(op, "tanh", 8, 9);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::MishOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "mish", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::QuadraticSumOp::quantizeBf16() {
  Operation *op = this->getOperation();
  // for high precision
  if (high_precision()) {
    setOpResultType(op->getResult(0), FloatType::getF32(getContext()));
  }
  return success();
}

LogicalResult tpu::PowOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "pow", 1, 2, coeff().convertToFloat());
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::ReduceL2Op::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "pow", 1, 2, 0.5f);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SigmoidOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "sigmoid", 1, 2, scale().convertToFloat(), bias().convertToFloat());
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SwishOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "swish", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::TanHOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "tanh", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::EluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "elu", 1, 2);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::ExpOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "exp", 1, 2, scale().convertToFloat(), bias().convertToFloat());
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SoftmaxOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  assert(getOpQuant() == "BF16");
  insertBf16LutOp(op, "exp", 1, 2, 1.0, 0.0);
  if (do_log() == false) {
    insertBf16LutOp(op, "pow", 3, 4, -1.0f);
  } else {
    insertBf16LutOp(op, "log", 3, 4);
  }
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

LogicalResult tpu::SoftPlusOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  insertBf16LutOp(op, "softplus", 1, 2, scale().convertToFloat(), bias().convertToFloat());
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
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
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastSubOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ClipOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CscOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CustomOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::DilateOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::MulConstOp)
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
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMeanOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMaxOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMinOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceSumOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ScaleLutOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::UpsampleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ZeroMaskOp)

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
