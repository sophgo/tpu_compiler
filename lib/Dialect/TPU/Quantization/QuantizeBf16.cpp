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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/NativeCpuImplementation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "tpuc/MachineInfo.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "quantize_bf16"

float BF16_TABLE_START;
float BF16_TABLE_END;


double sigmoid(double x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

using namespace mlir;


namespace mlir {

///
/// Conv Ops quantization method
///
template<typename OpTy>
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

  // quantization
  FloatToBFloat16(filter->data(), new_filter->data(), filterSize);

  // update op
  addWeightTensorAndUpdateWeightOp<bfloat16>(convOp.getOperand(1),
      "quant", *new_filter, filterShape, "BF16", wTF);
  if (bias) {
    addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(2),
        "quant", *bias, biasShape, "FP32", wTF);
  }

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

static float revert_threshold;
double my_mish_caffe_wrapper (double x) {
  return my_mish_caffe(x, revert_threshold);
}

double softplus_activate_wrapper (double x) {
  return softplus_activate(x, revert_threshold);
}


///
/// Reciprocal quantization method
///
LogicalResult quantizeBF16ReciprocalOps(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "GenReciprocalLut: " << "]\n";);

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;

  std::vector<float> y0_reciprocal_table;
  std::vector<float> y0_reciprocal_mantissa_table; // use in bf16
  std::vector<uint16_t> table_data_lut_bf16(table_hw);
  std::vector<uint16_t> table_data_mantissa_lut_bf16(table_hw);
  y0_reciprocal_table.resize(tbl_shape);
  y0_reciprocal_mantissa_table.resize(tbl_shape);

  const int expStart = -62;
  const int expEnd = 63;
  bf16_gen_reciprocal(expStart, expEnd, table_hw, table_data_lut_bf16.data());
  bf16_gen_reciprocal_mantissa(expStart, expEnd, table_hw, table_data_mantissa_lut_bf16.data());

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i){
    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + table_hw,
              y0_reciprocal_table.data() + i * table_hw);
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + table_hw,
              y0_reciprocal_mantissa_table.data() + i * table_hw);
  }

  // update op
  StringRef storageType = "BF16";
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto y0_reciprocal_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "reciprocal_table", y0_reciprocal_table, shape, storageType, wTF, wfV);
  auto mantissa_reciprocal_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "reciprocal_mantissa_table", y0_reciprocal_mantissa_table, shape, storageType, wTF, wfV);
  op->setOperand(1, y0_reciprocal_table_op);
  op->setOperand(2, mantissa_reciprocal_table_op);
  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}


///
/// Sqrt quantization method
///
LogicalResult quantizeBF16SqrtOps(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "GenSqrtLut: " << "]\n";);

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;

  std::vector<float> y0_sqrt_table;
  std::vector<float> y0_sqrt_mantissa_table; // use in bf16
  std::vector<uint16_t> table_data_lut_bf16(table_hw);
  std::vector<uint16_t> table_data_mantissa_lut_bf16(table_hw);
  y0_sqrt_table.resize(tbl_shape);
  y0_sqrt_mantissa_table.resize(tbl_shape);

  const int expStart = -62;
  bf16_gen_sqrt(expStart, table_hw, table_data_lut_bf16.data());
  bf16_gen_sqrt_mantissa(table_hw, table_data_mantissa_lut_bf16.data());

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i){
    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + table_hw,
              y0_sqrt_table.data() + i * table_hw);
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + table_hw,
              y0_sqrt_mantissa_table.data() + i * table_hw);
  }

  // update op
  StringRef storageType = "BF16";
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto y0_sqrt_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sqrt_table", y0_sqrt_table, shape, storageType, wTF, wfV);
  auto mantissa_sqrt_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sqrt_mantissa_table", y0_sqrt_mantissa_table, shape, storageType, wTF, wfV);
  op->setOperand(1, y0_sqrt_table_op);
  op->setOperand(2, mantissa_sqrt_table_op);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
  return success();
}

///
/// LayerNorm quantization method
///
LogicalResult quantizeBf16LayerNormOps(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Gen layernorm ops: "
                          << "]\n";);

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;

  // sqrt table
  std::vector<float> table(tbl_shape);
  std::vector<float> mantissa_table(tbl_shape); // use in bf16
  std::vector<uint16_t> table_data_lut_bf16(table_hw);
  std::vector<uint16_t> table_data_mantissa_lut_bf16(table_hw);

  const int expStart = -62;
  bf16_gen_reciprocal_sqrt(expStart, table_hw, table_data_lut_bf16.data());
  bf16_gen_reciprocal_sqrt_mantissa(table_hw,
                                    table_data_mantissa_lut_bf16.data());

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i) {
    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + table_hw,
              table.data() + i * table_hw);
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + table_hw,
              mantissa_table.data() + i * table_hw);
  }

  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "table", table, shape, "BF16", wTF, wfV);
  auto mantissa_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "mantissa_table", mantissa_table, shape, "BF16", wTF, wfV);
  op->setOperand(3, table_op);
  op->setOperand(4, mantissa_table_op);
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

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto lutOp = cast<OpTy>(op);

  BF16_TABLE_START = lutOp.min_range().convertToFloat();
  BF16_TABLE_END = lutOp.max_range().convertToFloat();

  // quantization
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                          << ", threshold_y = " << std::to_string(threshold_y)
                          << ", threshold_x = " << std::to_string(threshold_x)
                          << "\n";);
  int npu_num = MInfo::lane_num;

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
  float _bf16_table_start = BF16_TABLE_START;
  float _bf16_table_end = BF16_TABLE_END;

  // use function pointer
  double (*activate_func)(double);
  if (OpTy::getOperationName() == "tpu.sigmoid") {
    activate_func = sigmoid;
  } else if (OpTy::getOperationName() == "tpu.tanh") {
    auto castOp = dyn_cast<tpu::TanHOp>(op);
    BF16_TABLE_START = castOp.min_range().convertToFloat();
    BF16_TABLE_END = castOp.max_range().convertToFloat();
    activate_func = tanh;
  } else if (OpTy::getOperationName() == "tpu.exp") {
    auto castOp = dyn_cast<tpu::ExpOp>(op);
    BF16_TABLE_START = castOp.min_range().convertToFloat();
    BF16_TABLE_END = castOp.max_range().convertToFloat();
    activate_func = exp;
  }
  else if (OpTy::getOperationName() == "tpu.mish") {
    auto castOp = dyn_cast<tpu::MishOp>(op);
    BF16_TABLE_START = castOp.min_range().convertToFloat();
    BF16_TABLE_END = castOp.max_range().convertToFloat();
    revert_threshold = castOp.mish_threshold().convertToFloat();
    activate_func = my_mish_caffe_wrapper;
  }
  else if (OpTy::getOperationName() == "tpu.softplus") {
    auto castOp = dyn_cast<tpu::SoftPlusOp>(op);
    BF16_TABLE_START = castOp.min_range().convertToFloat();
    BF16_TABLE_END = castOp.max_range().convertToFloat();
    revert_threshold = castOp.threshold().convertToFloat();
    activate_func = softplus_activate_wrapper;
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

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  BF16_TABLE_START = _bf16_table_start;
  BF16_TABLE_END = _bf16_table_end;
  return success();
}

///
/// FC Ops quantization method
///
LogicalResult quantizeBf16FullyConnectedOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

  auto fcOp = cast<tpu::FullyConnectedOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);

  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(fcOp.filter().getDefiningOp());
  std::unique_ptr<std::vector<float> > filter;
  std::vector<int64_t> filterShape;
  int64_t filterSize;

  // get filter tensor
  if (weightOp) {
    filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
    getTensorShapeAndSize(fcOp.filter(), filterShape, filterSize);
  }

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(fcOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
    getTensorShapeAndSize(fcOp.bias(), biasShape, biasSize);
  }

  if (weightOp) {
    // create new tensors
    auto new_filter = std::make_unique<std::vector<bfloat16> >(filterSize);

    // quantization
    FloatToBFloat16(filter->data(), new_filter->data(), filterSize);

    // update op
    addWeightTensorAndUpdateWeightOp<bfloat16>(fcOp.getOperand(1),
        "quant", *new_filter, filterShape, "BF16", wTF);
  }

  if (bias) {
    addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(2),
        "quant", *bias, biasShape, "FP32", wTF);
  }

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

///
/// Gru Ops quantization method
///
LogicalResult quantizeBf16GruOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

  auto gruOp = cast<tpu::GruOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  LLVM_DEBUG(llvm::dbgs() << "GenSigmoidLut: "
                          << "]\n";);
  // Add lut table information

  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int start = -8, end = 8;
  std::vector<float> y0_sigmoid_table;
  std::vector<float> y0_sigmoid_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_sigmoid_table.resize(tbl_shape);
  y0_sigmoid_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);
  StringRef storageType = "BF16";

  // use function pointer
  LLVM_DEBUG(llvm::dbgs() << "use function pointer: "
                          << "]\n";);
  double (*activate_func)(double);
  activate_func = sigmoid;
  gen_bf16_table(start, end, table_hw, y0_fp32_table.data(), activate_func);
  gen_bf16_slope_table(start, end, table_hw, y0_fp32_table.data(),
                       y0_fp32_slope_table.data(), activate_func);
  LLVM_DEBUG(llvm::dbgs() << "convert fp32 to bf16: "
                          << "]\n";);
  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(), y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(), y0_bf16_slope_table.data(),
                  table_hw);

  // copy bf16 data to float table
  LLVM_DEBUG(llvm::dbgs() << "copy bf16 data to float table: "
                          << "]\n";);
  for (int i = 0; i < npu_num; ++i) {
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_sigmoid_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(), y0_bf16_slope_table.data() + table_hw,
              y0_sigmoid_slope_table.data() + i * table_hw);
  }

  // update op
  LLVM_DEBUG(llvm::dbgs() << "update op: "
                          << "]\n";);
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto y0_sigmoid_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sigmoid_table", y0_sigmoid_table, shape, storageType, wTF, wfV);
  auto mantissa_sigmoid_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sigmoid_slope_table", y0_sigmoid_slope_table, shape, storageType,
      wTF, wfV);
  gruOp.setOperand(4, y0_sigmoid_table_op);
  gruOp.setOperand(5, mantissa_sigmoid_table_op);

  // Add lut table information - tanh
  LLVM_DEBUG(llvm::dbgs() << "GenTanhLut: "
                          << "]\n";);
  activate_func = tanh;
  std::vector<float> y0_tanh_table;
  std::vector<float> y0_tanh_slope_table; // use in bf16
  y0_tanh_table.resize(tbl_shape);
  y0_tanh_slope_table.resize(tbl_shape);

  gen_bf16_table(start, end, table_hw, y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(start, end, table_hw, y0_fp32_table.data(),
                       y0_fp32_slope_table.data(), activate_func);

  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(), y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(), y0_bf16_slope_table.data(),
                  table_hw);

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i) {
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_tanh_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(), y0_bf16_slope_table.data() + table_hw,
              y0_tanh_slope_table.data() + i * table_hw);
  }

  // update op
  auto y0_tanh_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "tanh_table", y0_tanh_table, shape, storageType, wTF, wfV);
  auto mantissa_tanh_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "tanh_slope_table", y0_tanh_slope_table, shape, storageType, wTF,
      wfV);
  gruOp.setOperand(6, y0_tanh_table_op);
  gruOp.setOperand(7, mantissa_tanh_table_op);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

///
/// softmax Ops quantization method
///
LogicalResult quantizeBf16SoftmaxOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  auto softmaxOp = cast<tpu::SoftmaxOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  LLVM_DEBUG(llvm::dbgs() << "GenExponentialLut: " << "]\n";);
  // Add lut table information

  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  std::vector<float> y0_exponential_table;
  std::vector<float> y0_exponential_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_exponential_table.resize(tbl_shape);
  y0_exponential_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<float> y0_fp32_mantissa_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_mantissa_table(table_hw);
  StringRef storageType = "BF16";

  // use function pointer
  LLVM_DEBUG(llvm::dbgs() << "use function pointer: " << "]\n";);
  double (*activate_func)(double);
  activate_func = exp;
  const int expTableStart = -15;
  const int expTableEnd = 1;

  gen_bf16_table(expTableStart, expTableEnd, table_hw,
                 y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(expTableStart, expTableEnd, table_hw,
                       y0_fp32_table.data(), y0_fp32_slope_table.data(),
                       activate_func);
  y0_fp32_table[128] = 0; // Make lut exp(x) = 0 when x <= -15
  // for(int i = 0; i < table_hw; i++) {
  //   LLVM_DEBUG(llvm::dbgs() << "exponential data[" << i<< "]: " <<  y0_fp32_table[i]<< "]\n";);
  //   LLVM_DEBUG(llvm::dbgs() << "exponential data[" << i<< "]: " <<  y0_fp32_slope_table[i]<< "]\n";);
  // }

  LLVM_DEBUG(llvm::dbgs() << "convert fp32 to bf16: " << "]\n";);
  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(),
                  y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(),
                  y0_bf16_slope_table.data(), table_hw);

  // copy bf16 data to float table
  LLVM_DEBUG(llvm::dbgs() << "copy bf16 data to float table: " << "]\n";);
  for (int i = 0; i < npu_num; ++i){
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_exponential_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(),
              y0_bf16_slope_table.data() + table_hw,
              y0_exponential_slope_table.data() + i * table_hw);
  }

  // update op
  LLVM_DEBUG(llvm::dbgs() << "update op: " << "]\n";);
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto y0_exponential_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "exponential_table", y0_exponential_table, shape, storageType, wTF, wfV);
  auto mantissa_exponential_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "exponential_slope_table", y0_exponential_slope_table, shape, storageType, wTF, wfV);
  softmaxOp.setOperand(1, y0_exponential_table_op);
  softmaxOp.setOperand(2, mantissa_exponential_table_op);

  //Add lut table information - reciprocal
  LLVM_DEBUG(llvm::dbgs() << "GenReciprocalLut: " << "]\n";);

  std::vector<float> y0_reciprocal_table;
  std::vector<float> y0_reciprocal_mantissa_table; // use in bf16
  std::vector<uint16_t> table_data_lut_bf16(table_hw);
  std::vector<uint16_t> table_data_mantissa_lut_bf16(table_hw);
  y0_reciprocal_table.resize(tbl_shape);
  y0_reciprocal_mantissa_table.resize(tbl_shape);

  const int expStart = -62;
  const int expEnd = 63;
  bf16_gen_reciprocal(expStart, expEnd, table_hw, table_data_lut_bf16.data());
  bf16_gen_reciprocal_mantissa(expStart, expEnd, table_hw, table_data_mantissa_lut_bf16.data());

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i){
    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + table_hw,
              y0_reciprocal_table.data() + i * table_hw);
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + table_hw,
              y0_reciprocal_mantissa_table.data() + i * table_hw);
  }

  // update op
  auto y0_reciprocal_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "reciprocal_table", y0_reciprocal_table, shape, storageType, wTF, wfV);
  auto mantissa_reciprocal_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "reciprocal_mantissa_table", y0_reciprocal_mantissa_table, shape, storageType, wTF, wfV);
  softmaxOp.setOperand(3, y0_reciprocal_table_op);
  softmaxOp.setOperand(4, mantissa_reciprocal_table_op);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

///
/// Lrn Ops quantization method
///
LogicalResult quantizeBf16LrnOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto lrnOp = cast<tpu::LrnOp>(op);

  const int EXP_START = -62;
  const int NPU_NUM = MInfo::lane_num;
  const int TABLE_H_BF16 = 32;
  const int TABLE_W_BF16 = 8;
  const int TABLE_HW_BF16 = (TABLE_H_BF16 * TABLE_W_BF16);
  const int TBL_SHAPE_BF16 = (TABLE_HW_BF16 * NPU_NUM);

  float beta = lrnOp.beta().convertToFloat();

  // power table
  std::vector<uint16_t> power_exp_table_bf16(TBL_SHAPE_BF16);
  std::vector<uint16_t> power_mantissa_table_bf16(TBL_SHAPE_BF16);
  std::vector<float> power_exp_table(TBL_SHAPE_BF16);
  std::vector<float> power_mantissa_table(TBL_SHAPE_BF16);

  // gen exp table
  bf16_gen_power_exp_table(power_exp_table_bf16.data(), beta, EXP_START,
                           TABLE_HW_BF16);
  // gen matissa table
  bf16_gen_power_mantissa_table(power_mantissa_table_bf16.data(), beta,
                                TABLE_HW_BF16);

  // copy bf16 data to float table
  for (int i = 0; i < NPU_NUM; ++i) {
    std::copy(power_exp_table_bf16.data(),
              power_exp_table_bf16.data() + TABLE_HW_BF16,
              power_exp_table.data() + i * TABLE_HW_BF16);
    std::copy(power_mantissa_table_bf16.data(),
              power_mantissa_table_bf16.data() + TABLE_HW_BF16,
              power_mantissa_table.data() + i * TABLE_HW_BF16);
  }

  // update op params
  std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_BF16, TABLE_W_BF16};
  auto power_exp_op = addWeightTensorAndCreateWeightOp<float>(
      op, "power_exp_weight", power_exp_table, weightShape, "BF16", wTF, wfV);
  lrnOp.setOperand(1, power_exp_op);
  auto power_mantissa_op = addWeightTensorAndCreateWeightOp<float>(
      op, "power_mantissa_weight", power_mantissa_table, weightShape, "BF16",
      wTF, wfV);
  lrnOp.setOperand(2, power_mantissa_op);

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
  lreluOp->setAttr("negative_slope", builder.getF32FloatAttr(quant_negative_slope));

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

///
/// Lstm Ops quantization method
///
LogicalResult quantizeBf16LstmOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

  auto lstmOp = cast<tpu::LstmOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // get weight tensor
  auto weight = readAndDeleteWeightTensor<float>(lstmOp.weight(), wTF);
  std::vector<int64_t> weightShape;
  int64_t weightSize;
  getTensorShapeAndSize(lstmOp.weight(), weightShape, weightSize);

  // get recurrence tensor
  auto recurrence = readAndDeleteWeightTensor<float>(lstmOp.recurrence(), wTF);
  std::vector<int64_t> recurrenceShape;
  int64_t recurrenceSize;
  getTensorShapeAndSize(lstmOp.recurrence(), recurrenceShape, recurrenceSize);

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(lstmOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(lstmOp.bias(), wTF);
    getTensorShapeAndSize(lstmOp.bias(), biasShape, biasSize);
  }

  // get initial_h tensor
  std::unique_ptr<std::vector<float> > initial_h = nullptr;
  std::vector<int64_t> initial_hShape;
  int64_t initial_hSize = 0;
  if ( !isTensorNone(lstmOp.initial_h()) ) {
    initial_h = readAndDeleteWeightTensor<float>(lstmOp.initial_h(), wTF);
    getTensorShapeAndSize(lstmOp.initial_h(), initial_hShape, initial_hSize);
  }

  // get initial_c tensor
  std::unique_ptr<std::vector<float> > initial_c = nullptr;
  std::vector<int64_t> initial_cShape;
  int64_t initial_cSize = 0;

  auto initial_h_weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
                  lstmOp.initial_h().getDefiningOp());
  auto initial_c_weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
                  lstmOp.initial_c().getDefiningOp());

  bool b_initial_h_is_same_as_initial_c = false;
  assert(initial_h_weightOp.name() == initial_c_weightOp.name());

  if ( !isTensorNone(lstmOp.initial_c())) {
    if (initial_h_weightOp.name() == initial_c_weightOp.name()) {
      b_initial_h_is_same_as_initial_c = true;
    }
    else {
      initial_c = readAndDeleteWeightTensor<float>(lstmOp.initial_c(), wTF);
      getTensorShapeAndSize(lstmOp.initial_c(), initial_cShape, initial_cSize);
    }
  }

  // create new tensors
  auto new_weight = std::make_unique<std::vector<bfloat16> >(weightSize);
  auto new_recurrence = std::make_unique<std::vector<bfloat16> >(recurrenceSize);
  std::unique_ptr<std::vector<bfloat16> > new_initial_h = nullptr;
  std::unique_ptr<std::vector<bfloat16> > new_initial_c = nullptr;

  if (initial_h)
    new_initial_h = std::make_unique<std::vector<bfloat16> >(initial_hSize);

  if (initial_c)
    new_initial_c = std::make_unique<std::vector<bfloat16> >(initial_cSize);

  // quantization
  FloatToBFloat16(weight->data(), new_weight->data(), weightSize);
  FloatToBFloat16(recurrence->data(), new_recurrence->data(), recurrenceSize);

  if (initial_h)
    FloatToBFloat16(initial_h->data(), new_initial_h->data(), initial_hSize);

  if (initial_c)
    FloatToBFloat16(initial_c->data(), new_initial_c->data(), initial_cSize);

  // update op
  addWeightTensorAndUpdateWeightOp<bfloat16>(lstmOp.getOperand(1),
      "quant", *new_weight, weightShape, "BF16", wTF);
  addWeightTensorAndUpdateWeightOp<bfloat16>(lstmOp.getOperand(2),
      "quant", *new_recurrence, recurrenceShape, "BF16", wTF);

  if (bias)
    addWeightTensorAndUpdateWeightOp<float>(lstmOp.getOperand(3),
        "quant", *bias, biasShape, "FP32", wTF);

  if (initial_h)
    addWeightTensorAndUpdateWeightOp<bfloat16>(lstmOp.getOperand(4),
        "quant", *new_initial_h, initial_hShape, "BF16", wTF);

  if (initial_c)
    addWeightTensorAndUpdateWeightOp<bfloat16>(lstmOp.getOperand(5),
        "quant", *new_initial_c, initial_cShape, "BF16", wTF);
  else if (b_initial_h_is_same_as_initial_c)
    addWeightTensorAndUpdateWeightOp<bfloat16>(lstmOp.getOperand(5),
          "quant", *new_initial_h, initial_hShape, "BF16", wTF);

  LLVM_DEBUG(llvm::dbgs() << "GenSigmoidLut: " << "]\n";);
  // Add lut table information

  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  std::vector<float> y0_sigmoid_table;
  std::vector<float> y0_sigmoid_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_sigmoid_table.resize(tbl_shape);
  y0_sigmoid_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);
  StringRef storageType = "BF16";

  // use function pointer
  LLVM_DEBUG(llvm::dbgs() << "use function pointer: " << "]\n";);
  double (*activate_func)(double);
  activate_func = sigmoid;

  gen_bf16_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                 y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                       y0_fp32_table.data(), y0_fp32_slope_table.data(),
                       activate_func);
  // for(int i = 0; i < table_hw; i++) {
  //   LLVM_DEBUG(llvm::dbgs() << "sigmoid data[" << i<< "]: " <<  y0_fp32_table[i]<< "]\n";);
  //   LLVM_DEBUG(llvm::dbgs() << "sigmoid data[" << i<< "]: " <<  y0_fp32_slope_table[i]<< "]\n";);
  // }

  LLVM_DEBUG(llvm::dbgs() << "convert fp32 to bf16: " << "]\n";);
  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(),
                  y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(),
                  y0_bf16_slope_table.data(), table_hw);

  // copy bf16 data to float table
  LLVM_DEBUG(llvm::dbgs() << "copy bf16 data to float table: " << "]\n";);
  for (int i = 0; i < npu_num; ++i){
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_sigmoid_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(),
              y0_bf16_slope_table.data() + table_hw,
              y0_sigmoid_slope_table.data() + i * table_hw);
  }

  // update op
  LLVM_DEBUG(llvm::dbgs() << "update op: " << "]\n";);
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  auto y0_sigmoid_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sigmoid_table", y0_sigmoid_table, shape, storageType, wTF, wfV);
  auto mantissa_sigmoid_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sigmoid_slope_table", y0_sigmoid_slope_table, shape, storageType, wTF, wfV);
  lstmOp.setOperand(6, y0_sigmoid_table_op);
  lstmOp.setOperand(7, mantissa_sigmoid_table_op);

  //Add lut table information - tanh
  LLVM_DEBUG(llvm::dbgs() << "GenTanhLut: " << "]\n";);
  activate_func = tanh;
  std::vector<float> y0_tanh_table;
  std::vector<float> y0_tanh_slope_table; // use in bf16
  y0_tanh_table.resize(tbl_shape);
  y0_tanh_slope_table.resize(tbl_shape);

  gen_bf16_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                 y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
                       y0_fp32_table.data(), y0_fp32_slope_table.data(),
                       activate_func);
  //  for(int i = 0; i < table_hw; i++) {
  //   LLVM_DEBUG(llvm::dbgs() << "tanh data[" << i<< "]: " <<  y0_fp32_table[i]<< "]\n";);
  //   LLVM_DEBUG(llvm::dbgs() << "tanh data[" << i<< "]: " <<  y0_fp32_slope_table[i]<< "]\n";);
  // }

  // convert fp32 to bf16
  FloatToBFloat16(y0_fp32_table.data(),
                  y0_bf16_table.data(), table_hw);
  FloatToBFloat16(y0_fp32_slope_table.data(),
                  y0_bf16_slope_table.data(), table_hw);

  // copy bf16 data to float table
  for (int i = 0; i < npu_num; ++i){
    std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
              y0_tanh_table.data() + i * table_hw);
    std::copy(y0_bf16_slope_table.data(),
              y0_bf16_slope_table.data() + table_hw,
              y0_tanh_slope_table.data() + i * table_hw);
  }

  // update op
  auto y0_tanh_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "tanh_table", y0_tanh_table, shape, storageType, wTF, wfV);
  auto mantissa_tanh_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "tanh_slope_table", y0_tanh_slope_table, shape, storageType, wTF, wfV);
  lstmOp.setOperand(8, y0_tanh_table_op);
  lstmOp.setOperand(9, mantissa_tanh_table_op);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

/// PRelu Ops quantization method
///
LogicalResult quantizeBf16PReluOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");
  TensorFile *wTF = getWeightTensorFile(op);

  auto neg_slope =
      readAndDeleteWeightTensor<float>(op->getOperand(1), wTF);
  std::vector<int64_t> neg_slope_shape;
  int64_t neg_slope_size;
  getTensorShapeAndSize(op->getOperand(1), neg_slope_shape, neg_slope_size);

  auto quant_neg_slope = std::make_unique<std::vector<bfloat16> >(neg_slope_size);
  FloatToBFloat16(neg_slope->data(), quant_neg_slope->data(), neg_slope_size);

  addWeightTensorAndUpdateWeightOp<bfloat16>(op->getOperand(1),
      "quant", *quant_neg_slope, neg_slope_shape, "BF16", wTF);

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

  return success();
}

///
/// bypass Ops quantization method
///
LogicalResult quantizeBf16BypassOps(Operation *op) {
  assert(getOpQuant(op) == "BF16");

  setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));

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

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::AbsOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastMulOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastAddOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::BroadcastSubOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ConcatOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ClipOp)

LogicalResult tpu::ArgMaxOp::quantizeBf16() {
  Operation *op = this->getOperation();
  setOpResultType(op->getResult(0), FloatType::getF32(getContext()));
  return success();
}

LogicalResult tpu::Conv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv2DOp>(op, 2);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::DilateOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CustomOp)

LogicalResult tpu::DeConv2DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::DeConv2DOp>(op, 2);
}

LogicalResult tpu::Conv3DOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16ConvOps<tpu::Conv3DOp>(op, 3);
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
LogicalResult tpu::InterpOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto interpOp = cast<tpu::InterpOp>(op);
  llvm::StringRef type = "NONE";
  interpOp.setOpQuantMode(type);
  return success();
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::InstanceNormOp)

LogicalResult tpu::LayerNormOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LayerNormOps(op);
}

LogicalResult tpu::LeakyReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LeakyReluOps(op);
}

LogicalResult tpu::PReluOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n");
  Operation *op = this->getOperation();
  return quantizeBf16PReluOps(op);
}

LogicalResult tpu::LrnOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n");
  Operation *op = this->getOperation();
  return quantizeBf16LrnOps(op);
}

LogicalResult tpu::LstmOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16LstmOps(op);
}

LogicalResult tpu::MishOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::MishOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PadOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PermuteOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PixelShuffleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolAvg2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMax3DOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PoolMaskOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::PowerOp)
// DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReciprocalOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceL2Op)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMeanOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReduceMaxOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SquareOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::MatMulOp)

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
  return quantizeBF16SqrtOps(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::ScaleLutOp)

LogicalResult tpu::ReciprocalOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16ReciprocalOps(op);
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
// DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TanHOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SliceOp)
// DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SqrtOp)

LogicalResult tpu::SoftmaxOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBf16SoftmaxOps(op);
}

LogicalResult tpu::SoftPlusOp::quantizeBf16() {
  LLVM_DEBUG(llvm::errs() << "quantizeBf16: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeBF16LutOps<tpu::SoftPlusOp>(op);
}

DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::UpsampleOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::TileInterpOp)
DECLARE_QUANTIZE_BF16_BYPASS_METHOD(tpu::CscOp)

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
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnOneOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnTwoOp)
DECLARE_QUANTIZE_BF16_DISABLED_METHOD(tpu::LrnThreeOp)
} // namespace mlir
