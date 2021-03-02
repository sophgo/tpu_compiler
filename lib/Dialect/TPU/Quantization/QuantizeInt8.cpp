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
#include "tpuc/MachineInfo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <fstream>

#define DEBUG_TYPE "quantize_int8"

using namespace mlir;

namespace mlir {

///
/// Conv Ops quantization method
///
template<typename OpTy>
LogicalResult quantizeInt8ConvOps(Operation *op, int spatial_dims) {
  assert(getOpQuant(op) == "INT8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto convOp = cast<OpTy>(op);
  auto filterOp =
      llvm::dyn_cast<tpu::LoadWeightOp>(convOp.filter().getDefiningOp());
  // get filter tensor
  auto filter = readWeightTensor<float>(convOp.filter(), wTF);
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
  int64_t isz = filterSize / oc;

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(convOp.bias()) ) {
    bias = readWeightTensor<float>(convOp.bias(), wTF);
    getTensorShapeAndSize(convOp.bias(), biasShape, biasSize);
    assert(biasSize == oc);
    assert(biasSize == (int64_t)bias->size());
  }

  std::vector<float> filter_threshold;
  // get filter threshold if existed
  if (filterOp.threshold().hasValue()) {
    arrayAttrToVector(filterOp.threshold().getValue(), filter_threshold);
    if (filter_threshold.size() != (uint32_t)oc) {
      llvm::errs() << getOpName(op)
                   << " Filter threshold size is not same with oc ("
                   << filter_threshold.size() << "v.s." << oc << ")\n";
      llvm::errs() << "no use filter threshold\n";
      filter_threshold.clear();
    }else{
      llvm::outs() << getOpName(op) << " Use filter threshold\n";
    }
  }

  // create new tensors
  auto new_filter = std::make_unique<std::vector<float> >(filterSize);
  std::unique_ptr<std::vector<float> > new_bias = nullptr;
  if (bias) {
    new_bias = std::make_unique<std::vector<float> >(biasSize);
  }

  // create tensors for rshift and multiplier
  auto rshift_per_layer = std::make_unique<std::vector<float> >(1);
  auto rshift_per_channel = std::make_unique<std::vector<float> >(oc);
  auto multiplier_per_channel = std::make_unique<std::vector<float> >(oc);

  // get threshold
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
               << ", threshold_y = "<< std::to_string(threshold_y)
               << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

  // quantization
  if (!isOpQuantPerchannel(op)) {
    assert(getOpQuantParamType(op) == "RSHIFT_ONLY");
    quantizeWeightInt8PerLayer(filter->data(), bias ? bias->data() : nullptr,
                               oc, isz, threshold_y, threshold_x,
                               new_filter->data(), bias ? new_bias->data() : nullptr,
                               rshift_per_layer->data());

  } else if (getOpQuantParamType(op) == "RSHIFT_ONLY") {
    quantizeWeightInt8PerChannel(filter->data(), bias ? bias->data() : nullptr,
                               oc, isz, threshold_y, threshold_x,
                               new_filter->data(), bias ? new_bias->data() : nullptr,
                               rshift_per_channel->data());

  } else if (getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
    quantizeWeightInt8Multiplier(
        filter->data(), bias ? bias->data() : nullptr, oc, isz, threshold_y,
        threshold_x, new_filter->data(), bias ? new_bias->data() : nullptr,
        rshift_per_channel->data(), multiplier_per_channel->data(),
        filter_threshold);

  } else {
    assert(0);
  }
  std::string quant_value = std::to_string(threshold_y) + "_quant";
  // update op
  addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
      quant_value, *new_filter, filterShape, "INT8", wTF);
  if (bias) {
    // for per_channel quant, bias store as INT32, per layer use INT16
    StringRef storageType = isOpQuantPerchannel(op) ? "INT32" : "INT16";
    addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(2),
        quant_value, *new_bias, biasShape, storageType, wTF);
  }

  // add rshift and multiplier (if present) to weight
  if (!isOpQuantPerchannel(op)) {
    assert(getOpQuantParamType(op) == "RSHIFT_ONLY");
    auto shape = std::vector<int64_t>{1};
    StringRef storageType = "NONE";
    auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
        op, "rshift", *rshift_per_layer, shape, storageType,
        wTF, wfV);
    convOp.setOperand(5, rshift_op);
  } else if (getOpQuantParamType(op) == "RSHIFT_ONLY") {
    auto shape = std::vector<int64_t>{oc};
    StringRef storageType = "UINT32";
    auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
        op, "rshift", *rshift_per_channel, shape, storageType,
        wTF, wfV);
    convOp.setOperand(5, rshift_op);
  } else if (getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
    auto shape = std::vector<int64_t>{oc};
    StringRef storageType = "UINT32";
    auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
        op, "rshift", *rshift_per_channel, shape, storageType,
        wTF, wfV);
    convOp.setOperand(5, rshift_op);
    auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
        op, "multiplier", *multiplier_per_channel, shape, storageType,
        wTF, wfV);
    convOp.setOperand(6, multiplier_op);
  } else {
    assert(0);
  }
  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}


///
/// FC Ops quantization method
///
LogicalResult quantizeInt8FullyConnectedOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // support rshift-only only for now
  setOpQuantParamType(op, "RSHIFT_AND_M_I32");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto fcOp = cast<tpu::FullyConnectedOp>(op);

  // parse param
  int m, k, n;
  parseFullyConnectedParam(fcOp.input(), fcOp.output(), fcOp.filter(),
                           m, k, n);

  // get filter tensor
  auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
  std::vector<int64_t> filterShape;
  int64_t filterSize;
  getTensorShapeAndSize(fcOp.filter(), filterShape, filterSize);
  assert(filterSize == k * n);

  // get bias tensor
  std::unique_ptr<std::vector<float> > bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if ( !isTensorNone(fcOp.bias()) ) {
    bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
    getTensorShapeAndSize(fcOp.bias(), biasShape, biasSize);
    assert(biasSize == n);
  }

  // create new tensors
  auto new_filter = std::make_unique<std::vector<float> >(filterSize);
  std::unique_ptr<std::vector<float> > new_bias = nullptr;
  if (bias) {
    new_bias = std::make_unique<std::vector<float> >(biasSize);
  }

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier_per_layer = std::make_unique<std::vector<float>>(1);
  // get threshold
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
               << ", threshold_y = "<< std::to_string(threshold_y)
               << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

  // quantization
  quantizeWeightInt8PerLayerMultiplier(filter->data(), bias ? bias->data() : nullptr,
                               n, k, threshold_y, threshold_x,
                               new_filter->data(), bias ? new_bias->data() : nullptr,
                               rshift->data(),
                               multiplier_per_layer->data());

  // update op
  addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(1),
      "quant", *new_filter, filterShape, "INT8", wTF);
  if (bias) {
    // qdm mode, bias use INT32
    addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(2),
        "quant", *new_bias, biasShape, "INT32", wTF);
  }

  // add rshift to weight
  auto shape = std::vector<int64_t>{1};
  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, "NONE",
      wTF, wfV);
  fcOp.setOperand(5, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier_per_layer, shape, "NONE", wTF, wfV);
  fcOp.setOperand(6, multiplier_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// LeakyRelu Ops quantization method
///
LogicalResult quantizeInt8LeakyReluOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto lreluOp = cast<tpu::LeakyReluOp>(op);

  float negative_slope = lreluOp.negative_slope().convertToFloat();
  LLVM_DEBUG(llvm::errs() << "  negative_slope: "
                          << std::to_string(negative_slope) << "\n";);

  // create tensors for rshift and multiplier
  auto rshift_pos = std::vector<float>(1);
  auto multiplier_pos = std::vector<float>(1);
  auto rshift_neg = std::vector<float>(1);
  auto multiplier_neg = std::vector<float>(1);

  // quantization
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
               << ", threshold_y = "<< std::to_string(threshold_y)
               << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

  // positive
  double qscale_pos = threshold_x / threshold_y;
  if (fabs(threshold_x - threshold_y) < 1e-5 * std::min(threshold_x, threshold_y)) {
    // no positive scale
    rshift_pos[0] = 0;
    multiplier_pos[0] = 1.0f;
    LLVM_DEBUG(llvm::errs() << "  Positive: no_scale\n";);
  } else {
    uint32_t uint_multiplier_pos;
    rshift_pos[0] = (float)findRShiftAndMultiplierFromQScale(qscale_pos,
        &uint_multiplier_pos, false);
    multiplier_pos[0] = (float)uint_multiplier_pos;
    LLVM_DEBUG(llvm::errs() << "  Positive: ";);
    LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                            << std::to_string(multiplier_pos[0]) << " : "
                            << std::to_string(rshift_pos[0]) << "]\n";);
  }
  // negative
  float qscale_neg = fabs(qscale_pos * negative_slope);
  uint32_t uint_multiplier_neg = 0;
  rshift_neg[0] = (float)findRShiftAndMultiplierFromQScale(qscale_neg,
      &uint_multiplier_neg, false);
  multiplier_neg[0] = (float)uint_multiplier_neg;
  LLVM_DEBUG(llvm::errs() << "  Negative: ";);
  LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                          << std::to_string(multiplier_neg[0]) << " : "
                          << std::to_string(rshift_neg[0]) << "]\n";);

  // update op
  auto shape = std::vector<int64_t>{1};
  StringRef storageType = "NONE";

  auto rshift_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_pos", rshift_pos, shape, storageType,
      wTF, wfV);
  lreluOp.setOperand(5, rshift_pos_op);

  auto multiplier_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier_pos", multiplier_pos, shape, storageType,
      wTF, wfV);
  lreluOp.setOperand(6, multiplier_pos_op);

  auto rshift_neg_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_neg", rshift_neg, shape, storageType,
      wTF, wfV);
  lreluOp.setOperand(7, rshift_neg_op);

  auto multiplier_neg_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier_neg", multiplier_neg, shape, storageType,
      wTF, wfV);
  lreluOp.setOperand(8, multiplier_neg_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// Prelu Ops quantization method
///
LogicalResult quantizeInt8PReluOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto preluOp = cast<tpu::PReluOp>(op);

  //get negative slope tensor info
  auto negative_slope_weight =
      readAndDeleteWeightTensor<float>(op->getOperand(1), wTF);
  std::vector<int64_t> neg_slope_shape;
  int64_t neg_slope_size;
  getTensorShapeAndSize(op->getOperand(1), neg_slope_shape, neg_slope_size);
  int c = neg_slope_shape[1];
  // create tensors for rshift and multiplier
  auto new_negative_slope =
      std::vector<float>(neg_slope_size);

  auto rshift_pos = std::vector<float>(1);
  auto multiplier_pos = std::vector<float>(1);
  auto rshift_neg = std::vector<float>(1);
  auto multiplier_neg = std::vector<float>(1);

  // quantization
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                          << ", threshold_y = " << std::to_string(threshold_y)
                          << ", threshold_x = " << std::to_string(threshold_x)
                          << "\n";);

  // positive
  double qscale_pos = threshold_x / threshold_y;
  if (fabs(threshold_x - threshold_y) <
      1e-5 * std::min(threshold_x, threshold_y)) {
    // no positive scale
    rshift_pos[0] = 0;
    multiplier_pos[0] = 1.0f;
    LLVM_DEBUG(llvm::errs() << "  Positive: no_scale\n";);
  } else {
    uint32_t uint_multiplier_pos;
    rshift_pos[0] = (float)findRShiftAndMultiplierFromQScale(
        qscale_pos, &uint_multiplier_pos, false);
    multiplier_pos[0] = (float)uint_multiplier_pos;
    LLVM_DEBUG(llvm::errs() << "  Positive: ";);
    LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                            << std::to_string(multiplier_pos[0]) << " : "
                            << std::to_string(rshift_pos[0]) << "]\n";);
  }
  // negative
  LLVM_DEBUG(llvm::errs() << "  Negative:";);

  // I8 Multiplier
  double max_abs_negative_qscale = fabs(qscale_pos * negative_slope_weight->at(0));
  for (int i = 1; i < c; ++i) {
    if (fabs(qscale_pos * negative_slope_weight->at(i)) > max_abs_negative_qscale) {
      max_abs_negative_qscale = fabs(qscale_pos * negative_slope_weight->at(i));
    }
  }
  uint32_t uint_multiplier_neg = 0;
  float rshift_tmp = (float)findRShiftAndMultiplierFromQScale(
                max_abs_negative_qscale, &uint_multiplier_neg, false);
  rshift_neg[0] = rshift_tmp;
  LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                          << std::to_string(uint_multiplier_neg) << " : "
                          << std::to_string(rshift_neg[0]) << "]\n";);

  for (int i = 0; i < c; ++i) {
    new_negative_slope[i] = (float)quantizeFilterRShift(
        negative_slope_weight->at(i), threshold_y, threshold_x, rshift_neg[0]);
  }

  // update op
  auto shape = std::vector<int64_t>{1};
  StringRef storageType = "INT8";
  addWeightTensorAndUpdateWeightOp<float>(op->getOperand(1),
      "negative_slope", new_negative_slope, neg_slope_shape, storageType, wTF);
  auto rshift_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_pos", rshift_pos, shape, storageType, wTF, wfV);
  preluOp.setOperand(6, rshift_pos_op);

  auto multiplier_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier_pos", multiplier_pos, shape, storageType, wTF,
      wfV);
  preluOp.setOperand(7, multiplier_pos_op);

  auto rshift_neg_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_neg", rshift_neg, shape, storageType, wTF, wfV);
  preluOp.setOperand(8, rshift_neg_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// Lut Ops quantization method
///
template<typename OpTy>
LogicalResult quantizeInt8LutOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use LUT
  setOpQuantParamType(op, "LUT_INT8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto lutOp = cast<OpTy>(op);

  // quantization
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                          << ", threshold_y = " << std::to_string(threshold_y)
                          << ", threshold_x = " << std::to_string(threshold_x)
                          << "\n";);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw config
  int table_h;
  int table_w;
  int table_hw;

  int tbl_shape;
  std::vector<float> y0_table;

  //<! 1880v2 hw int8 config
  table_h = 16;
  table_w = 16;
  table_hw = table_h * table_w;

  tbl_shape = npu_num * table_hw;
  y0_table.resize(tbl_shape);
  std::vector<float> table_mantissa(tbl_shape,0); //do nothing during int8 quant

  // input: 0~127, -128~ -1, Y=1/(1+EXP(-X*thx/128)) * 128/thy
  // output:0~127, negative is invalid
  for (int n = 0; n < npu_num; n++) {
    for (int idx = 0; idx < table_hw; ++idx) {
      char lutInput = static_cast<char>(idx);
      float index = lutInput * threshold_x / 127.0;

      if(OpTy::getOperationName()=="tpu.reciprocal"){
        int lutOutputI32 = 127;
        if (index != 0) {
          float lutOutput = 1.0 / (index)*127.0 / threshold_y;
          lutOutputI32 = std::floor(lutOutput + 0.5);
          lutOutputI32 = (lutOutputI32 > 127)
                             ? 127
                             : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        }
        y0_table[n * table_hw + idx] = lutOutputI32;
      }else if(OpTy::getOperationName()=="tpu.sqrt"){
        float lutOutput = pow(index,0.5) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                          ? 127
                          : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else if (OpTy::getOperationName() == "tpu.sigmoid") {
        index = -lutInput * threshold_x / 127.0;
        float lutOutput = 1.0 / (1 + std::exp(index)) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else if (OpTy::getOperationName() == "tpu.tanh") {
        index = lutInput * threshold_x / 127.0;
        float lutOutput = std::tanh(index) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else if (OpTy::getOperationName() == "tpu.exp") {
        index = lutInput * threshold_x / 127.0;
        float lutOutput = std::exp(index) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else if (OpTy::getOperationName() == "tpu.mish") {
        index = lutInput * threshold_x / 127.0;
        auto castOp = dyn_cast<tpu::MishOp>(op);
        float mish_threshold = castOp.mish_threshold().convertToFloat();
        float lutOutput = my_mish_caffe(index, mish_threshold) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else if (OpTy::getOperationName() == "tpu.softplus") {
        index = lutInput * threshold_x / 127.0;
        auto castOp = dyn_cast<tpu::SoftPlusOp>(op);
        float threshold = castOp.threshold().convertToFloat();
        float lutOutput = softplus_activate(index, threshold) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      } else {
        assert(false && "not support now");
      }
    }
  }

  // update op
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  StringRef storageType = "INT8";
  auto y0_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "y0_table", y0_table, shape, storageType, wTF, wfV);
  auto mantissa_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "mantissa_table", table_mantissa, shape, storageType, wTF, wfV);
  lutOp.setOperand(1, y0_table_op);
  lutOp.setOperand(2, mantissa_table_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

// input 0~255 => output -128~127
LogicalResult quantizeInt8ScaleLutOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto castOp = cast<tpu::ScaleLutOp>(op);
  std::vector<float> scale;
  std::vector<float> bias;
  arrayAttrToVector(castOp.scale(), scale);
  arrayAttrToVector(castOp.bias(), bias);

  int table_h = 16;
  int table_w = 16;
  int table_hw = table_h * table_w;
  int npu_num = MInfo::lane_num;
  int table_size = npu_num * table_hw;
  std::vector<float> table(table_size, 0.0f);
  for (int i = 0; i < 3; i++) {
    for (int idx = 0; idx < table_hw; ++idx) {
      float data = std::floor(idx * scale[i] + bias[i] + 0.5);
      data = std::min(std::max(data, -128.0f), 127.0f);
      table[i * table_hw + idx] = data;
    }
  }
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  StringRef storageType = "INT8";
  auto table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "table", table, shape, storageType, wTF, wfV);
  castOp.setOperand(1, table_op);
  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));
  return success();
}

///
/// default no weight Ops quantization method
/// for operations that has no weight, but still need to do rescaling
/// for multiple inputs op (eltwise add/max, concat)
///      - first n operands are variadic
///      - 4 quant operands are following
/// special handling for some operations
///   1. PoolAvg2D: needs to take 1 / (kh * kw) into account
///   2. ReduceMean: needs to take 1 / reduced_size into account
///
template<typename OpTy>
LogicalResult quantizeInt8RescaleNoWeightOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  bool bypass = true;
  float bypass_eps = 1e-5;
  // get thresholds
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
                          << std::to_string(threshold_y) << "\n";);

  // get operands
  unsigned nInputs = op->getNumOperands() - 4;
  if (isa<tpu::ClipOp>(op)) {
    nInputs = 1; // clip ONLY one input

    // update its min/max
    auto castOp = dyn_cast<tpu::ClipOp>(op);
    auto rewriter = Builder(castOp.getContext());
    int min = castOp.min().convertToFloat() * 127.0 / threshold_y;
    int max = castOp.max().convertToFloat() * 127.0 / threshold_y;
    castOp->setAttr("min", rewriter.getF32FloatAttr(min));
    castOp->setAttr("max", rewriter.getF32FloatAttr(max));
  }

  std::vector<float> threshold_x(nInputs);
  for (unsigned i = 0; i < nInputs; ++i) {
    threshold_x[i] = getPreviousOpThreshold(op, i);
    LLVM_DEBUG(llvm::errs() << "  threshold_x[" << i << "] = "
               << std::to_string(threshold_x[i]) << "\n";);
    if (fabs(threshold_y - threshold_x[i]) > bypass_eps) {
      bypass = false;
    }
  }
  if (bypass && OpTy::getOperationName() != "tpu.pool_avg_2d"
             && OpTy::getOperationName() != "tpu.concat"
             && OpTy::getOperationName() != "tpu.eltwise_max"
             && OpTy::getOperationName() != "tpu.eltwise_min"
             && OpTy::getOperationName() != "tpu.eltwise_add"
             && OpTy::getOperationName() != "tpu.reduce_mean"
             && OpTy::getOperationName() != "tpu.reduce_max") {
    // leave quant_rshift and quant_multiplier as NoneOp to indicate bypass
    LLVM_DEBUG(llvm::errs() << " < " << getOpName(op)
                            << ",  quantization bypassed\n";);
    setOpQuantParamType(op, "NONE");
    setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

    return success();
  }

  //
  // determine the qscale first, scale is different among inputs
  //   qscale is a fp32 scale applied to each input
  //   so that the share the same threshold_y
  //
  std::vector<float> qscale(nInputs);
  for (unsigned i = 0; i < nInputs; ++i) {
    qscale[i] = threshold_x[i] / threshold_y;
  }

  // special handlings for eltwise with coeff
  if (OpTy::getOperationName() == "tpu.eltwise_add") {
    auto castOp = dyn_cast<tpu::EltwiseAddOp>(op);
    std::vector<float> coeff(nInputs);
    if (castOp.coeff().hasValue()){
      arrayAttrToVector(castOp.coeff().getValue(), coeff);
      for (unsigned i = 0; i < nInputs; ++i) {
        qscale[i] = coeff[i] * threshold_x[i] / threshold_y;
      }
    }
  }
  // special handlings
  if (OpTy::getOperationName() == "tpu.pool_avg_2d") {
    assert(nInputs);
    auto castOp = dyn_cast<tpu::PoolAvg2DOp>(op);
    assert(castOp);
    int kh = castOp.param().kernel_h().getInt();
    int kw = castOp.param().kernel_w().getInt();
    qscale[0] = qscale[0] / (kh * kw);
  }

  // special handlings with pool_avg_2d count_include_padding
  if (auto castOp = dyn_cast<tpu::PoolAvg2DOp>(op)) {
    // tpu  handle pool_avg_2d only use count_include_padding.
    // set true
    auto rewriter = Builder(castOp.getContext());
    castOp->setAttr("param",
        tpu::PoolParam::get(
            castOp.param().kernel_h(),
            castOp.param().kernel_w(),
            castOp.param().padding_t(),
            castOp.param().padding_b(),
            castOp.param().padding_l(),
            castOp.param().padding_r(),
            castOp.param().pad_value(),
            castOp.param().stride_h(),
            castOp.param().stride_w(),
            castOp.param().do_relu(),
            rewriter.getBoolAttr(true),
            castOp.getContext()));
  }

  // special handling
  if (OpTy::getOperationName() == tpu::ReduceMeanOp::getOperationName()) {
    auto castOp = dyn_cast<tpu::ReduceMeanOp>(op);
    assert(castOp && "Expect ReduceMeanOp");

    std::vector<int64_t> axes;
    if (castOp.axes().hasValue()) {
      // Collect reduced axes
      for (auto val : castOp.axes().getValue())
        axes.push_back(val.cast<IntegerAttr>().getInt());

      // Calculate size of reduced axes from input dimensions
      auto type = castOp.input().getType().template cast<TensorType>();
      std::vector<int64_t> inputShapes(type.getShape());
      int64_t size = 1;
      for (auto dim : axes) {
        assert(static_cast<unsigned>(dim) < inputShapes.size() &&
               "Expect valid axis");
        size *= inputShapes[dim];
      }

      qscale[0] /= size;
    }
  }

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(nInputs);

  //
  // decompose into int8 multiplier and rshift
  //
  // find one rshift for all inputs, and multipliers for each inputs
  //   each qscale will be implemented by hardware as
  //   qscale[i] = multiplier / (1 << rshift)
  //   find a rshift, that put max(output) into range (64, 127)
  //

  // get the max abs(qscale)
  float max_qscale = 0.0;
  for (auto & q : qscale) {
    float p = q > 0.0 ? q : 0.0 - q;
    if (max_qscale < p) {
      max_qscale = p;
    }
  }

  int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(max_qscale);
  rshift->at(0) = (float)rshift_i8;
  LLVM_DEBUG(llvm::errs() << "  rshift = "
                          << std::to_string(rshift->at(0)) << "\n");
  for (unsigned i = 0; i < nInputs; ++i) {
    int8_t multiplier_i8 = findMultiplierI8FromQScaleAndRShift(qscale[i],
                                 rshift_i8);
    multiplier->at(i) = (float)multiplier_i8;
    LLVM_DEBUG(llvm::errs() << "  multiplier[" << i << "] = "
                            << std::to_string(multiplier->at(i)) << "\n");
    LLVM_DEBUG(llvm::errs() << "  qscale[" << i << "] = "
                            << std::to_string(qscale[i]) << "\n");
  }

  // add rshift and multiplier to weight
  StringRef storageType = "NONE";

  auto shape_rshift = std::vector<int64_t>{1};
  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape_rshift, storageType,
      wTF, wfV);
  op->setOperand(nInputs + 2, rshift_op);

  auto shape_multiplier = std::vector<int64_t>{nInputs};
  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape_multiplier, storageType,
      wTF, wfV);
  op->setOperand(nInputs + 3, multiplier_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

template<typename T>
static bool checkFloatNeedQuant(const std::vector<T> &data_v) {
  for (auto &data: data_v) {
    if (data != std::floor(data) || data > 127 || data < -127) {
      return true;
    }
  }
  return false;
}

template<typename OpTy>
LogicalResult quantizeInt8MultiplyConstOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I32");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;
  assert(nInputs == 2 && "support only 2 inputs multiply");
  // get thresholds
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
               << std::to_string(threshold_y) << "\n";);
  float threshold_x = 0;
  int const_idx = 0;

  for (unsigned i = 0; i < nInputs; ++i) {
    auto formerOp = op->getOperand(i).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(formerOp)) {
      const_idx = i;
      continue;
    }
    threshold_x = getOpThreshold(formerOp);
    LLVM_DEBUG(llvm::errs() << "  threshold_x = "
               << std::to_string(threshold_x) << "\n";);
  }

  auto const_opd = readAndDeleteWeightTensor<float>(op->getOperand(const_idx), wTF);
  std::vector<int64_t> const_shape;
  int64_t const_size;
  getTensorShapeAndSize(op->getOperand(const_idx), const_shape, const_size);
  assert(const_size == (int64_t)const_opd->size());

  auto max_elem = *std::max_element(const_opd->begin(), const_opd->end());
  //
  // determine the qscale
  //
  float qscale = 0;
  bool need_quant = checkFloatNeedQuant(*const_opd);
  if (need_quant) {
    qscale = max_elem * threshold_x / threshold_y / 127.0;
  } else {
    qscale = threshold_x / threshold_y;
  }

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(1);

  // create tensors for rshift and multiplier
  uint32_t multiplier_u32;
  int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(qscale,
                         &multiplier_u32, true);
  rshift->at(0) = static_cast<float>(rshift_i8);
  multiplier->at(0) = static_cast<float>(multiplier_u32);
  LLVM_DEBUG(llvm::errs()
             << "  rshift = "
             << std::to_string(rshift->at(0))
             << ", multiplier = "
             << std::to_string(multiplier->at(0)) << "\n");

  std::vector<float> quant_const(const_size, 0);
  if (need_quant) {
    for (int i = 0; i < const_size; i++) {
      float float_quant = floor((*const_opd)[i] * 127.0 / max_elem + 0.5);
      quant_const[i] = std::round(float_quant);
      if (quant_const[i] > 127)
        quant_const[i] = 127.0;
      if (quant_const[i] < -128)
        quant_const[i] = -128.0;
    }
  } else {
    for (int i = 0; i < const_size; i++) {
      quant_const[i] = (*const_opd)[i];
    }
  }

  // update op
  addWeightTensorAndUpdateWeightOp<float>(op->getOperand(const_idx),
      "quant", quant_const, const_shape, "INT8", wTF);

  // add rshift and multiplier to weight
  StringRef storageType = "NONE";
  auto shape = std::vector<int64_t>{1};

  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, storageType,
      wTF, wfV);
  op->setOperand(4, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape, storageType,
      wTF, wfV);
  op->setOperand(5, multiplier_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

template<typename OpTy>
LogicalResult quantizeInt8OpsWithSkip(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "NONE");

  TensorFile *wTF = getWeightTensorFile(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;
  for (unsigned i = 0; i < nInputs; ++i) {
    auto formerOp = op->getOperand(i);
    if (false == isa<tpu::LoadWeightOp>(formerOp.getDefiningOp())) {
      continue;
    }
    auto const_opd = readAndDeleteWeightTensor<float>(formerOp, wTF);
    std::vector<int64_t> const_shape;
    int64_t const_size;
    getTensorShapeAndSize(formerOp, const_shape, const_size);
    assert(const_size == (int64_t)const_opd->size());
    std::vector<float> quant_const(const_size, 0);
    for (int i = 0; i < const_size; i++) {
      quant_const[i] = std::floor((*const_opd)[i]);
    }
    addWeightTensorAndUpdateWeightOp<float>(formerOp,
      "quant", quant_const, const_shape, "INT8", wTF);
  }

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));
  return success();
}

template<typename OpTy>
LogicalResult quantizeInt8AddConstOps(Operation *op) {
  // duplicate from quantizeInt8MultiplyConstOps
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I32");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;
  assert(nInputs == 2 && "support only 2 inputs multiply");
  // get thresholds
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
      << std::to_string(threshold_y) << "\n";);
  float threshold_x = 0;
  int const_idx;

  for (unsigned i = 0; i < nInputs; ++i) {
    auto formerOp = op->getOperand(i).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(formerOp)) {
      const_idx = i;
      continue;
    }
    threshold_x = getOpThreshold(formerOp);
    LLVM_DEBUG(llvm::errs() << "  threshold_x = "
        << std::to_string(threshold_x) << "\n";);
  }

  auto const_opd = readAndDeleteWeightTensor<float>(op->getOperand(const_idx), wTF);
  std::vector<int64_t> const_shape;
  int64_t const_size;
  getTensorShapeAndSize(op->getOperand(const_idx), const_shape, const_size);
  assert(const_size == (int64_t)const_opd->size());

  auto max_elem = *std::max_element(const_opd->begin(), const_opd->end());
  //
  // determine the qscale
  //
  float qscale = threshold_x / threshold_y;

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(1);
  auto max_multiplier = max_elem * qscale;
  //
  // decompose into int8 mulitplier and rshift
  //
  uint32_t multiplier_u32;
  uint32_t multiplier_i8;
  auto shape_multiplier = std::vector<int64_t>{1};

  int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(qscale,
      &multiplier_u32, true, max_multiplier);
  std::vector<float> quant_const(const_size, 0);

  setOpQuantParamType(op, "RSHIFT_AND_M_I8");
  std::vector<float> qscales(nInputs);
  qscales[0] = qscale;
  qscales[1] = 127.0 / (float)max_elem;
  float max_qscale = *std::max_element(std::begin(qscales), std::end(qscales));
  if(max_qscale > 127){
    llvm::errs() << "[Warning!] qscale( "<< max_qscale <<") > max_multiplier (127)\n";
    llvm::errs() << "[Warning! set qscale 126.99]\n";
    max_qscale = 126.99;
  }
  rshift_i8 = findRShiftAndMultiplierFromQScale(max_qscale);
  multiplier_i8 = findMultiplierI8FromQScaleAndRShift(qscales[0], rshift_i8);

  rshift = std::make_unique<std::vector<float> >(nInputs);
  multiplier = std::make_unique<std::vector<float> >(nInputs);
  shape_multiplier = std::vector<int64_t>{nInputs};
  assert(const_idx == 1 && "weight must as second input");
  multiplier->at(0) = static_cast<float>(multiplier_i8);

  // later apply multipiler
  quant_const.assign(const_opd->begin(), const_opd->end());
  multiplier_i8 = findMultiplierI8FromQScaleAndRShift(qscales[1], rshift_i8);
  multiplier->at(1) = static_cast<float>(multiplier_i8);

  rshift->at(0) = static_cast<float>(rshift_i8);
  LLVM_DEBUG(llvm::errs()
      << "  rshift = "
      << std::to_string(rshift->at(0))
      << ", multiplier = "
      << std::to_string(multiplier->at(0)) << "\n");

  // update op
  addWeightTensorAndUpdateWeightOp<float>(op->getOperand(const_idx),
      "quant", quant_const, const_shape, "INT8", wTF);

  // add rshift and multiplier to weight
  StringRef storageType = "NONE";
  auto shape = std::vector<int64_t>{1};

  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, storageType,
      wTF, wfV);
  op->setOperand(4, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape_multiplier, storageType,
      wTF, wfV);
  op->setOperand(5, multiplier_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// default multiply Ops quantization method
/// for operations that has no weight, and input operands are
/// multiplied, eg. BroardcastMul, EltwiseMul, etc.
/// for multiple inputs op (broadcast mul, eltwise mul)
///   - first n operands are variadic
///   - 4 quant operands are following
///
template<typename OpTy>
LogicalResult quantizeInt8MultiplyOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I32");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;

  // get thresholds
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
               << std::to_string(threshold_y) << "\n";);
  std::vector<float> threshold_x(nInputs);
  for (unsigned i = 0; i < nInputs; ++i) {
    threshold_x[i] = getPreviousOpThreshold(op, i);
    LLVM_DEBUG(llvm::errs() << "  threshold_x[" << i << "] = "
               << std::to_string(threshold_x[i]) << "\n";);
  }

  //
  // determine the qscale
  //
  float threshold_prod = std::accumulate(
      threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
  float qscale = threshold_prod / threshold_y / 127.0;

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(1);

  //
  // decompose into int8 mulitplier and rshift
  //
  uint32_t multiplier_u32;
  int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(qscale,
                         &multiplier_u32, true);
  rshift->at(0) = static_cast<float>(rshift_i8);
  multiplier->at(0) = static_cast<float>(multiplier_u32);
  LLVM_DEBUG(llvm::errs()
             << "  rshift = "
             << std::to_string(rshift->at(0))
             << ", multiplier = "
             << std::to_string(multiplier->at(0)) << "\n");

  // add rshift and multiplier to weight
  StringRef storageType = "NONE";
  auto shape = std::vector<int64_t>{1};

  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, storageType,
      wTF, wfV);
  op->setOperand(4, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape, storageType,
      wTF, wfV);
  op->setOperand(5, multiplier_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// bypass Ops quantization method
/// which means threshold_x and threshold_y are the same
/// therefore no rescaling needed
/// operation handles on INT8 and pass through directly
///
LogicalResult quantizeInt8BypassOps(Operation *op) {
  assert(getOpQuant(op) == "INT8" || getOpQuant(op) == "UINT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "NONE");

  bool skip_checking = false;

  if (isa<tpu::InterpOp>(op)
      || isa<tpu::InputOp>(op)
      || isa<tpu::InstanceNormOp>(op)
      || isa<tpu::LrnOneOp>(op)
      || isa<tpu::LrnTwoOp>(op)
      || isa<tpu::LrnThreeOp>(op)
      || isa<tpu::LrnOp>(op)
      || isa<tpu::TransposeOp>(op)
      || isa<tpu::PermuteOp>(op)
      || isa<tpu::ROIPoolingOp>(op)
      || isa<tpu::SwapChannelOp>(op)
      || isa<tpu::CropOp>(op)
      || isa<tpu::SoftmaxOp>(op)
      || isa<tpu::PadOp>(op)
      || isa<tpu::PoolMax2DOp>(op)
      || isa<tpu::SoftmaxCpuOp>(op)
      || isa<tpu::CscOp>(op)
      || isa<tpu::ZeroMaskOp>(op)) {
    skip_checking = true;
  }

  //if (isa<tpu::CustomOp>(op) &&
  //    !cast<tpu::CustomOp>(op).quantifiable()) {
  //  skip_checking = true;
  //}

  if (!skip_checking) {
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_x - threshold_y > 0.001) {
      llvm::errs() << "QuantizeInt8 Bypass pattern, threshold not match"
                   << ", op name: " << getOpName(op)
                   << ", x = " << std::to_string(threshold_x)
                   << ", y = " << std::to_string(threshold_y) << "\n";
      assert(false);
    }
  }

  auto bSigned = (getOpQuant(op) == "INT8") ? IntegerType::Signed : IntegerType::Unsigned;
  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, bSigned));

  return success();
}

//===----------------------------------------------------------------------===//
// quantizeInt8 API
//===----------------------------------------------------------------------===//

#define DECLARE_QUANTIZE_INT8_BYPASS_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << "]\n";); \
    Operation *op = this->getOperation(); \
    return quantizeInt8BypassOps(op); \
  }

LogicalResult tpu::BroadcastMulOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8MultiplyOps<tpu::BroadcastMulOp>(op);
}

LogicalResult tpu::BroadcastAddOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::BroadcastAddOp>(op);
}

LogicalResult tpu::BroadcastSubOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::BroadcastSubOp>(op);
}

LogicalResult tpu::ConcatOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ConcatOp>(op);
}

LogicalResult tpu::Conv2DOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8ConvOps<tpu::Conv2DOp>(op, 2);
}

LogicalResult tpu::Conv3DOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8ConvOps<tpu::Conv3DOp>(op, 3);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::AbsOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CscOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CustomOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::DilateOp)

LogicalResult tpu::DeConv2DOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8ConvOps<tpu::DeConv2DOp>(op, 2);
}

LogicalResult tpu::EltwiseAddOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  bool ConstOpd = false;
  for (unsigned i = 0; i < 2; ++i) {
    auto formerOp = op->getOperand(i).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(formerOp)) {
      ConstOpd = true;
      break;
    }
  }
  if (this->quant_skip() == true) {
    return quantizeInt8OpsWithSkip<tpu::EltwiseAddOp>(op);
  }

  if (ConstOpd) {
    // yolo tiny v4 case
    return quantizeInt8AddConstOps<tpu::EltwiseAddOp>(op);
  } else {
    return quantizeInt8RescaleNoWeightOps<tpu::EltwiseAddOp>(op);
  }
}

LogicalResult tpu::EltwiseMaxOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::EltwiseMaxOp>(op);
}

LogicalResult tpu::EltwiseMinOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::EltwiseMinOp>(op);
}

LogicalResult tpu::EltwiseMulOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  bool hasConstOpd = false;
  for (unsigned i = 0; i < 2; ++i) {
    auto formerOp = op->getOperand(i).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(formerOp)) {
      hasConstOpd = true;
      break;
    }
  }
  if (this->quant_skip() == true) {
    return quantizeInt8OpsWithSkip<tpu::EltwiseMulOp>(op);
  }
  if (hasConstOpd) {
    return quantizeInt8MultiplyConstOps<tpu::EltwiseMulOp>(op);
  } else {
    return quantizeInt8MultiplyOps<tpu::EltwiseMulOp>(op);
  }
}

LogicalResult tpu::FullyConnectedOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8FullyConnectedOps(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::InputOp)

LogicalResult tpu::InterpOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto interpOp = cast<tpu::InterpOp>(op);

  std::string _coordinate_transformation_mode =
      interpOp.coordinate_transformation_mode().str();
  if (_coordinate_transformation_mode != "half_pixel") {
    std::string err_msg = "No support " + _coordinate_transformation_mode
                                        + " mode \n";
    llvm_unreachable(err_msg.c_str());
  }
  llvm::StringRef type = "NONE";
  interpOp.setOpQuantMode(type);
  return success();
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::InstanceNormOp)

LogicalResult tpu::LeakyReluOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LeakyReluOps(op);
}

LogicalResult tpu::PReluOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8PReluOps(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::GruOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LrnOneOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LrnTwoOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LrnThreeOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LrnOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LstmOp)

LogicalResult tpu::MishOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::MishOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PadOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::TileOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::TileInterpOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PermuteOp)

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PixelShuffleOp)

LogicalResult tpu::ClipOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ClipOp>(op);
}

LogicalResult tpu::PoolAvg2DOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::PoolAvg2DOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMax3DOp)

LogicalResult tpu::PowerOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //Operation *op = this->getOperation();
  //return quantizeInt8LutOps<tpu::PowerOp>(op);
  // TODO:
  assert(false);
  return failure();
}

LogicalResult tpu::ReciprocalOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::ReciprocalOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ShuffleChannelOp)

LogicalResult tpu::ScaleLutOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8ScaleLutOps(op);
}

LogicalResult tpu::SigmoidOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SigmoidOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SliceOp)

LogicalResult tpu::SoftPlusOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SoftPlusOp>(op);
}

LogicalResult tpu::SqrtOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SqrtOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SoftmaxOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SquareOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::QuadraticSumOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::MatMulOp)

LogicalResult tpu::TanHOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::TanHOp>(op);
}

LogicalResult tpu::ExpOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::ExpOp>(op);
}

LogicalResult tpu::ReduceMeanOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ReduceMeanOp>(op);
}

LogicalResult tpu::ReduceMaxOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ReduceMaxOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::UpsampleOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ZeroMaskOp)

#define DECLARE_QUANTIZE_INT8_DISABLED_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << ", disabled]\n";); \
    assert(false); \
    return failure(); \
  }
/// This Ops does not support quantizie
/// their quant interface are kept for holding threshold only
DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::ReshapeOp)

} // namespace mlir
