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
LogicalResult quantizeInt8ConvOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);
  auto convOp = cast<OpTy>(op);

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
  int64_t isz = filterSize / oc;

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
    quantizeWeightInt8Multiplier(filter->data(), bias ? bias->data() : nullptr,
                               oc, isz, threshold_y, threshold_x,
                               new_filter->data(), bias ? new_bias->data() : nullptr,
                               rshift_per_channel->data(),
                               multiplier_per_channel->data());

  } else {
    assert(0);
  }

  // update op
  addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
      "quant", *new_filter, filterShape, "INT8", wTF);
  if (bias) {
    // for per_channel quant, bias store as INT32, per layer use INT16
    StringRef storageType = isOpQuantPerchannel(op) ? "INT32" : "INT16";
    addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(2),
        "quant", *new_bias, biasShape, storageType, wTF);
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
  setOpResultType(op, StandardTypes::Integer, 8);

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
  setOpQuantParamType(op, "RSHIFT_ONLY");

  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);
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

  // get threshold
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
               << ", threshold_y = "<< std::to_string(threshold_y)
               << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

  // quantization
  quantizeWeightInt8PerLayer(filter->data(), bias ? bias->data() : nullptr,
                             n, k, threshold_y, threshold_x,
                             new_filter->data(), bias ? new_bias->data() : nullptr,
                             rshift->data());

  // update op
  addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(1),
      "quant", *new_filter, filterShape, "INT8", wTF);
  if (bias) {
    // for per layer, bias use INT16
    addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(2),
        "quant", *new_bias, biasShape, "INT16", wTF);
  }

  // add rshift to weight
  auto shape = std::vector<int64_t>{1};
  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, "NONE",
      wTF, wfV);
  fcOp.setOperand(5, rshift_op);

  setOpResultType(op, StandardTypes::Integer, 8);

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
  Value *wfV = getWeightFileValue(op);
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
    multiplier_pos[0] = 0;
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

  setOpResultType(op, StandardTypes::Integer, 8);

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
  Value *wfV = getWeightFileValue(op);
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
    multiplier_pos[0] = 0;
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
  float max_abs_negative_qscale = fabs(qscale_pos * negative_slope_weight->at(0));
  for (int i = 1; i < c; ++i) {
    if (fabs(qscale_pos * negative_slope_weight->at(i)) >
        max_abs_negative_qscale) {
      max_abs_negative_qscale = fabs(qscale_pos * negative_slope_weight->at(i)) >
                                max_abs_negative_qscale;
    }
  }
  uint32_t uint_multiplier_neg = 0;
  float rshift_tmp = (float)findRShiftAndMultiplierFromQScale(
      fabs(max_abs_negative_qscale), &uint_multiplier_neg, false);
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
  auto neg_slope_op = addWeightTensorAndCreateWeightOp<float>(
      op, "negative_slope", new_negative_slope, neg_slope_shape, storageType,
      wTF, wfV);
  preluOp.setOperand(1, neg_slope_op);
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

  setOpResultType(op, StandardTypes::Integer, 8);

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
        float lutOutput = 1.0 /(index) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                            ? 127
                            : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      }else if(OpTy::getOperationName()=="tpu.sqrt"){
        float lutOutput = pow(index,0.5) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                          ? 127
                          : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      }else if(OpTy::getOperationName()=="tpu.sigmoid"){
        index = -lutInput * threshold_x / 127.0;
        float lutOutput = 1.0 / (1 + std::exp(index)) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                          ? 127
                          : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * table_hw + idx] = lutOutputI32;
      }else{
        assert(false&&"not support now");
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

  setOpResultType(op, StandardTypes::Integer, 8);

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
///
template<typename OpTy>
LogicalResult quantizeInt8RescaleNoWeightOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;

  bool bypass = true;
  float bypass_eps = 1e-5;
  // get thresholds
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
                          << std::to_string(threshold_y) << "\n";);
  std::vector<float> threshold_x(nInputs);
  for (unsigned i = 0; i < nInputs; ++i) {
    threshold_x[i] = getPreviousOpThreshold(op, i);
    LLVM_DEBUG(llvm::errs() << "  threshold_x[" << i << "] = "
               << std::to_string(threshold_x[i]) << "\n";);
    if (fabs(threshold_y - threshold_x[i]) > bypass_eps) {
      bypass = false;
    }
  }
  if (bypass && OpTy::getOperationName() != "tpu.pool_avg_2d") {
    // leave quant_rshift and quant_mulitplier as NoneOp to indicate bypass
    LLVM_DEBUG(llvm::errs() << " < " << getOpName(op)
                            << ",  quantization bypassed\n";);
    setOpQuantParamType(op, "NONE");
    setOpResultType(op, StandardTypes::Integer, 8);

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

  // special handlings
  if (OpTy::getOperationName() == "tpu.pool_avg_2d") {
    assert(nInputs);
    auto castOp = dyn_cast<tpu::PoolAvg2DOp>(op);
    assert(castOp);
    int kh = castOp.param().kernel_h().getValue().getLimitedValue();
    int kw = castOp.param().kernel_w().getValue().getLimitedValue();
    qscale[0] = qscale[0] / (kh * kw);
  }

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(nInputs);

  //
  // decompose into int8 mulitplier and rshift
  //
  // find one rshift for all inputs, and multipliers for each inputs
  //   each qscale will be implemented by hardware as
  //   qscale[i] = multiplier / (1 << rshift)
  //   find a rshift, that put max(output) into range (64, 127)
  //
  float max_qscale = *std::max_element(std::begin(qscale), std::end(qscale));
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

  setOpResultType(op, StandardTypes::Integer, 8);

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
  Value *wfV = getWeightFileValue(op);

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

  setOpResultType(op, StandardTypes::Integer, 8);

  return success();
}

///
/// bypass Ops quantization method
/// which means threshold_x and threshold_y are the same
/// therefore no rescaling needed
/// operation handles on INT8 and pass through directly
///
LogicalResult quantizeInt8BypassOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "NONE");

  bool skip_checking = false;
  if (isa<tpu::InputOp>(op)
      || isa<tpu::PreprocessOp>(op)
      || isa<tpu::TransposeOp>(op)) {
    skip_checking = true;
  }

  if (!skip_checking) {
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_x != threshold_y) {
      llvm::errs() << "QuantizeInt8 Bypass pattern, threshold not match"
                   << ", x = " << std::to_string(threshold_x)
                   << ", y = " << std::to_string(threshold_y) << "\n";
      assert(false);
    }
  }

  setOpResultType(op, StandardTypes::Integer, 8);

  return success();
}

//===----------------------------------------------------------------------===//
// quantizeInt8 API
//===----------------------------------------------------------------------===//

#define DECLARE_QUANTIZE_INT8_BYPASS_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << "]\n"; \
    Operation *op = this->getOperation(); \
    return quantizeInt8BypassOps(op); \
  }

LogicalResult tpu::BroadcastMulOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8MultiplyOps<tpu::BroadcastMulOp>(op);
}

LogicalResult tpu::ConcatOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ConcatOp>(op);
}

LogicalResult tpu::Conv2DOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8ConvOps<tpu::Conv2DOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CropOp)

LogicalResult tpu::DeConv2DOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8ConvOps<tpu::DeConv2DOp>(op);
}

LogicalResult tpu::EltwiseAddOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::EltwiseAddOp>(op);
}

LogicalResult tpu::EltwiseMaxOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::EltwiseMaxOp>(op);
}

LogicalResult tpu::EltwiseMulOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8MultiplyOps<tpu::EltwiseMulOp>(op);
}

LogicalResult tpu::FullyConnectedOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8FullyConnectedOps(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::InputOp)

LogicalResult tpu::LeakyReluOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8LeakyReluOps(op);
}

LogicalResult tpu::PReluOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8PReluOps(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PreprocessOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PermuteOp)

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PixelShuffleOp)

LogicalResult tpu::PoolAvg2DOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::PoolAvg2DOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMax2DOp)

LogicalResult tpu::PowerOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //Operation *op = this->getOperation();
  //return quantizeInt8LutOps<tpu::PowerOp>(op);
  // TODO:
  assert(false);
  return failure();
}

LogicalResult tpu::ReciprocalOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::ReciprocalOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReluOp)

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ShuffleChannelOp)

LogicalResult tpu::SigmoidOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SigmoidOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SliceOp)

LogicalResult tpu::SqrtOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SqrtOp>(op);
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SwapChannelOp)

LogicalResult tpu::TanHOp::quantizeInt8() {
  llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //Operation *op = this->getOperation();
  //return quantizeInt8LutOps<tpu::TanHOp>(op);
  // TODO:
  assert(false);
  return failure();
}

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::TransposeOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::UpsampleOp)

#define DECLARE_QUANTIZE_INT8_DISABLED_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << ", disabled]\n"; \
    assert(false); \
    return failure(); \
  }
/// This Ops does not support quantizie
/// their quant interface are kept for holding threshold only
DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::ReshapeOp)

} // namespace mlir
