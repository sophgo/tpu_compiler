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

inline static float INT8(float data) {
  data = std::floor(data + 0.5);
  return std::max(std::min(data, 127.0f), -128.0f);
}

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
  int bitwidth = filterOp.quant_bitwidth();

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
        filter_threshold, bitwidth);

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
  int batch_high, batch_low, m, k, n;
  parseFullyConnectedParam<tpu::FullyConnectedOp>(op, batch_high, batch_low, m,
                                                  k, n);
  int batch = batch_high * batch_low;
  int64_t filterSize = 0;
  std::vector<int64_t> filterShape;
  auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
  getTensorShapeAndSize(fcOp.filter(), filterShape, filterSize);
  assert(filterSize == batch * k * n);
  float * filter_data = filter->data();

  // get bias tensor
  std::unique_ptr<std::vector<float>> bias = nullptr;
  std::vector<int64_t> biasShape;
  int64_t biasSize = 0;
  if (!isTensorNone(fcOp.bias())) {
    bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
    getTensorShapeAndSize(fcOp.bias(), biasShape, biasSize);
    assert(biasSize == batch * n);
  }

  // create new tensors
  auto new_filter = std::make_unique<std::vector<float>>(filterSize);
  std::unique_ptr<std::vector<float>> new_bias = nullptr;
  if (bias) {
    new_bias = std::make_unique<std::vector<float>>(biasSize);
  }

  // create tensors for rshift and multiplier
  auto rshift_per_batch = std::make_unique<std::vector<float>>(batch);
  auto multiplier_per_batch = std::make_unique<std::vector<float>>(batch);
  // get threshold
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op)
                          << ", threshold_y = " << std::to_string(threshold_y)
                          << ", threshold_x = " << std::to_string(threshold_x)
                          << "\n";);

  // quantization
  quantizeWeightInt8ForFC(
      filter_data, bias ? bias->data() : nullptr, batch, n, k, threshold_y,
      threshold_x, new_filter->data(), bias ? new_bias->data() : nullptr,
      rshift_per_batch->data(), multiplier_per_batch->data());

  // update op
  addWeightTensorAndUpdateWeightOp<float>(
      fcOp.getOperand(1), "quant", *new_filter, filterShape, "INT8", wTF);

  if (bias) {
    // qdm mode, bias use INT32
    addWeightTensorAndUpdateWeightOp<float>(fcOp.getOperand(2), "quant",
                                            *new_bias, biasShape, "INT32", wTF);
  }

  // add rshift to weight
  auto shape = std::vector<int64_t>{batch};
  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift_per_batch, shape, "NONE", wTF, wfV);
  fcOp.setOperand(5, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier_per_batch, shape, "NONE", wTF, wfV);
  fcOp.setOperand(6, multiplier_op);

  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

static void quantize_fraction(float x, float y, int &rshift_width,
                              int &x_quantized) {
  float y_ceiling = 256.0 / x * y;
  rshift_width = 0;
  x_quantized = 0;
  float y_quantized = 1.0;
  while ((y_quantized * 2) < y_ceiling && rshift_width < 15) {
    rshift_width += 1;
    y_quantized = (float)(1 << rshift_width);
  }
  x_quantized = (int)std::floor((x / y) * y_quantized + 0.5);
}

LogicalResult quantizeInt8LrnOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  auto lrnOp = cast<tpu::LrnOp>(op);
  auto rewriter = Builder(lrnOp.getContext());

  const int NPU_NUM = MInfo::lane_num;
  const int TABLE_H_INT8 = 16;
  const int TABLE_W_INT8 = 16;
  const int TABLE_HW_INT8 = (TABLE_H_INT8 * TABLE_W_INT8);
  const int TBL_SHAPE_INT8 = (TABLE_HW_INT8 * NPU_NUM);

  uint32_t local_size = lrnOp.local_size();
  float alpha = lrnOp.alpha().convertToFloat();
  float beta = lrnOp.beta().convertToFloat();
  float k = lrnOp.k().convertToFloat();
  std::vector<float> threhsold_parts;
  arrayAttrToVector(lrnOp.threshold_parts().getValue(), threhsold_parts);

  float sq_thy = threhsold_parts[0];
  float sumsq_thy = threhsold_parts[1];
  float scale_thy = threhsold_parts[2];
  float threshold_x = getPreviousOpThreshold(op);
  float threshold_y = getOpThreshold(op);
  // quant x and rshift
  int quant_x0, sum_rshift, quant_x1, lrn_rshift;
  quantize_fraction(sq_thy, sumsq_thy, sum_rshift, quant_x0);
  quantize_fraction(threshold_x * scale_thy, threshold_y * 256.0, lrn_rshift,
                    quant_x1);
  lrnOp->setAttr("sum_rshift", rewriter.getI32IntegerAttr(sum_rshift));
  lrnOp->setAttr("quant_data0", rewriter.getI32IntegerAttr(quant_x0));
  lrnOp->setAttr("lrn_rshift", rewriter.getI32IntegerAttr(lrn_rshift));
  lrnOp->setAttr("quant_data1", rewriter.getI32IntegerAttr(quant_x1));
  // sq table
  std::vector<float> sq_table(TBL_SHAPE_INT8);

  for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
    float lut_input = threshold_x / 128.0 * idx;
    float lut_output = std::pow(lut_input, 2) * 256.0 / sq_thy;
    lut_output = lut_output * alpha / local_size;
    lut_output = std::floor(lut_output + 0.5);
    if (lut_output > 255.0) {
      lut_output = 255.0;
    }
    for (int n = 0; n < NPU_NUM; n++) {
      sq_table[n * TABLE_HW_INT8 + idx] = lut_output;
    }
  }

  // power table
  std::vector<float> power_table(TBL_SHAPE_INT8);

  for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
    float lut_input = (float)idx / (256.0 / sumsq_thy);
    float lut_output = std::pow(lut_input + k, -beta);
    lut_output = lut_output * (256.0 / scale_thy);
    lut_output = std::floor(lut_output + 0.5);
    if (lut_output > 255.0) {
      lut_output = 255.0;
    }
    for (int n = 0; n < NPU_NUM; n++) {
      power_table[n * TABLE_HW_INT8 + idx] = lut_output;
    }
  }

  std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_INT8, TABLE_W_INT8};
  // sq weight
  auto sq_weight_op = addWeightTensorAndCreateWeightOp<float>(
      op, "sq_gen_weight", sq_table, weightShape, "UINT8", wTF, wfV);
  lrnOp.setOperand(1, sq_weight_op);

  // power weight
  auto power_weight_op = addWeightTensorAndCreateWeightOp<float>(
      op, "power_gen_weight", power_table, weightShape, "UINT8", wTF, wfV);
  lrnOp.setOperand(2, power_weight_op);

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
  std::vector<float> new_negative_slope(neg_slope_size);

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
  addWeightTensorAndUpdateWeightOp<float>(preluOp.filter(),
      "negative_slope", new_negative_slope, neg_slope_shape, "INT8", wTF);
  auto rshift_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_pos", rshift_pos, shape, "NONE", wTF, wfV);
  preluOp.setOperand(6, rshift_pos_op);

  auto multiplier_pos_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier_pos", multiplier_pos, shape, "NONE", wTF,
      wfV);
  preluOp.setOperand(7, multiplier_pos_op);

  auto rshift_neg_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift_neg", rshift_neg, shape, "NONE", wTF, wfV);
  preluOp.setOperand(8, rshift_neg_op);

  setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

///
/// Lut Ops quantization method
///
template <typename OpTy>
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

  //<! 1880v2 hw int8 config
  int table_h = 16;
  int table_w = 16;
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;

  std::vector<float> y0_table(tbl_shape, 0);

  // input: 0~127, -128~ -1, Y=1/(1+EXP(-X*thx/128)) * 128/thy
  // output:0~127, negative is invalid
  for (int n = 0; n < npu_num; n++) {
    for (int idx = 0; idx < table_hw; ++idx) {
      char lutInput = static_cast<char>(idx);
      float index = lutInput * threshold_x / 127.0;
      float lutOutput = 127.0f;
      if (OpTy::getOperationName() == "tpu.reciprocal") {
        if (index != 0) {
          lutOutput = 1.0 / (index)*127.0 / threshold_y;
        }
      } else if (OpTy::getOperationName() == "tpu.sqrt") {
        lutOutput = pow(index, 0.5) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.sigmoid") {
        auto sigmoidOp = cast<tpu::SigmoidOp>(op);
        float scale = sigmoidOp.scale().convertToFloat();
        float bias = sigmoidOp.bias().convertToFloat();
        index = -lutInput * threshold_x / 127.0;
        lutOutput = (scale / (1 + std::exp(index)) + bias) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.log") {
        index = lutInput * threshold_x / 127.0;
        lutOutput = std::log(index) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.swish") {
        index = lutInput * threshold_x / 127.0;
        lutOutput = index / (1 + std::exp(-index)) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.tanh") {
        index = lutInput * threshold_x / 127.0;
        lutOutput = std::tanh(index) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.elu") {
        index = lutInput * threshold_x / 127.0;
        lutOutput = ((index >= 0) ? index : (std::exp(index) - 1)) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.exp") {
        auto expOp = cast<tpu::ExpOp>(op);
        float scale = expOp.scale().convertToFloat();
        float bias = expOp.bias().convertToFloat();
        index = lutInput * threshold_x / 127.0;
        lutOutput = (scale * std::exp(index) + bias)* 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.mish") {
        index = lutInput * threshold_x / 127.0;
        lutOutput = my_mish_activate(index) * 127.0 / threshold_y;
      } else if (OpTy::getOperationName() == "tpu.softplus") {
        auto spOp = cast<tpu::SoftPlusOp>(op);
        float scale = spOp.scale().convertToFloat();
        float bias = spOp.bias().convertToFloat();
        index = lutInput * threshold_x / 127.0;
        lutOutput = (scale * logf(expf(index) + 1) + bias) * 127.0 / threshold_y;
      } else {
        assert(false && "not support now");
      }
      y0_table[n * table_hw + idx] = INT8(lutOutput);
    }
  }

  // update op
  auto shape = std::vector<int64_t>{1, npu_num, table_h, table_w};
  StringRef storageType = "INT8";
  auto y0_table_op = addWeightTensorAndCreateWeightOp<float>(
      op, "y0_table", y0_table, shape, storageType, wTF, wfV);
  lutOp.setOperand(1, y0_table_op);

  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

// input 0~255 => output -128~127
LogicalResult quantizeInt8ScaleLutOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);
  auto castOp = cast<tpu::ScaleLutOp>(op);
  auto input_shape = getTensorShape(castOp.getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);
  std::vector<float> scale;
  std::vector<float> bias;
  arrayAttrToVector(castOp.scale(), scale);
  arrayAttrToVector(castOp.bias(), bias);
  assert(scale.size() >= (uint64_t)c);
  assert(bias.size() >= (uint64_t)c);

  int table_h = 16;
  int table_w = 16;
  int table_hw = table_h * table_w;
  int table_size = c * table_hw;
  std::vector<float> table(table_size, 0);
  for (int i = 0; i < c; i++) {
    for (int idx = 0; idx < table_hw; ++idx) {
      table[i * table_hw + idx] = INT8(idx * scale[i] + bias[i]);
    }
  }
  auto shape = std::vector<int64_t>{1, c, table_h, table_w};
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
             && OpTy::getOperationName() != "tpu.eltwise_max"
             && OpTy::getOperationName() != "tpu.eltwise_min"
             && OpTy::getOperationName() != "tpu.eltwise_add"
             && OpTy::getOperationName() != "tpu.reduce_mean"
             && OpTy::getOperationName() != "tpu.reduce_min"
             && OpTy::getOperationName() != "tpu.reduce_sum"
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

LogicalResult quantizeInt8ConcatOps(Operation *op) {
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  float threshold_y = getOpThreshold(op);
  auto concatOp = cast<tpu::ConcatOp>(op);
  unsigned nInputs = concatOp.getNumInputs();
  auto rshift = std::make_unique<std::vector<float>>(nInputs, 0.0f);
  auto multiplier = std::make_unique<std::vector<float>>(nInputs, 1.0f);
  for (unsigned i = 0; i < nInputs; ++i) {
    float threshold_x = getPreviousOpThreshold(op, i);
    float qscale = threshold_x / threshold_y;
    if (fabs(threshold_y - threshold_x) <= 1e-5) {
      qscale = 1.0f;
    }
    if (qscale != 1.0f) {
      uint32_t multiplier_u32;
      int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(qscale, &multiplier_u32);
      rshift->at(i) = (float)rshift_i8;
      multiplier->at(i) = (float)multiplier_u32;
    }
  }
  // add rshift and multiplier to weight
  StringRef storageType = "NONE";

  auto shape_quant = std::vector<int64_t>{nInputs};
  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape_quant, storageType, wTF, wfV);
  op->setOperand(nInputs + 2, rshift_op);
  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape_quant, storageType, wTF, wfV);
  op->setOperand(nInputs + 3, multiplier_op);

  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

template<typename T>
static bool checkFloatNeedQuant(const std::vector<T> &data_v) {
  for (auto &data: data_v) {
    if (data != (T)INT8(data)) {
      return true;
    }
  }
  return false;
}

template <typename OpTy>
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
    LLVM_DEBUG(llvm::errs() << "  threshold_x = " << std::to_string(threshold_x)
                            << "\n";);
  }
  auto weightOp = op->getOperand(const_idx);
  auto const_opd = readAndDeleteWeightTensor<float>(weightOp, wTF);
  std::vector<int64_t> const_shape;
  int64_t const_size;
  getTensorShapeAndSize(weightOp, const_shape, const_size);
  assert(const_size == (int64_t)const_opd->size());

  auto max_elem = *std::max_element(
      const_opd->begin(), const_opd->end(),
      [](float a, float b) { return (std::abs(a) < std::abs(b)); });
  max_elem = std::abs(max_elem);
  //
  // determine the qscale
  //
  float qscale = 0;
  bool need_quant = checkFloatNeedQuant(*const_opd);
  if ((threshold_x / threshold_y) >= 1.0) {
    need_quant = true;
  }
  if (need_quant) {
    qscale = max_elem * threshold_x / threshold_y / 127.0;
  } else {
    qscale = threshold_x / threshold_y;
  }

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float>>(1);
  auto multiplier = std::make_unique<std::vector<float>>(1);

  // create tensors for rshift and multiplier
  uint32_t multiplier_u32;
  int8_t rshift_i8 =
      findRShiftAndMultiplierFromQScale(qscale, &multiplier_u32, true);
  rshift->at(0) = static_cast<float>(rshift_i8);
  multiplier->at(0) = static_cast<float>(multiplier_u32);
  LLVM_DEBUG(llvm::errs() << "  rshift = " << std::to_string(rshift->at(0))
                          << ", multiplier = "
                          << std::to_string(multiplier->at(0)) << "\n");

  std::vector<float> quant_const(const_size, 0);
  if (need_quant) {
    for (int i = 0; i < const_size; i++) {
      float float_quant = (*const_opd)[i] * 127.0 / max_elem;
      quant_const[i] = INT8(float_quant);
    }
  } else {
    for (int i = 0; i < const_size; i++) {
      quant_const[i] = (*const_opd)[i];
    }
  }

  // update op
  addWeightTensorAndUpdateWeightOp<float>(weightOp, "quant", quant_const,
                                           const_shape, "INT8", wTF);

  // add rshift and multiplier to weight
  StringRef storageType = "NONE";
  auto shape = std::vector<int64_t>{1};

  auto rshift_op = addWeightTensorAndCreateWeightOp<float>(
      op, "rshift", *rshift, shape, storageType, wTF, wfV);
  op->setOperand(4, rshift_op);

  auto multiplier_op = addWeightTensorAndCreateWeightOp<float>(
      op, "multiplier", *multiplier, shape, storageType, wTF, wfV);
  op->setOperand(5, multiplier_op);

  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));

  return success();
}

template<typename OpTy>
LogicalResult quantizeInt8AddConstOps(Operation *op) {
  // duplicate from quantizeInt8MultiplyConstOps
  assert(getOpQuant(op) == "INT8");
  // support per-tensor only for now
  setOpQuantPerchannel(op, false);
  // use rshift and INT8 multiplier
  setOpQuantParamType(op, "RSHIFT_AND_M_I8");

  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // get operands
  const unsigned nInputs = op->getNumOperands() - 4;
  assert(nInputs == 2 && "support only 2 inputs multiply");

  // get threshold of opd 0
  float threshold_y = getOpThreshold(op);
  LLVM_DEBUG(llvm::errs() << " > " << getOpName(op) << ", threshold_y = "
      << std::to_string(threshold_y) << "\n";);
  auto opd0 = op->getOperand(0).getDefiningOp();
  assert(!isa<tpu::LoadWeightOp>(opd0));
  float threshold_x = getOpThreshold(opd0);

  // get max element of opd 1
  int const_idx = 1;
  int64_t const_size;
  std::vector<int64_t> const_shape;
  auto weightOp = op->getOperand(const_idx);
  auto const_opd = readAndDeleteWeightTensor<float>(weightOp, wTF);
  getTensorShapeAndSize(weightOp, const_shape, const_size);
  assert(const_size == (int64_t)const_opd->size());

  auto max_elem = *std::max_element(const_opd->begin(), const_opd->end(),
                                    [](float a, float b) {
                                      return (std::abs(a) < std::abs(b));
                                    });
  max_elem = std::abs(max_elem);

  std::vector<float> quant_const(const_size, 0);
  for (int i = 0; i < const_size; i++) {
    float float_quant = (*const_opd)[i] * 127.0 / max_elem;
    quant_const[i] = INT8(float_quant);
  }
  // update op
  addWeightTensorAndUpdateWeightOp<float>(weightOp,
      "quant", quant_const, const_shape, "INT8", wTF);
  //
  // determine the qscale
  //
  float qscale[2];
  qscale[0] = threshold_x / threshold_y;
  qscale[1] = max_elem / threshold_y;
  float max_qscale = std::max(qscale[0], qscale[1]);

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1);
  auto multiplier = std::make_unique<std::vector<float> >(2);

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
  double threshold_prod = std::accumulate(
      threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
  float qscale = threshold_prod / threshold_y / 127.0;

  // create tensors for rshift and multiplier
  auto rshift = std::make_unique<std::vector<float> >(1, 0.0f);
  auto multiplier = std::make_unique<std::vector<float> >(1, 0.0f);

  //
  // decompose into int8 mulitplier and rshift
  //
  if (std::abs(qscale - 1.0f) > 1e-5) {
    uint32_t multiplier_u32;
    int8_t rshift_i8 = findRShiftAndMultiplierFromQScale(qscale,
                          &multiplier_u32, true);
    rshift->at(0) = static_cast<float>(rshift_i8);
    multiplier->at(0) = static_cast<float>(multiplier_u32);
  }
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
      || isa<tpu::ReduceL2Op>(op)
      || isa<tpu::ROIPoolingOp>(op)
      || isa<tpu::SoftmaxOp>(op)
      || isa<tpu::LayerNormOp>(op)
      || isa<tpu::StdOp>(op)
      || isa<tpu::ZeroMaskOp>(op)
      || isa<tpu::SoftmaxCpuOp>(op)
      || isa<tpu::CscOp>(op)
      || isa<tpu::GruOp>(op)
      || isa<tpu::LstmOp>(op)
      || isa<tpu::ArgMaxOp>(op)
      || isa<tpu::EmbeddingOp>(op)
      || isa<tpu::PoolMaskOp>(op)) {
    skip_checking = true;
  }

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
  return quantizeInt8ConcatOps(op);
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

LogicalResult tpu::InterpOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto interpOp = cast<tpu::InterpOp>(op);

  llvm::StringRef type = "NONE";
  if (interpOp.coordinate_transformation_mode().startswith("nearest")) {
    type = "INT8";
    setOpResultType(op->getResult(0),
                    IntegerType::get(op->getContext(), 8, IntegerType::Signed));
  }
  interpOp.setOpQuantMode(type);
  return success();
}

LogicalResult tpu::ReflectionPadOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
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
        op, "_select_" + std::to_string(pad_idx), select, shape, "INT8", wTF,
        wfV);
    op->setOperand(pad_idx + 1, select_op);
    pad_idx++;
  }
  setOpResultType(op->getResult(0),IntegerType::get(op->getContext(), 8, IntegerType::Signed));
  return success();
}

LogicalResult tpu::LrnOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LrnOps(op);
}

LogicalResult tpu::LeakyReluOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LeakyReluOps(op);
}

LogicalResult tpu::LogOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::LogOp>(op);
}

LogicalResult tpu::PReluOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8PReluOps(op);
}

LogicalResult tpu::MishOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::MishOp>(op);
}

LogicalResult tpu::MatMulOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8MultiplyOps<tpu::MatMulOp>(op);
}

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

LogicalResult tpu::SwishOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::SwishOp>(op);
}

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

LogicalResult tpu::TanHOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::TanHOp>(op);
}

LogicalResult tpu::EluOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8LutOps<tpu::EluOp>(op);
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

LogicalResult tpu::ReduceMinOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ReduceMinOp>(op);
}

LogicalResult tpu::ReduceSumOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  return quantizeInt8RescaleNoWeightOps<tpu::ReduceSumOp>(op);
}

LogicalResult tpu::ArgMaxOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
                << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();
  quantizeInt8BypassOps(op);
  setOpResultType(op->getResult(0), FloatType::getF32(getContext()));
  return success();
}

LogicalResult tpu::EmbeddingOp::quantizeInt8() {
  LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  Operation *op = this->getOperation();

  auto castOp = cast<tpu::EmbeddingOp>(op);
  TensorFile *wTF = getWeightTensorFile(op);

  int64_t tableSize;
  std::vector<int64_t> tableShape;
  std::unique_ptr<std::vector<float> > table;
  table = readAndDeleteWeightTensor<float>(op->getOperand(1), wTF);
  getTensorShapeAndSize(op->getOperand(1), tableShape, tableSize);
  // create new tensors
  auto new_table = std::make_unique<std::vector<float> >(tableSize);
  float threshold_y = getOpThreshold(op);
  float scale = 127.0 / threshold_y;
  auto src_ptr = table->data();
  auto dst_ptr = new_table->data();
  for (int i = 0; i < tableSize; i++) {
    dst_ptr[i] = INT8(src_ptr[i] * scale);
  }
  addWeightTensorAndUpdateWeightOp<float>(castOp.table(),
      "quant", *new_table, tableShape, "INT8", wTF);
  setOpResultType(op->getResult(0),
                  IntegerType::get(op->getContext(), 8, IntegerType::Signed));
  return success();
}

/*
These Ops does not do quantization, need threshold_x == thrshold_y
*/
#define DECLARE_QUANTIZE_INT8_BYPASS_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << "]\n";); \
    Operation *op = this->getOperation(); \
    return quantizeInt8BypassOps(op); \
  }

DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::AbsOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CropOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CscOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::CustomOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::DilateOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::GruOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::InputOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::InstanceNormOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LayerNormOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::LstmOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PadOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PermuteOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PixelShuffleOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMax2DOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMax3DOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::PoolMaskOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::QuadraticSumOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReduceL2Op)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReluOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReorgOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ROIPoolingOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ReverseOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ShuffleChannelOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SwapChannelOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SliceOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SoftmaxOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SoftmaxCpuOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::SquareOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::StdOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::TileOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::UpsampleOp)
DECLARE_QUANTIZE_INT8_BYPASS_METHOD(tpu::ZeroMaskOp)

/*
This Ops does not support quantizie
their quant interface are kept for holding threshold only
*/
#define DECLARE_QUANTIZE_INT8_DISABLED_METHOD(OP) \
  LogicalResult OP::quantizeInt8() { \
    LLVM_DEBUG(llvm::errs() << "quantizeInt8: " << getOperationName() \
                 << " [" << getOpName() << ", disabled]\n";); \
    assert(false); \
    return failure(); \
  }

DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::ReshapeOp)
DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::LrnOneOp)
DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::LrnTwoOp)
DECLARE_QUANTIZE_INT8_DISABLED_METHOD(tpu::LrnThreeOp)

} // namespace mlir
