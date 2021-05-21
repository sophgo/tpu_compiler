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
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "tpuc/SimpleAnalysis.h"

using namespace mlir;

template <typename OpTy>
uint64_t SimpleConv2DMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details,
    int batch_size) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw,
      pad_value;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(), n,
                 ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt,
                 pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, pad_value);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;

  if (ic > 4095) {
    // TODO, need to support deep fusion conv with ic > 4095
    return MInfo::lmem_per_lane + 1;
  }
  // filter working size *2 for double buffer
  if (g == 1) {
    // for non-dw conv, assuming oc_step = lane_num
    int oc_step = MInfo::lane_num;
    filterSizePerLane = MInfo::getSizePerLane(ic, oc_step, kh, kw, false) * 2;
  } else if (g != ic || g != oc) { // TODO, need to support group convolution in feature.
    return MInfo::lmem_per_lane + 1;
  } else {
    // for dw conv, load weight all in once
    filterSizePerLane = MInfo::getSizePerLane(1, oc, kh, kw, false) * 2;
  }
  // load bias all in once
  int bias_size = with_bias ? 9 : 5;
  uint64_t biasSizePerLane = MInfo::getSizePerLane(1, oc, 1, bias_size, false);

  //
  // if next op is relu, reserve working buffer
  // TODO: not supported yet
  //
  uint64_t reluWorkingSizePerLane = 0;
  // if (do_relu) {
  //}

  //
  // if next op is eltwise, count eltwise input size
  // TODO: not supported yet
  //
  uint64_t eltwiseInputSizePerLane = 0;
  uint64_t eltwiseWorkingSizePerLane = 0;
  //if (do_eltwise) {
  //  eltwiseInputSizePerLane = outputNeuronSizePerLane;
  //  #define MIN_ELTWISE_WORKING_SIZE    (32)
  //  eltwiseWorkingSizePerLane = MIN_ELTWISE_WORKING_SIZE * 2;
  //}

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                          + filterSizePerLane + biasSizePerLane
                          + reluWorkingSizePerLane
                          + eltwiseInputSizePerLane + eltwiseWorkingSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = eltwiseInputSizePerLane;
    details->eltwiseWorkingSizePerLane = eltwiseWorkingSizePerLane;
  }
  return totalPerLane;
}

template
uint64_t SimpleConv2DMemoryUsageAnalysis(tpu::TG_INT8_PC_Conv2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);
template
uint64_t SimpleConv2DMemoryUsageAnalysis(tpu::TG_INT8_PC_DeConv2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);
template
uint64_t SimpleConv2DMemoryUsageAnalysis(tpu::Conv2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);

template <typename OpTy>
uint64_t SimpleEltwiseMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details, int batch_size) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  bool do_relu = op.do_relu();

  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t filterSizePerLane = 0;
  uint64_t biasSizePerLane = 0;

  uint64_t reluWorkingSizePerLane = 0;
  if (do_relu) {
    #define MIN_RELU_WORKING_SIZE    (8 * MInfo::eu_num)
    reluWorkingSizePerLane = MIN_RELU_WORKING_SIZE * 2;
  }

  uint64_t eltwiseInputSizePerLane = 0;
  uint64_t eltwiseWorkingSizePerLane = 0;
  #define MIN_ELTWISE_WORKING_SIZE    (8 * MInfo::eu_num)
  // 2 addend buffers for ping-pong
  // 2 for partial result low and high
  eltwiseWorkingSizePerLane = MIN_ELTWISE_WORKING_SIZE * 4;
  //todo : eltwiseMul ignore this

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                          + filterSizePerLane + biasSizePerLane
                          + reluWorkingSizePerLane
                          + eltwiseInputSizePerLane + eltwiseWorkingSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = eltwiseInputSizePerLane;
    details->eltwiseWorkingSizePerLane = eltwiseWorkingSizePerLane;
  }
  return totalPerLane;
}

template
uint64_t SimpleEltwiseMemoryUsageAnalysis(tpu::TG_INT8_EltwiseAddOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);
template
uint64_t SimpleEltwiseMemoryUsageAnalysis(tpu::TG_INT8_EltwiseMulOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);

template <typename OpTy>
uint64_t SimpleLutMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details,
    int batch_size) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }

  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t filterSizePerLane = 0;
  uint64_t biasSizePerLane = 0;
  uint64_t reluWorkingSizePerLane = 0;

  uint64_t lutInputSizePerLane = 0;
  const uint64_t lutWorkingSizePerLane = 256; //int8 table size

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                          + filterSizePerLane + biasSizePerLane
                          + reluWorkingSizePerLane
                          + lutInputSizePerLane + lutWorkingSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = 0;
    details->eltwiseWorkingSizePerLane = 0;
  }
  return totalPerLane;
}

template
uint64_t SimpleLutMemoryUsageAnalysis(tpu::TG_INT8_LutOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);

template <typename OpTy>
uint64_t SimpleScaleMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details,
    int batch_size) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t filterSizePerLane = MInfo::getSizePerLane(1, c, 1, 1, true);
  uint64_t biasSizePerLane = MInfo::getSizePerLane(1, c, 1, 5, true); // 5 = rightShift(1) + multiplier(4)
  uint64_t reluWorkingSizePerLane = 0;


  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                          + filterSizePerLane + biasSizePerLane
                          + reluWorkingSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = 0;
    details->eltwiseWorkingSizePerLane = 0;
  }
  return totalPerLane;
}

template
uint64_t SimpleScaleMemoryUsageAnalysis(tpu::TG_INT8_ScaleOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);

template <typename OpTy>
uint64_t SimpleIOMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details,
    int batch_size) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(), shape, input_size);
  getNCHW(shape, n, c, h, w);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op.getResult(), shape, output_size);
  getNCHW(shape, on, oc, oh, ow);

  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(on, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;
  uint64_t biasSizePerLane = 0;
  uint64_t reluWorkingSizePerLane = 0;

  uint64_t lutInputSizePerLane = 0;

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                          + filterSizePerLane + biasSizePerLane
                          + reluWorkingSizePerLane
                          + lutInputSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = 0;
    details->eltwiseWorkingSizePerLane = 0;
  }
  return totalPerLane;
}

template
uint64_t SimpleIOMemoryUsageAnalysis(tpu::TG_INT8_PoolMax2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);
template
uint64_t SimpleIOMemoryUsageAnalysis(tpu::TG_INT8_PoolAvg2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1);

uint64_t SimplePixelShuffleMemoryUsageAnalysis(tpu::TG_INT8_PixelShuffleOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(), shape, input_size);
  getNCHW(shape, n, c, h, w);
  uint32_t factor = op.upscale_factor();

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op.getResult(), shape, output_size);
  getNCHW(shape, on, oc, oh, ow);

  uint64_t inputNeuronSizePerLane =
            MInfo::getSizePerLane(factor, factor * MInfo::lane_num, h, w, true);
  uint64_t outputNeuronSizePerLane =
                  MInfo::getSizePerLane(on, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;
  uint64_t biasSizePerLane = 0;
  uint64_t reluWorkingSizePerLane = 0;

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;

  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = outputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = 0;
    details->eltwiseWorkingSizePerLane = 0;
  }
  return totalPerLane;
}

uint64_t SimplePReluMemoryUsageAnalysis(tpu::TG_INT8_PReluOp &op,
    struct SimpleMemoryUsageAnalysis_details *details = nullptr,
    int batch_size = -1) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  if (batch_size != -1) {
    n = batch_size;
    assert((batch_size <= n) && "batch_size error");
  }
  uint64_t inputNeuronSizePerLane =
                  MInfo::getSizePerLane(n, c, h, w, true);
  uint64_t filterSizePerLane = MInfo::getSizePerLane(1, c, 1, 1, true);
  uint64_t biasSizePerLane = 0;
  uint64_t reluWorkingSizePerLane = 0;

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane * 2 + filterSizePerLane;
  // return
  if (details) {
    details->inputNeuronSizePerLane = inputNeuronSizePerLane;
    details->outputNeuronSizePerLane = inputNeuronSizePerLane;
    details->filterSizePerLane = filterSizePerLane;
    details->biasSizePerLane = biasSizePerLane;
    details->reluWorkingSizePerLane = reluWorkingSizePerLane;
    details->eltwiseInputSizePerLane = 0;
    details->eltwiseWorkingSizePerLane = 0;
  }

  return totalPerLane;
}

