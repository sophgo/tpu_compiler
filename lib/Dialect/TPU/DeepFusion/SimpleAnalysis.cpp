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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "MachineInfo.h"
#include "SimpleAnalysis.h"

using namespace mlir;

template <typename OpTy>
uint64_t SimpleConv2DMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);
  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;
  // filter working size *2 for double buffer
  if (g != oc) {
    if(g != 1) { // TODO, not support group convolution now.
      return MInfo::lmem_per_lane + 1;
    }
    // for non-dw conv, assuming oc_step = lane_num
    int oc_step = MInfo::lane_num;
    filterSizePerLane = MInfo::getSizePerLane(ic, oc_step, kh, kw, false) * 2;
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
    struct SimpleMemoryUsageAnalysis_details *details);
template
uint64_t SimpleConv2DMemoryUsageAnalysis(tpu::TG_INT8_PC_DeConv2DOp &op,
    struct SimpleMemoryUsageAnalysis_details *details);

template <typename OpTy>
uint64_t SimpleEltwiseMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
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
    struct SimpleMemoryUsageAnalysis_details *details);
template
uint64_t SimpleEltwiseMemoryUsageAnalysis(tpu::TG_INT8_EltwiseMulOp &op,
    struct SimpleMemoryUsageAnalysis_details *details);

template <typename OpTy>
uint64_t SimpleLutMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details) {
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op.getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

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
    struct SimpleMemoryUsageAnalysis_details *details);