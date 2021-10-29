//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_QUANTIZATION_ARITHMETIC_H_
#define MLIR_DIALECT_TPU_QUANTIZATION_ARITHMETIC_H_

#include "mlir/IR/Operation.h"
#include <omp.h>

namespace mlir {

///
/// INT8
///
uint32_t findRShiftForFilter(float max_filter,
                             float threshold_y, float threshold_x);
uint32_t findRShiftForBiasI16(float max_bias, float threshold_y);
uint32_t findRShiftForBiasI32(float max_bias, float threshold_y);

double findQScaleForFilter(float max_filter,
                          float threshold_y, float threshold_x, int quant_bitwidth = 8);
double findQScaleForBiasI32(float max_bias, float threshold_y);
int8_t findRShiftAndMultiplierFromQScale(double qscale,
                                           uint32_t *multiplier = nullptr,
                                           bool qdm = false,
                                           uint32_t max_multiplier = 127);
uint32_t findMultiplierU32FromQScaleAndRShift(double qscale, uint32_t rshift);
int8_t findMultiplierI8FromQScaleAndRShift(double qscale, int8_t rshift);

int8_t quantizeFilterRShift(float w, float threshold_y, float threshold_x,
                            uint32_t rshift);
int16_t quantizeBiasRShiftI16(float w, float threshold_y, uint32_t rshift);
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift);
int8_t quantizeFilterRShiftAndMultiplier(float w,
                                         float threshold_y, float threshold_x,
                                         uint32_t rshift, uint32_t multiplier,
                                         bool qdm = false);
int32_t quantizeBiasRShiftAndMultiplier(float w,
                                        float threshold_y,
                                        uint32_t rshift, uint32_t multiplier,
                                        bool qdm = false);

int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift);
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v, uint32_t rshift,
                                               uint32_t multiplier,
                                               bool qdm = false);
int8_t applyMultiplierAndRShiftAndSaturateInt8(int32_t v,
                                               uint32_t rshift, uint32_t multiplier,
                                               bool qdm = false);

float applyZeroPointSaturateInt8(float v, int offset);

//
// Wrapper APIs
//
void quantizeWeightInt8PerLayer(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias, float *rshift_per_layer);

void quantizeWeightInt8PerChannel(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias, float *rshift_per_channel);

void quantizeWeightInt8ForFC(float *filter, float *bias, int64_t batch,
                             int64_t N, int64_t K, float threshold_y,
                             float threshold_x, float *new_filter,
                             float *new_bias, float *rshift_per_batch,
                             float *multiplier_per_batch);

void quantizeBiasInt8PerLayerMultiplier(float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias,
    float *rshift_per_layer, float *multiplier_per_layer, double qscale,
    bool qdm);

void quantizeWeightInt8Multiplier(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias,
    float *rshift_per_channel, float *multiplier_per_channel,
    std::vector<float> &filter_threshold, int quant_bitwidth = 8);

void quantizeActivationInt8PerLayerRshift(float *output, float *input,
    int64_t size, uint32_t rshift,int offset=0);

void quantizeActivationInt8PerChannelRShift(float *output, float *input,
    int64_t on, int64_t oc, int64_t isz, float *rshift_per_channel, int offset=0);

void quantizeActivationInt8PerChannelMultiplierAndRShift(
    float *output, float *input, float *bias, bool do_relu, int64_t on, int64_t oc,
    int64_t isz, float *rshift_per_channel, float *multiplier_per_channel);


typedef uint16_t bfloat16;

int8_t F32ToInt8(float v, int round_mode);
uint8_t F32ToUint8(float v, int round_mode);

bfloat16 F32ToBF16(float src, bool rounding = true);

void F32ToBF16(float *src, bfloat16 *dst, size_t size, bool rounding=true);

float BF16(float src, bool rounding = true);

void BF16(float *src, float *dst, size_t size, bool rounding = true);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_QUANTIZATION_ARITHMETIC_H_
