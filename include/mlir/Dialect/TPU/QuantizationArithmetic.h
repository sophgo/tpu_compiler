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

namespace mlir {

///
/// utils
///
float findMaxWeight(float *weight, size_t size);

///
/// INT8
///
uint32_t findRShiftForFilter(float max_filter,
                             float threshold_y, float threshold_x);
uint32_t findRShiftForBiasI16(float max_bias, float threshold_y);
uint32_t findRShiftForBiasI32(float max_bias, float threshold_y);

double findQScaleForFilter(float max_filter,
                          float threshold_y, float threshold_x);
double findQScaleForBiasI32(float max_bias, float threshold_y);
uint32_t findRShiftAndMultiplierFromQScale(double qscale,
                                           uint32_t *multiplier = nullptr,
                                           bool qdm = false,
                                           uint32_t max_multiplier = 127);
uint32_t findMultiplierFromQScaleAndRShift(double qscale, uint32_t rshift);

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
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v,
                                               uint32_t rshift, uint32_t multiplier,
                                               bool qdm = false);

int8_t quantizeNeuron(float v, float threshold);
float dequantizeNeuron(int8_t q, float threshold);

///
/// BF16
///
//struct bfloat16 {
//  bfloat16() {}
//  explicit bfloat16(const uint16_t v) : value(v) {}
//  uint16_t value;
//};
typedef uint16_t bfloat16;

void FloatToBFloat16(const float* src, bfloat16* dst, size_t size,
    bool rounding = true);
void BFloat16ToFloat(const bfloat16* src, float* dst, size_t size);

//
// Wrapper APIs
//
void quantizeWeightInt8PerLayer(float *filter, float *bias, int oc, int isz,
                                float threshold_y, float threshold_x,
                                float *new_filter, float *new_bias,
                                float *rshift_per_layer);

void quantizeWeightInt8PerChannel(float *filter, float *bias, int oc, int isz,
                                  float threshold_y, float threshold_x,
                                  float *new_filter, float *new_bias,
                                  float *rshift_per_channel);

void quantizeWeightInt8PerLayerMultiplier(float *filter, float *bias, int oc,
                                          int isz, float threshold_y,
                                          float threshold_x, float *new_filter,
                                          float *new_bias,
                                          float *rshift_per_layer,
                                          float *multiplier_per_layer);

void quantizeWeightInt8Multiplier(float *filter, float *bias, int oc, int isz,
                                  float threshold_y, float threshold_x,
                                  float *new_filter, float *new_bias,
                                  float *rshift_per_channel,
                                  float *multiplier_per_channel);
} // namespace mlir

#endif // MLIR_DIALECT_TPU_QUANTIZATION_ARITHMETIC_H_
