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
/// INT8
///
uint32_t findRShift(float max_weight, float threshold_y, float threshold_x);
float findQScale(float max_weight, float threshold_y, float threshold_x);
uint32_t findRShiftAndMultiplierFromQScale(float qscale,
    uint32_t *multiplier = nullptr, bool qdm = false,
    uint32_t max_multiplier = 127);
uint32_t findMultiplierFromQScaleAndRShift(float qscale, uint32_t rshift);

int8_t quantizeFilterRShift(float w, float threshold_y, float threshold_x,
    uint32_t rshift);
int16_t quantizeBiasRShiftI16(float w, float threshold_y, uint32_t rshift);
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift);
int8_t quantizeFilterRShiftAndMultiplier(float w, float threshold_y,
    float threshold_x, uint32_t rshift, uint32_t multiplier,
    bool qdm = false);
int32_t quantizeBiasRShiftAndMultiplier(float w, float threshold_y,
    uint32_t rshift, uint32_t multiplier, bool qdm = false);

int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift);
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v,
    uint32_t rshift, uint32_t multiplier, bool qdm = false);

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

} // namespace mlir

#endif // MLIR_DIALECT_TPU_QUANTIZATION_ARITHMETIC_H_
