#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "quant_arithmetic"

namespace mlir {

/// find a RShift (no multiplier)
///   Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
///   find a rshift so that Q(max_weight) is in range (64, 127)
/// used in BM1880 or BM1880v2 legacy per-layer mode
/// During runtime, HW will do
///   apply rshift (i.e. divide by (1 << rshift))
///   then saturate to INT8
uint32_t findRShift(float max_weight, float threshold_y, float threshold_x) {
  assert(threshold_y > 0 && threshold_x > 0);
  float a = max_weight * threshold_x / threshold_y;
  assert(a < 128);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (a * (1 << rshift)) >= 64 )
      return rshift;
  }
  assert(false);
  return 31;
}

/// find a QScale (with multiplier)
///   Q(W) = W * (threshold_x / threshold_y) * (1 / QScale)
///   find a QScale so that Q(max_weight) = 127
/// used in BM1880v2 per-channel mode
/// During runtime
///   HW needs to multiple the accumulated result by QScale before saturate to Int8
///   QScale is then decomposed into a multipler and a rshift
///   => QScale = Multiplier / (1 << RShift)
///   where Multiplier is an interger
float findQScale(float max_weight, float threshold_y, float threshold_x) {
  assert(threshold_y > 0 && threshold_x > 0);
  float qscale = (max_weight * threshold_x) / (127.0f * threshold_y);
  return qscale;
}

/// find RShift and Multiplier from QScale
///   QScale = Multiplier / (1 << RShift)
///   Multiplier is an integer
///   Sometimes Multiplier is limited to be int8_t or uint8_t
///   Sometimes Multiplier could be uint32_t
/// used in BM1880v2 per-channel mode
///   if 'uint32_t *multiplier' is present, return multipler together
uint32_t findRShiftAndMultiplierFromQScale(float qscale,
    uint32_t max_multiplier = 127,  uint32_t *multiplier = nullptr) {
  // TODO: refine max_mutliplier
  assert(qscale < max_multiplier);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (qscale * (1 << (rshift + 1))) >= max_multiplier ) {
      if (multiplier) {
        *multiplier = (uint32_t)(qscale * (1 << rshift));
      }
      return rshift;
    }
  }
  llvm::errs() << "failed to find rshift, qscale = " << std::to_string(qscale) << "\n";
  assert(false);
  return 32;
}

/// find Multiplier from QScale and RShift
///   QScale = Multiplier / (1 << RShift)
///   Multiplier = QScale * (1 << RShift)
uint32_t findMultiplierFromQScaleAndRShift(float qscale, uint32_t rshift) {
  return (uint32_t)(qscale * (1 << rshift));
}

/// saturate a float to range [-128, 127]
static inline int8_t saturateInt8(float f) {
  #if 0
  // cast
  int q = (int)f;
  #elif 0
  // away_from_zero
  int q = (f >= 0) ? (int)ceil(f) : (int)floor(f);
  #elif 0
  // round
  int q = (int)roundf(f);
  #elif 0
  // trancate, (towards zero)
  int q = (f >= 0) ? (int)floor(f) : (int)ceil(f);
  #else
  // from caffe_int8
  int q = floor(f + 0.5);
  #endif
  //assert( (q <= 127) && (q >= -128) );
  DEBUG_WITH_TYPE(DEBUG_TYPE"_WARNING",
    if ( (q > 127) || (q < -128) ) {
      llvm::errs() << "exceeds limits [-128, 127] : " << std::to_string(f) << "\n";
    }
  );
  if ( q > 127 )
    q = 127;
  if ( q < -128 )
    q = -128;

  return (int8_t)q;
}

/// saturate a float to int16_t
static inline int16_t saturateInt16(float f) {
  #if 0
  // cast
  int q = (int)f;
  #elif 0
  // away_from_zero
  int q = (f >= 0) ? (int)ceil(f) : (int)floor(f);
  #elif 0
  // round
  int q = (int)roundf(f);
  #elif 0
  // trancate, (towards zero)
  int q = (f >= 0) ? (int)floor(f) : (int)ceil(f);
  #else
  // from caffe_int8
  int q = floor(f + 0.5);
  #endif
  if ( (q > 32767) || (q < -32768) ) {
    llvm::errs() << "exceeds limits [-32768, 32767] : " << std::to_string(f) << "\n";
  }
  assert( (q <= 32767) && (q >= -32768) );
  if ( q > 32767 )
    q = 32767;
  if ( q < -32768 )
    q = -32768;
  return (int16_t)q;
}

/// saturate a float to int
static inline int32_t saturateInt32(float f) {
  if ( (f > INT_MAX) || (f < INT_MIN) ) {
    llvm::errs() << "exceeds INT limits : " << std::to_string(f) << "\n";
  }
  assert( (f <= INT_MAX) && (f >= INT_MIN) );
  #if 0
  // cast
  int q = (int)f;
  #elif 0
  // away_from_zero
  int q = (f >= 0) ? (int)ceil(f) : (int)floor(f);
  #elif 0
  // round
  int q = (int)roundf(f);
  #elif 0
  // trancate, (towards zero)
  int q = (f >= 0) ? (int)floor(f) : (int)ceil(f);
  #else
  // from caffe_int8
  int q = floor(f + 0.5);
  #endif
  return (int32_t)q;
}

/// quantize a filter weight value into int8 based on rshift (no multiplier)
///   Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
/// used in BM1880 or BM1880v2 legacy per-layer mode
int8_t quantizeFilterRShift(float w, float threshold_y, float threshold_x,
    uint32_t rshift) {
  float factor = (threshold_x / threshold_y) * (1 << rshift);
  float q_f = w * factor;
  return saturateInt8(q_f);
}

/// quantize a bias weight value into int16
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880 or BM1880v2 legacy per-layer mode (16bit bias)
int16_t quantizeBiasRShiftI16(float w, float threshold_y, uint32_t rshift) {
  float factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = w * factor;
  return saturateInt16(q_f);
}

/// quantize a bias weight value into int32
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880v2 per-channel mode (32bit bias), by with no multiplier
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift) {
  float factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = w * factor;
  return saturateInt32(q_f);
}

/// quantize a filter weight value into int8 based on rshift and multiplier
///   Q(W) = W * (threshold_x / threshold_y) * (1 / QScale)
///   QScale = Multiplier / (1 << RShift)
/// used in BM1880 or BM1880v2 legacy per-layer mode
int8_t quantizeFilterRShiftAndMultiplier(float w, float threshold_y,
    float threshold_x, uint32_t rshift, uint32_t multiplier) {
  float factor = (threshold_x / threshold_y) * (1 << rshift) / multiplier;
  float q_f = w * factor;
  return saturateInt8(q_f);
}

/// quantize a bias weight value into int32 based on rshift and multiplier
///   Q(B) = B * (128.0f / threshold_y) * (1 / QScale)
///   QScale = Multiplier * (1 << RShift)
/// used in BM1880v2 per-channel mode (32bit bias)
int32_t quantizeBiasRShiftAndMultiplier(float w, float threshold_y,
    uint32_t rshift, uint32_t multiplier) {
  float factor = (128.0f / threshold_y) * (1 << rshift) / multiplier;
  float q_f = w * factor;
  return saturateInt32(q_f);
}

/// Simulate HW behavior, after accumuation, do rshift and saturate
int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift) {
  return saturateInt8(v / (1 << rshift));
}

/// Simulate HW behavior, after accumuation
/// apply multiplier, do rshift, and then saturate to INT8
/// used in BM1880v2 per-channel mode (32bit bias)
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v,
    uint32_t rshift, uint32_t multiplier) {
  return saturateInt8(v * multiplier / (1 << rshift));
}

/// Quantize a Neuron value into INT8, given threshold
int8_t quantizeNeuron(float v, float threshold) {
  return saturateInt8(v * 128.0 / threshold);
}

/// DeQuantize a Neuron value from INT8, given threshold
float dequantizeNeuron(int8_t q, float threshold) {
  return (float)q * threshold / 128.0;
}

} // namespace
