#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
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
  if (a >= 128) {
    llvm::errs() << "WARNING: findRshift, max_weight too large "
                 << std::to_string(max_weight) << "\n";
    return 0;
  }
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
///   HW multiples the accumulated result by QScale before saturate to Int8
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
/// case 1: specifically multiply a int8/uint8 multplier, then rshift
///   used in layers like element_wise, pooling, concat, etc
///   qdm is false
///   a max_multiplier (127 or 255 normally) has to be provided
/// case 2: dqm mode
///   used in BM1880v2 per-channel conv mode
///   qdm is true
///   reference to [arxiv 1712.05877]
///     choose the int32 value nearest to 2^31 * M0, M0 in [0.5, 1]
///     this value is always at least 2^30 and have at least 30 bits accuracy
///   the max_multiplier argument is ignored, fixed to (1 << 31)
/// if 'uint32_t *multiplier' is present, return multipler alongside
uint32_t findRShiftAndMultiplierFromQScale(float qscale,
    uint32_t *multiplier, bool qdm, uint32_t max_multiplier) {
  if (qdm) {
    max_multiplier = (1 << 31);
  }
  assert(qscale < max_multiplier);
  for (uint32_t rshift = 0; rshift < 63; ++rshift) {
    if ( ((double)qscale * (1ULL << (rshift + 1))) >= (double)max_multiplier ) {
      if (multiplier) {
        *multiplier = (uint32_t)((double)qscale * (1ULL << rshift));
      }
      if (qdm) {
        return rshift - 31;
      } else {
        return rshift;
      }
    }
  }
  llvm::errs() << "failed to find rshift, qscale = "
               << std::to_string(qscale) << "\n";
  assert(false);
  return 64;
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
      llvm::errs() << "exceeds limits [-128, 127] : "
                   << std::to_string(f) << "\n";
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
    llvm::errs() << "exceeds limits [-32768, 32767] : "
                 << std::to_string(f) << "\n";
  }
  //assert( (q <= 32767) && (q >= -32768) );
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
  double factor = (threshold_x / threshold_y) * (1 << rshift);
  float q_f = (float)(w * factor);
  return saturateInt8(q_f);
}

/// quantize a bias weight value into int16
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880 or BM1880v2 legacy per-layer mode (16bit bias)
int16_t quantizeBiasRShiftI16(float w, float threshold_y, uint32_t rshift) {
  double factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = (float)(w * factor);
  return saturateInt16(q_f);
}

/// quantize a bias weight value into int32
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880v2 per-channel mode (32bit bias), by with no multiplier
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift) {
  double factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = (float)(w * factor);
  return saturateInt32(q_f);
}

/// quantize a filter weight value into int8 based on rshift and multiplier
///   Q(W) = W * (threshold_x / threshold_y) * (1 / QScale)
///   QScale = Multiplier / (1 << RShift)
/// used in BM1880 or BM1880v2 legacy per-layer mode
int8_t quantizeFilterRShiftAndMultiplier(float w, float threshold_y,
    float threshold_x, uint32_t rshift, uint32_t multiplier, bool qdm) {
  if (qdm) {
    rshift += 31;
  }
  double factor = (double)(threshold_x / threshold_y)
                  * (1ULL << rshift) / multiplier;
  float q_f = (float)(w * factor);
  return saturateInt8(q_f);
}

/// quantize a bias weight value into int32 based on rshift and multiplier
///   Q(B) = B * (128.0f / threshold_y) * (1 / QScale)
///   QScale = Multiplier * (1 << RShift)
/// used in BM1880v2 per-channel mode (32bit bias)
int32_t quantizeBiasRShiftAndMultiplier(float w, float threshold_y,
    uint32_t rshift, uint32_t multiplier, bool qdm) {
  if (qdm) {
    rshift += 31;
  }
  double factor = (double)(128.0f / threshold_y)
                  * (1ULL << rshift) / multiplier;
  float q_f = (float)(w * factor);
  return saturateInt32(q_f);
}

/// Simulate HW behavior, after accumuation, do rshift and saturate
int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift) {
  return saturateInt8(v / (1 << rshift));
}

// USE_GOOGLE_GEMMLOWP_QDM
typedef int32_t s32;
static inline s32 RoundingDivideByPOT(s32 x, int exponent)
{
  const s32 shift_vec = -exponent;
  const s32 fixup = (x & shift_vec) >> 31;
  const s32 fixed_up_x = x + fixup;

  // Handel positive right shift
  s32 nudge = (exponent > 0) ? (1 << (exponent - 1)) : 0;
  s32 val = (fixed_up_x + nudge) >> exponent;

  return val;
}

static inline s32 SaturatingRoundingDoublingHighMul(s32 a, s32 b)
{
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  s32 nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  s32 ab_x2_high32 = static_cast<s32>((ab_64 + nudge) / (1ll << 31));

  return ab_x2_high32;
}
// END_GOOGLE_GEMMLOWP_QDM

/// Simulate HW behavior, after accumuation
/// apply multiplier, do rshift, and then saturate to INT8
/// used in BM1880v2 per-channel mode (32bit bias)
/// qdm mode
///   use GOOGLE GEMMLOWP QDM multiply and shift
///   during multiply, a factor of (1 << 31) has been devided
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v,
    uint32_t rshift, uint32_t multiplier, bool qdm) {
  if (qdm) {
    int32_t q = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul((int32_t)v, (int32_t)multiplier),
        rshift);
    //llvm::errs() << "v,rshift,multiplier,q = " << v << ","
    //             << rshift << "," << multiplier << "," << q << "\n";
    return saturateInt8((float)q);
  } else {
    return saturateInt8(v * multiplier / (1 << rshift));
  }
}

/// Quantize a Neuron value into INT8, given threshold
int8_t quantizeNeuron(float v, float threshold) {
  return saturateInt8(v * 128.0 / threshold);
}

/// DeQuantize a Neuron value from INT8, given threshold
float dequantizeNeuron(int8_t q, float threshold) {
  return (float)q * threshold / 128.0;
}

///
/// BFLOAT16 support
/// from tensorflow
///   tensorflow/tensorflow/core/framework/bfloat16.cc
///   tensorflow/tensorflow/core/framework/bfloat16.h
///

// Compact 16-bit encoding of floating point numbers. This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.  It
// is assumed that floats are in IEEE 754 format so the representation is just
// bits 16-31 of a single precision float.
//
// NOTE: The IEEE floating point standard defines a float16 format that
// is different than this format (it has fewer bits of exponent and more
// bits of mantissa).  We don't use that format here because conversion
// to/from 32-bit floats is more complex for that format, and the
// conversion for this format is very simple.
//
// Because of the existing IEEE float16 type, we do not name our representation
// "float16" but just use "uint16".
//
// <-----our 16bits float------->
// s e e e e e e e e f f f f f f f f f f f f f f f f f f f f f f f
// <------------------------------float-------------------------->
// 3 3             2 2             1 1                           0
// 1 0             3 2             5 4                           0
//
//
// This type only supports conversion back and forth with float.
//
// This file must be compilable by nvcc.
//
// The type is defined in framework/numeric_types.h.

void FloatToBFloat16(const float* src, bfloat16* dst, size_t size,
    bool rounding) {
  // const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  const uint16_t* p = nullptr;
  /// if rounding is prefered than trancating
  /// float_val *= 1.001957f;
  float *src_round = nullptr;
  if (rounding) {
    src_round = (float *)malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
      src_round[i] = src[i] * 1.001957f;
    }
    p = reinterpret_cast<const uint16_t*>(src_round);
  } else {
    p = reinterpret_cast<const uint16_t*>(src);
  }

  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p += 2, q++, size--) {
    *q = p[0];
  }
#else
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
#endif
  if (rounding) {
    free(src_round);
  }
}

void BFloat16ToFloat(const bfloat16* src, float* dst, size_t size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p++, q += 2, size--) {
    q[0] = *p;
    q[1] = 0;
  }
#else
  for (; size != 0; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
#endif
}

} // namespace
