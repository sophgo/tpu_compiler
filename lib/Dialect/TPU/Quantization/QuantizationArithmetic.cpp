#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"


#define DEBUG_TYPE "quant_arithmetic"

namespace mlir {

/// find max_weight for each c
float findMaxWeight(float *weight, int64_t size) {
  float max_abs = fabs(weight[0]);
  for (int64_t i = 0; i < size; ++i) {
    if (fabs(weight[i]) > max_abs) {
      max_abs = fabs(weight[i]);
    }
  }
  return max_abs;
}

/// find a RShift For Filter (no multiplier)
///   Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
///   find a rshift so that Q(max_filter) is in range (64, 127)
/// used in BM1880 or BM1880v2 legacy per-layer mode
/// During runtime, HW will do
///   apply rshift (i.e. divide by (1 << rshift))
///   then saturate to INT8
uint32_t findRShiftForFilter(float max_filter,
    float threshold_y, float threshold_x) {
  assert(threshold_y > 0 && threshold_x > 0);
  float a = max_filter * threshold_x / threshold_y;
  if (a > 127) {
    LLVM_DEBUG(llvm::errs() << "WARNING: findRShiftForFilter, max_filter too large "
                 << std::to_string(max_filter)
                 << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 127);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (a * (1 << rshift)) >= 64 ) {
      LLVM_DEBUG(
        if (rshift >= 25) {
          llvm::errs() << "WARNING: findRShiftForFilter, large rshift = " << rshift
                       << ", max_filter = " << max_filter
                       << ", threshold_y = " << std::to_string(threshold_y)
                       << ", threshold_x = " << std::to_string(threshold_x)
                       << "\n";
        }
      );
      return rshift;
    }
  }
  // we are here because a < 64 / (1 << 32), which mean max_filter is near zero
  //assert(false);
  LLVM_DEBUG(llvm::errs() << "WARNING: findRShiftForFilter, max_filter too small\n";);
  // return 0, to make sure activation will go zero by multiply such small weight
  return 0;
}

/// find a RShift For Bias I16
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
///   find a rshift so that Q(max_bias) is in range (16384, 32767)
/// used in BM1880 or BM1880v2 legacy per-layer mode
/// During runtime, HW will do
///   apply rshift (i.e. divide by (1 << rshift))
///   then saturate to INT16
uint32_t findRShiftForBiasI16(float max_bias, float threshold_y) {
  assert(threshold_y > 0);
  float a = max_bias * 128.0 / threshold_y;
  if (a > 32767) {
    LLVM_DEBUG(llvm::errs() << "WARNING: findRShiftForBiasI16, max_bias too large "
                 << std::to_string(max_bias)
                 << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 32767);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (a * (1 << rshift)) >= 16384 ) {
      return rshift;
    }
  }
  // we are here because a < 16384 / (1 << 32), which mean max_bias is near zero
  // llvm::errs() << "WARNING: findRShiftForBiasI16, max_bias small enough\n";
  // return 31, to not limiting final rshift (min(rshift_filter, rshift_bias))
  return 31;
}

/// find a RShift For Bias I32
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
///   find a rshift so that Q(max_bias) is in range (0x40000000, 0x7fffffff)
/// used in BM1880 or BM1880v2 legacy per-channel mode
/// During runtime, HW will do
///   apply rshift (i.e. divide by (1 << rshift))
///   then saturate to INT32
uint32_t findRShiftForBiasI32(float max_bias, float threshold_y) {
  assert(threshold_y > 0);
  float a = max_bias * 128.0 / threshold_y;
  if (a > 0x7fffffff) {
    LLVM_DEBUG(llvm::errs() << "WARNING: findRShiftForBiasI32, max_bias too large "
                 << std::to_string(max_bias)
                 << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 0x7fffffff);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (a * (1 << rshift)) >= 0x40000000 ) {
      LLVM_DEBUG(
        if (rshift < 25) {
          llvm::errs() << "WARNING: findRShiftForBiasI32, find rshift = " << rshift
                       << ", max_bias = " << std::to_string(max_bias)
                       << ", threshold_y = " << std::to_string(threshold_y)
                       << "\n";
        }
      );
      return rshift;
    }
  }
  // we are here because a < 0x40000000 / (1 << 32), which mean max_bias is small enough
  // llvm::errs() << "WARNING: findRShiftForBiasI32, max_bias small enough\n";
  // return 31, to not limiting final rshift (min(rshift_filter, rshift_bias))
  return 31;
}

/// find a QScale for Filter (with multiplier)
///   Q(W) = W * (threshold_x / threshold_y) * (1 / QScale)
///   find a QScale so that Q(max_filter) = 127
/// used in BM1880v2 per-channel mode
/// During runtime
///   HW multiples the accumulated result by QScale before saturate to Int8
///   QScale is then decomposed into a multipler and a rshift
///   => QScale = Multiplier / (1 << RShift)
///   where Multiplier is an interger
double findQScaleForFilter(float max_filter, float threshold_y, float threshold_x) {
  assert(threshold_y > 0 && threshold_x > 0);
  double qscale = (max_filter * threshold_x) / (127.0f * threshold_y);
  return qscale;
}

/// find a QScale For Bias I32
///   Q(B) = B * (128.0f / threshold_y)  * (1 / QScale)
///   find a QScale so that Q(max_bias) = 0x7fffffff
/// used in BM1880v2 per-channel mode
/// During runtime
///   HW multiples the accumulated result by QScale before saturate to Int8
///   QScale is then decomposed into a multipler and a rshift
///   => QScale = Multiplier / (1 << RShift)
///   where Multiplier is an interger
double findQScaleForBiasI32(float max_bias, float threshold_y) {
  assert(threshold_y > 0);
  // 0x7fffffff * 0.99 to workaround precision issue
  double qscale = (max_bias * 127.0f) / (0x7fffffff * 0.99 * threshold_y);
  return qscale;
}

// reference to reference to [arxiv 1712.05877]
// This implementation comes from tensorflow
// https://github.com/tensorflow/tensorflow/blob/98ff991500a0247f8f57c60db9a206204268bc42/tensorflow/lite/kernels/internal/quantization_util.cc#L52-L90
#define Tensorflow_QuantizeMultiplier QuantizeMultiplier
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));

  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }

  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

/// find RShift and Multiplier from QScale
///   QScale = Multiplier / (1 << RShift)
///   Multiplier is an integer
/// case 1: specifically multiply a int8/uint8 multplier, then rshift
///   used in layers like element_wise, pooling, concat, etc
///   qdm is false
///   a max_multiplier (127 or 255 normally) has to be provided
/// case 2: qdm mode
///   used in BM1880v2 per-channel conv mode
///   qdm is true
///   reference to [arxiv 1712.05877]
///     choose the int32 value nearest to 2^31 * M0, M0 in [0.5, 1]
///     this value is always at least 2^30 and have at least 30 bits accuracy
///   the max_multiplier argument is ignored, fixed to (1 << 31)
/// if 'uint32_t *multiplier' is present, return multipler alongside
int8_t findRShiftAndMultiplierFromQScale(double qscale,
    uint32_t *multiplier, bool qdm, uint32_t max_multiplier) {
  if (qdm) {
    #if 0
    max_multiplier = (1 << 31);
    for (uint32_t rshift = 0; rshift < 63; ++rshift) {
      if ( ((double)qscale * (1ULL << (rshift + 1))) >= (double)max_multiplier ) {
        if (multiplier) {
          *multiplier = (uint32_t)((double)qscale * (1ULL << rshift));
        }
        return rshift - 31;
      }
    }
    #endif
    // this ensures if qscale is 0, both multiplier and shift will be 0
    int32_t quantized_multiplier = 0;
    int lshift = 0;
    Tensorflow_QuantizeMultiplier(qscale, &quantized_multiplier, &lshift);
    *multiplier = quantized_multiplier;
    int rshift = -lshift;
    assert(rshift >= 0);
    LLVM_DEBUG(
      if (rshift > 25) {
        llvm::errs() << "WARNING: large rshift = " << rshift
                     << ", qscale = " << qscale
                     << "\n";
      }
    );
    return (int8_t)rshift;
  } else {
    if(qscale > max_multiplier){
      llvm::errs() << "Error: qscale > max_multipiler ( " << qscale << " v.s. "
                   << max_multiplier << " )\n";
      assert(false);
    }
    for (int8_t rshift = 0; rshift < 63; ++rshift) {
      if ( ((double)qscale * (1ULL << (rshift + 1))) >= (double)max_multiplier ) {
        if (multiplier) {
          *multiplier = (uint32_t)((double)qscale * (1ULL << rshift));
        }
        return rshift;
      }
    }
    //assert(false);
    LLVM_DEBUG(llvm::errs() << "WARNING: failed to find rshift, qscale = "
                 << std::to_string(qscale) << "\n";);
    // we are here because qscale is too small, return 0 for both shift and multiplier
    if (multiplier) {
      *multiplier = 0;
    }
    return 0;
  }
}

/// find Multiplier from QScale and RShift
///   QScale = Multiplier / (1 << RShift)
///   Multiplier = QScale * (1 << RShift)
uint32_t findMultiplierU32FromQScaleAndRShift(double qscale, int8_t rshift) {
  return (uint32_t)(qscale * (1 << rshift));
}

int8_t findMultiplierI8FromQScaleAndRShift(double qscale, int8_t rshift) {
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
  #elif 1
  // from caffe_int8
  int q = floor(f + 0.5);
  #else
  // looks HW is different than std::round()
  // we shall apply round only for input quant()
  int q = std::round(f);
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
  LLVM_DEBUG(
    if ( (q > 32767) || (q < -32768) ) {
      llvm::errs() << "WARNING: exceeds limits [-32768, 32767] : "
                   << std::to_string(f) << "\n";
    }
  );
  //assert( (q <= 32767) && (q >= -32768) );
  if ( q > 32767 )
    q = 32767;
  if ( q < -32768 )
    q = -32768;
  return (int16_t)q;
}

/// saturate a float to int
static inline int32_t saturateInt32(float f) {
  LLVM_DEBUG(
    if ( (f > INT_MAX) || (f < INT_MIN) ) {
      llvm::errs() << "WARNING: exceeds INT limits : "
                   << std::to_string(f) << "\n";
    }
  );
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
  LLVM_DEBUG(
    if ( (q_f > INT_MAX) || (q_f < INT_MIN) ) {
      llvm::errs() << "WARNING: quantizeBiasRShiftI16, exceeds INT limits : "
                   << std::to_string(q_f) << "\n";
      llvm::errs() << "  w: " << std::to_string(w)
                   << ", threshold_y: " << std::to_string(threshold_y)
                   << ", rshift: " << std::to_string(rshift)
                   << "\n";
    }
  );
  return saturateInt16(q_f);
}

/// quantize a bias weight value into int32
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880v2 per-channel mode (32bit bias), by with no multiplier
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift) {
  double factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = (float)(w * factor);
  LLVM_DEBUG(
    if ( (q_f > INT_MAX) || (q_f < INT_MIN) ) {
      llvm::errs() << "WARNING: quantizeBiasRShiftI32, exceeds INT limits : "
                   << std::to_string(q_f) << "\n";
      llvm::errs() << "  w: " << std::to_string(w)
                   << ", threshold_y: " << std::to_string(threshold_y)
                   << ", rshift: " << std::to_string(rshift)
                   << "\n";
    }
  );
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
  double factor = (multiplier == 0) ? 0 :
      (double)(threshold_x / threshold_y) * (1ULL << rshift) / multiplier;
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
  double factor = (multiplier == 0) ? 0 :
      (double)(128.0f / threshold_y) * (1ULL << rshift) / multiplier;
  float q_f = (float)(w * factor);
  LLVM_DEBUG(
    if ( (q_f > INT_MAX) || (q_f < INT_MIN) ) {
      llvm::errs() << "WARNING: quantizeBiasRShiftI32, exceeds INT limits : "
                   << std::to_string(q_f) << "\n";
      llvm::errs() << "  w: " << std::to_string(w)
                   << ", threshold_y: " << std::to_string(threshold_y)
                   << ", multiplier: " << multiplier
                   << ", rshift: " << std::to_string(rshift)
                   << "\n";
    }
  );
  return saturateInt32(q_f);
}

/// Simulate HW behavior, after accumuation, do rshift and saturate
int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift, int offset) {
  return saturateInt8((v / (1 << rshift)) + offset);
}

/// Simulate HW behavior, after accumuation, do rshift and saturate
float applyZeroPointSaturateInt8(float v, int offset) {
  return (float)saturateInt8(v + offset);
}

// USE_GOOGLE_GEMMLOWP_QDM
typedef int32_t s32;
static inline s32 RoundingDivideByPOT(s32 x, int exponent)
{
  if (x == 0) {
    return 0;
  }
  if (exponent == 0) {
    return x;
  }
  assert(exponent > 0 && exponent <= 31);
  const int32_t mask = (1ll << exponent) - 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return ((x >> exponent) + ((remainder > threshold) ? 1 : 0));
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
    uint32_t rshift, uint32_t multiplier, bool qdm, int offset) {
  if (qdm) {
    int32_t q = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul((int32_t)v, (int32_t)multiplier),
        rshift);
    return saturateInt8((float)(q + offset));
  } else {
    return saturateInt8((float)(((v * multiplier)) / (1 << rshift) + offset));
  }
}

int8_t applyMultiplierAndRShiftAndSaturateInt8(int32_t v,
    uint32_t rshift, uint32_t multiplier, bool qdm, int offset) {
  if (qdm) {
    int32_t q = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(v, (int32_t)multiplier),
        rshift);
    return saturateInt8((float)(q + offset));
  } else {
    return saturateInt8((float)(((v * multiplier)) / (1 << rshift) + offset));
  }
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
      float value = src[i];
      uint32_t* u32_val = reinterpret_cast<uint32_t*>(&value);
      uint32_t lsb = (*u32_val >> 16) & 1;
      *u32_val += (0x7fff + lsb); // rounding_bias
      float *ret = reinterpret_cast<float*>(u32_val);
      src_round[i] = *ret;
    }
    p = reinterpret_cast<const uint16_t*>(src_round);
  } else {
    p = reinterpret_cast<const uint16_t*>(src);
  }

  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p += 2, q++, size--) {
    *q = p[0];
    /* HW behavior */
    // infinity set to max finite positive value
    if ((*q & 0x7f80) == 0x7f80) {
      *q = 0x7f7f;
    }
  }
#else
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
    /* HW behavior */
    // infinity set to max finite positive value
    if ((*q & 0x7f80) == 0x7f80) {
      *q = 0x7f7f;
    }
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

//
// Tensors wise API
//
void quantizeWeightInt8PerLayer(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias, float *rshift_per_layer)  {
  // find rshift
  float max_filter = findMaxWeight(filter, oc * isz);
  float rshift_filter =
      (float)findRShiftForFilter(max_filter, threshold_y, threshold_x);
  rshift_per_layer[0] = rshift_filter;
  float max_bias = 0.0f;
  if (bias) {
    float max_bias = findMaxWeight(bias, oc);
    float rshift_bias =
        (float)findRShiftForBiasI16(max_bias, threshold_y);
    if (rshift_bias < rshift_filter) {
      LLVM_DEBUG(llvm::errs() << "WARNING: adjust rshift for bias"
                   << ", rshift_filter = " << std::to_string(rshift_filter)
                   << ", rshift_bias = " << std::to_string(rshift_bias)
                   << "\n";);
      rshift_per_layer[0] = rshift_bias;
    }
  }
  LLVM_DEBUG(llvm::errs() << "  max_filter : " << std::to_string(max_filter)
                          << ", max_bias : " << std::to_string(max_bias)
                          << ", rshift : " << rshift_per_layer[0] << "\n";);

  // quantize weight
  for (int64_t i = 0; i < oc * isz; ++i) {
    new_filter[i] = (float)quantizeFilterRShift(
        filter[i], threshold_y, threshold_x, rshift_per_layer[0]);
  }
  if (bias) {
    for (int64_t i = 0; i < oc; ++i) {
      new_bias[i] = (float)quantizeBiasRShiftI16(bias[i], threshold_y,
                                                 rshift_per_layer[0]);
    }
  }
}

void quantizeWeightInt8PerChannel(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias, float *rshift_per_channel) {
  // find rshift
  auto max_filter = std::vector<float>(oc);
  auto max_bias = std::vector<float>(oc);
  for (int i = 0; i < oc; ++i) {
    max_filter[i] = findMaxWeight(&filter[isz * i], isz);
    float rshift_filter =
        (float)findRShiftForFilter(max_filter[i], threshold_y, threshold_x);
    rshift_per_channel[i] = rshift_filter;
    max_bias[i] = 0.0f;
    if (bias) {
      max_bias[i] = fabs(bias[i]);
      float rshift_bias =
          (float)findRShiftForBiasI32(max_bias[i], threshold_y);
      if (rshift_bias < rshift_filter) {
        LLVM_DEBUG(llvm::errs() << "WARNING: adjust rshift for bias"
                     << ", rshift_filter = " << std::to_string(rshift_filter)
                     << ", rshift_bias = " << std::to_string(rshift_bias)
                     << "\n";);
        rshift_per_channel[i] = rshift_bias;
      }
    }
    LLVM_DEBUG(llvm::errs() << "  max_filter[" << i << "] : "
                            << std::to_string(max_filter[i])
                            << ", bias[" << i << "] : "
                            << std::to_string(max_bias[i])
                            << ", rshift_per_channel[" << i
                            << "] : " << std::to_string(rshift_per_channel[i])
                            << "\n";);
  }

  // quantize weight
  for (int64_t i = 0; i < oc; ++i) {
    for (int64_t j = 0; j < isz; ++j) {
      new_filter[isz * i + j] = (float)quantizeFilterRShift(
          filter[isz * i + j], threshold_y, threshold_x, rshift_per_channel[i]);
    }
    if (bias) {
      new_bias[i] = (float)quantizeBiasRShiftI32(bias[i], threshold_y,
                                                 rshift_per_channel[i]);
    }
  }
}

void quantizeWeightInt8PerLayerMultiplier(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias,
    float *rshift_per_layer, float *multiplier_per_layer) {
  auto max_bias = std::vector<float>(oc);

  // find qscale
  float max_filter = findMaxWeight(filter, oc * isz);
  double qscale =
      findQScaleForFilter(max_filter, threshold_y, threshold_x);
  if (bias) {
    for (auto i = 0; i < oc; ++i){
      max_bias[i] = fabs(bias[i]);
      double qscale_bias = findQScaleForBiasI32(max_bias[i], threshold_y);
      if (qscale_bias > qscale) {
        LLVM_DEBUG(llvm::errs() << "WARNING: adjust qscale for bias"
                     << ", qscale_filter = " << qscale
                     << ", qscale_bias = " << qscale_bias << "\n";);
        qscale = qscale_bias;
      }
    }
  }
  // decompose qscale into rshift and muliplier
  uint32_t multiplier;
  rshift_per_layer[0] =
      (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true, 255);
  multiplier_per_layer[0] = (float)multiplier;
  for (auto i = 0; i < oc; ++i) {
    LLVM_DEBUG(llvm::errs()
                   << "  max_filter: " << std::to_string(max_filter)
                   << ", max_bias[" << i
                   << "] : " << std::to_string(max_bias[i])
                   << ", qscale : " << qscale << "  [multiplier : rshift]= ["
                   << std::to_string(multiplier_per_layer[0]) << " : "
                   << std::to_string(rshift_per_layer[0]) << "]\n";);
  }
  // quantize weight
  for (int64_t i = 0; i < oc * isz; ++i) {
    new_filter[i] = (float)quantizeFilterRShiftAndMultiplier(
        filter[i], threshold_y, threshold_x, rshift_per_layer[0],
        multiplier_per_layer[0], true);

    if (bias) {
      for (int64_t i = 0; i < oc; ++i) {
        new_bias[i] = (float)quantizeBiasRShiftAndMultiplier(
            bias[i], threshold_y, rshift_per_layer[0], multiplier_per_layer[0],
            true);
      }
    }
  }
}

void quantizeWeightInt8Multiplier(float *filter, float *bias,
    int64_t oc, int64_t isz, float threshold_y, float threshold_x,
    float *new_filter, float *new_bias,
    float *rshift_per_channel, float *multiplier_per_channel) {
  auto max_filter = std::vector<float>(oc);
  auto max_bias = std::vector<float>(oc);
  for (int64_t i = 0; i < oc; ++i) {
    // find qscale
    max_filter[i] = findMaxWeight(&filter[isz * i], isz);
    double qscale = findQScaleForFilter(max_filter[i], threshold_y, threshold_x);
    if(qscale >= 1){
      // Now 1880v2 not support lshift, if qscale > 1, rshift <= 0 not working now
      // we fix threshold_w to limit value
      // qscale = (thr_w * thr_x) / (127.0 * thr_y)
      // thr_w = qscale * 127.0 * thr_y / thr_x
      // qscale = 0.99999999
      qscale = 0.999999;
      max_filter[i] = qscale * 127.0 * threshold_y / threshold_x;
      LLVM_DEBUG(llvm::errs() << "WARNING: adjust threshold_w for qscale"
                   << ", qscale_filter = " << qscale << ", max_filter[" << i
                   << "] = " << max_filter[i] << "\n";);
    }
    max_bias[i] = 0.0f;
    if (bias) {
      max_bias[i] = fabs(bias[i]);
      double qscale_bias = findQScaleForBiasI32(max_bias[i], threshold_y);
      if (qscale_bias > qscale) {
        LLVM_DEBUG(llvm::errs() << "WARNING: adjust qscale for bias"
                     << ", qscale_filter = " << qscale
                     << ", qscale_bias = " << qscale_bias
                     << "\n";);
        if (qscale_bias >= 1) {
          // prevent for auto tuning
          LLVM_DEBUG(llvm::errs() << "WARNING:  qscale_bias are valid, keep org qscale"
            << ", qscale_filter = " << qscale
            << ", qscale_bias = " << qscale_bias
            << "\n";);
        }
        else {
          qscale = qscale_bias;
        }
      }
    }

    // decompose qscale into rshift and muliplier
    uint32_t multiplier;
    rshift_per_channel[i] =
        (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true);
    multiplier_per_channel[i] = (float)multiplier;

    LLVM_DEBUG(llvm::errs() << "  max_filter[" << i << "] : "
                            << std::to_string(max_filter[i])
                            << ", max_bias[" << i << "] : "
                            << std::to_string(max_bias[i])
                            << ", qscale : "
                            << qscale
                            << "  [multiplier : rshift][" << i << "] = ["
                            << std::to_string(multiplier_per_channel[i]) << " : "
                            << std::to_string(rshift_per_channel[i]) << "]\n";);
  }

  // quantize weight
  for (int i = 0; i < oc; ++i) {
    for (int j = 0; j < isz; ++j) {
      new_filter[isz * i + j] = (float)quantizeFilterRShiftAndMultiplier(
          filter[isz * i + j], threshold_y, threshold_x, rshift_per_channel[i],
          multiplier_per_channel[i], true);
    }
    if (bias) {
      new_bias[i] = (float)quantizeBiasRShiftAndMultiplier(
          bias[i], threshold_y, rshift_per_channel[i],
          multiplier_per_channel[i], true);
    }
  }
}

static inline signed char float2int8(float v, int mode = 0)
{
  if (mode == 0) {
    int int32 = std::round(v);
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
  } else {
    int int32 = 0;
    float fraction, integer;
    float abs_v = std::abs(v);
    fraction = std::modf(abs_v, &integer);
    int32 = (int)integer;
    if (fraction > 0.5) {
      int32 = int32 + 1;
    } else if (fraction == 0.5) {
      if (int32 & 0x01) {
        int32 = int32 + 1;
      }
    }

    if (v < 0) int32 = -int32;

    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
  }
}

/// Quantize an Activation tensor into INT8, given threshold
void quantizeActivationInt8WithThreshold(float *output, float *input,
    int64_t size, float threshold, bool tpu_mode, int zero_point) {
  float scale = 128.0 / threshold;

  if (tpu_mode) {
    if (zero_point != 0) {
      assert("zero_point in tpu mode not ready, todo");
    }
    bfloat16 bf_scale, bf_tmp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);

    for (int64_t i = 0; i < size; ++i) {
      // note this is using std::round() rather than floor(v+0.5f)
      // to compliance with NEON implemention on runtime
      // output[i] = (float)saturateInt8(input[i] * 128.0 / threshold);
      float f_tmp = input[i];
      // remove [17:31] mantissa part
      FloatToBFloat16(&f_tmp, &bf_tmp, 1, false);

      f_tmp = BFloat16ToFloat(bf_tmp);
      f_tmp = f_tmp * scale;
      // align backend
      bf_tmp = FloatToBFloat16(f_tmp);
      f_tmp = BFloat16ToFloat(bf_tmp);
      output[i] = (float)float2int8(f_tmp, 1);
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      float scale = 128.0 / threshold;
      // note this is using std::round() rather than floor(v+0.5f)
      // to compliance with NEON implemention on runtime
      // output[i] = (float)saturateInt8(input[i] * 128.0 / threshold);
      output[i] = (float)float2int8((input[i] * scale) + zero_point, 0);
    }
  }
}
/// DeQuantize an Activation tensor from INT8, given threshold
void dequantizeActivationInt8WithThreshold(float *output, float *input,
    int64_t size, float threshold, bool tpu_mode, int zero_point) {
  float scale = threshold / 128.0;
  if (tpu_mode) {
    if (zero_point != 0) {
      assert("zero_point in tpu mode not ready, todo");
    }
    bfloat16 bf_scale, bf_tmp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);

    for (int64_t i = 0; i < size; ++i) {
      // i8->bf16
      bf_tmp = FloatToBFloat16(input[i]);
      float fp_tmp = BFloat16ToFloat(bf_tmp);

      // bf16 mul scale
      fp_tmp = fp_tmp * scale;

      // bf16 -> fp32
      bf_tmp = FloatToBFloat16(fp_tmp);
      output[i] = BFloat16ToFloat(bf_tmp);
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      float scale = threshold / 128.0;
      output[i] = (input[i] + zero_point) * scale;
    }
  }
}

static uint8_t float_isnan(const float x) {
  //return isnan(x);
  return x != x;
}

/// HW float to bfloat16
bfloat16 FloatToBFloat16(float value)
{
  if (float_isnan(value))
    return 0x7FC0 /*NAN_VALUE*/;

  float f32_val = value;
  uint32_t* u32_val = reinterpret_cast<uint32_t*>(&f32_val);
  uint32_t input = *u32_val;
  uint32_t lsb = (input >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  bfloat16 bf_val = (bfloat16)(input >> 16);

  /* HW behavior */
  if ((bf_val & 0x7f80) == 0x7f80) {
    bf_val = 0x7f7f;
  }
  return bf_val;
}

float BFloat16ToFloat(bfloat16 value)
{
  float dst = 0;
  uint16_t* p = reinterpret_cast<uint16_t*>(&value);
  uint16_t* q = reinterpret_cast<uint16_t*>(&dst);

  q[0] = 0;
  q[1] = *p;

  return dst;
}

/// Quantize an Bf16 Activation tensor into INT8, given threshold
/// Keep interpreter bf16 quant align with TPU
/// TPU HW round mode support 0: round to nearest even, 1: round to zero
void quantizeActivationFromBf16ToInt8WithThreshold(float *output, float *input,
    int64_t size, float threshold) {
    float scale = 128.0 / threshold;
    bfloat16 bf_scale, bf_tmp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);
    for (int64_t i = 0; i < size; ++i) {
      float f_tmp = input[i] * scale;
      bf_tmp = FloatToBFloat16(f_tmp);
      f_tmp = BFloat16ToFloat(bf_tmp);
      output[i] = (float)float2int8(f_tmp, 1);
    }
}

/// Dequant an Int8 Activation tensor to Bf16, given threshold
/// Keep interpreter int8 quant align with TPU
void dequantizeActivationFromInt8ToBf16WithThreshold(float *output, float *input,
    int64_t size, float threshold) {
    float scale = threshold / 128.0;
    bfloat16 bf_scale;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);
    for (int64_t i = 0; i < size; ++i) {
      bfloat16 out = FloatToBFloat16(input[i] * scale);
      output[i] = (float)BFloat16ToFloat(out);
    }
}

/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerLayerRshift(float *output, float *input,
    int64_t size, uint32_t rshift) {
  for (int64_t i = 0; i < size; ++i) {
    output[i] = (float)applyRShiftAndSaturateInt8(input[i], rshift);
  }
}

/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerChannelRShift(float *output, float *input,
    int64_t on, int64_t oc, int64_t isz, float *rshift_per_channel) {
  for (int64_t n = 0; n < on; ++n) {
    for (int64_t i = 0; i < oc; ++i) {
      for (int64_t j = 0; j < isz; ++j) {
        output[n * oc * isz +i * isz + j] = (float)applyRShiftAndSaturateInt8(
            input[n * oc * isz +i * isz + j], rshift_per_channel[i]);
      }
    }
  }
}

/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerChannelMultiplierAndRShift(
    float *output, float *input, float *bias, bool do_relu, int64_t on, int64_t oc,
    int64_t isz, float *rshift_per_channel, float *multiplier_per_channel, int output_offset) {
  for (int64_t n = 0; n < on; ++n) {
    for (int64_t i = 0; i < oc; ++i) {
      for (int64_t j = 0; j < isz; ++j) {
        int32_t v = (int32_t)input[n * oc * isz + i * isz + j];
        if (bias != NULL) {
          v += (int32_t)bias[i];
        }
        v = applyMultiplierAndRShiftAndSaturateInt8(
                v, rshift_per_channel[i], multiplier_per_channel[i], true, output_offset);
        if (do_relu && (v < 0)) {
          v = 0;
        }
        output[n * oc * isz + i * isz + j] = (float)v;
      }
    }
  }
}

} // namespace
