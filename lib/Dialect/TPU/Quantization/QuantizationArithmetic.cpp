#include "tpuc/QuantizationArithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
uint32_t findRShiftForFilter(float max_filter, float threshold_y,
                             float threshold_x) {
  assert(threshold_y > 0 && threshold_x > 0);
  float a = max_filter * threshold_x / threshold_y;
  if (a > 127) {
    LLVM_DEBUG(llvm::errs()
                   << "WARNING: findRShiftForFilter, max_filter too large "
                   << std::to_string(max_filter) << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 127);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ((a * (1 << rshift)) >= 64) {
      LLVM_DEBUG(if (rshift >= 25) {
        llvm::errs() << "WARNING: findRShiftForFilter, large rshift = "
                     << rshift << ", max_filter = " << max_filter
                     << ", threshold_y = " << std::to_string(threshold_y)
                     << ", threshold_x = " << std::to_string(threshold_x)
                     << "\n";
      });
      return rshift;
    }
  }
  // we are here because a < 64 / (1 << 32), which mean max_filter is near zero
  // assert(false);
  LLVM_DEBUG(
      llvm::errs() << "WARNING: findRShiftForFilter, max_filter too small\n";);
  // return 0, to make sure activation will go zero by multiply such small
  // weight
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
    LLVM_DEBUG(llvm::errs()
                   << "WARNING: findRShiftForBiasI16, max_bias too large "
                   << std::to_string(max_bias) << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 32767);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ((a * (1 << rshift)) >= 16384) {
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
    LLVM_DEBUG(llvm::errs()
                   << "WARNING: findRShiftForBiasI32, max_bias too large "
                   << std::to_string(max_bias) << ", lshift might needed\n";);
    return 0;
  }
  assert(a <= 0x7fffffff);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ((a * (1 << rshift)) >= 0x40000000) {
      LLVM_DEBUG(if (rshift < 25) {
        llvm::errs() << "WARNING: findRShiftForBiasI32, find rshift = "
                     << rshift << ", max_bias = " << std::to_string(max_bias)
                     << ", threshold_y = " << std::to_string(threshold_y)
                     << "\n";
      });
      return rshift;
    }
  }
  // we are here because a < 0x40000000 / (1 << 32), which mean max_bias is
  // small enough llvm::errs() << "WARNING: findRShiftForBiasI32, max_bias small
  // enough\n"; return 31, to not limiting final rshift (min(rshift_filter,
  // rshift_bias))
  return 31;
}

inline static float MAX_QUANT(int quant_bitwidth){
  return (float)((1 << (quant_bitwidth-1)) -1);
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
double findQScaleForFilter(float max_filter, float threshold_y,
                           float threshold_x, int quant_bitwidth) {
  // assert(threshold_y > 0 && threshold_x > 0);
  assert(threshold_y > 0);
  double qscale = (max_filter * threshold_x) / (MAX_QUANT(quant_bitwidth)* threshold_y);
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
void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier,
                        int *shift) {
  assert(double_multiplier >= 0);
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
int8_t findRShiftAndMultiplierFromQScale(double qscale, uint32_t *multiplier,
                                         bool qdm, uint32_t max_multiplier) {
  if (qdm) {
    // this ensures if qscale is 0, both multiplier and shift will be 0
    int32_t quantized_multiplier = 0;
    int lshift = 0;
    if (qscale >= 1) {
      qscale = 0.999999;
    }
    Tensorflow_QuantizeMultiplier(qscale, &quantized_multiplier, &lshift);
    *multiplier = quantized_multiplier;
    int rshift = -lshift;
    assert(rshift >= 0);
    LLVM_DEBUG(if (rshift > 25) {
      llvm::errs() << "WARNING: large rshift = " << rshift
                   << ", qscale = " << qscale << "\n";
    });
    return (int8_t)rshift;
  } else {
    if (qscale > max_multiplier) {
      llvm::errs() << "Error: qscale > max_multipiler ( " << qscale << " v.s. "
                   << max_multiplier << " )\n";
      //assert(false);
      *multiplier = max_multiplier;
      return 0;
    }
    for (int8_t rshift = 0; rshift < 63; ++rshift) {
      if (((double)qscale * (1ULL << (rshift + 1))) >= (double)max_multiplier) {
        if (multiplier) {
          *multiplier = (uint32_t)((double)qscale * (1ULL << rshift));
        }
        return rshift;
      }
    }
    // assert(false);
    LLVM_DEBUG(llvm::errs() << "WARNING: failed to find rshift, qscale = "
                            << std::to_string(qscale) << "\n";);
    // we are here because qscale is too small, return 0 for both shift and
    // multiplier
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
  // assert( (q <= 127) && (q >= -128) );
  DEBUG_WITH_TYPE(
      DEBUG_TYPE "_WARNING", if ((q > 127) || (q < -128)) {
        llvm::errs() << "exceeds limits [-128, 127] : " << std::to_string(f)
                     << "\n";
      });
  if (q > 127)
    q = 127;
  if (q < -128)
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
  LLVM_DEBUG(if ((q > 32767) || (q < -32768)) {
    llvm::errs() << "WARNING: exceeds limits [-32768, 32767] : "
                 << std::to_string(f) << "\n";
  });
  // assert( (q <= 32767) && (q >= -32768) );
  if (q > 32767)
    q = 32767;
  if (q < -32768)
    q = -32768;
  return (int16_t)q;
}

/// saturate a float to int
static inline int32_t saturateInt32(float f) {
  LLVM_DEBUG(if ((f > INT_MAX) || (f < INT_MIN)) {
    llvm::errs() << "WARNING: exceeds INT limits : " << std::to_string(f)
                 << "\n";
  });
  assert((f <= INT_MAX) && (f >= INT_MIN));
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
  LLVM_DEBUG(if ((q_f > INT_MAX) || (q_f < INT_MIN)) {
    llvm::errs() << "WARNING: quantizeBiasRShiftI16, exceeds INT limits : "
                 << std::to_string(q_f) << "\n";
    llvm::errs() << "  w: " << std::to_string(w)
                 << ", threshold_y: " << std::to_string(threshold_y)
                 << ", rshift: " << std::to_string(rshift) << "\n";
  });
  return saturateInt16(q_f);
}

/// quantize a bias weight value into int32
///   Q(B) = B * (128.0f / threshold_y) * (1 << rshift)
/// used in BM1880v2 per-channel mode (32bit bias), by with no multiplier
int32_t quantizeBiasRShiftI32(float w, float threshold_y, uint32_t rshift) {
  double factor = (128.0f / threshold_y) * (1 << rshift);
  float q_f = (float)(w * factor);
  LLVM_DEBUG(if ((q_f > INT_MAX) || (q_f < INT_MIN)) {
    llvm::errs() << "WARNING: quantizeBiasRShiftI32, exceeds INT limits : "
                 << std::to_string(q_f) << "\n";
    llvm::errs() << "  w: " << std::to_string(w)
                 << ", threshold_y: " << std::to_string(threshold_y)
                 << ", rshift: " << std::to_string(rshift) << "\n";
  });
  return saturateInt32(q_f);
}

/// quantize a filter weight value into int8 based on rshift and multiplier
///   Q(W) = W * (threshold_x / threshold_y) * (1 / QScale)
///   QScale = Multiplier / (1 << RShift)
/// used in BM1880 or BM1880v2 legacy per-layer mode
int8_t quantizeFilterRShiftAndMultiplier(float w, float threshold_y,
                                         float threshold_x, uint32_t rshift,
                                         uint32_t multiplier, bool qdm) {
  if (qdm) {
    rshift += 31;
  }
  double factor = (multiplier == 0) ? 0
                                    : (double)(threshold_x / threshold_y) *
                                          (1ULL << rshift) / multiplier;
  float q_f = (float)(w * factor);
  return saturateInt8(q_f);
}

/// quantize a bias weight value into int32 based on rshift and multiplier
///   Q(B) = B * (128.0f / threshold_y) * (1 / QScale)
///   QScale = Multiplier * (1 << RShift)
/// used in BM1880v2 per-channel mode (32bit bias)
int32_t quantizeBiasRShiftAndMultiplier(float w, float threshold_y,
                                        uint32_t rshift, uint32_t multiplier,
                                        bool qdm) {
  if (qdm) {
    rshift += 31;
  }
  double factor = (multiplier == 0) ? 0
                                    : (double)(128.0f / threshold_y) *
                                          (1ULL << rshift) / multiplier;
  float q_f = (float)(w * factor);
  LLVM_DEBUG(if ((q_f > INT_MAX) || (q_f < INT_MIN)) {
    llvm::errs() << "WARNING: quantizeBiasRShiftI32, exceeds INT limits : "
                 << std::to_string(q_f) << "\n";
    llvm::errs() << "  w: " << std::to_string(w)
                 << ", threshold_y: " << std::to_string(threshold_y)
                 << ", multiplier: " << multiplier
                 << ", rshift: " << std::to_string(rshift) << "\n";
  });
  return saturateInt32(q_f);
}

/// Simulate HW behavior, after accumuation, do rshift and saturate
int8_t applyRShiftAndSaturateInt8(float v, uint32_t rshift) {
  return saturateInt8((v / (1 << rshift)));
}

// USE_GOOGLE_GEMMLOWP_QDM
typedef int32_t s32;
static inline s32 RoundingDivideByPOT(s32 x, int exponent) {
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

static inline s32 SaturatingRoundingDoublingHighMul(s32 a, s32 b) {
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
int8_t applyMultiplierAndRShiftAndSaturateInt8(float v, uint32_t rshift,
                                               uint32_t multiplier, bool qdm) {
  if (qdm) {
    int32_t q = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul((int32_t)v, (int32_t)multiplier),
        rshift);
    return saturateInt8((float)(q));
  } else {
    return saturateInt8((float)(((v * multiplier)) / (1 << rshift)));
  }
}

int8_t applyMultiplierAndRShiftAndSaturateInt8(int32_t v, uint32_t rshift,
                                               uint32_t multiplier, bool qdm) {
  if (qdm) {
    int32_t q = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(v, (int32_t)multiplier), rshift);
    return saturateInt8((float)(q));
  } else {
    return saturateInt8((float)(((v * multiplier)) / (1 << rshift)));
  }
}


//
// Tensors wise API
//
void quantizeWeightInt8PerLayer(float *filter, float *bias, int64_t oc,
                                int64_t isz, float threshold_y,
                                float threshold_x, float *new_filter,
                                float *new_bias, float *rshift_per_layer) {
  // find rshift
  float max_filter = findMaxWeight(filter, oc * isz);
  float rshift_filter =
      (float)findRShiftForFilter(max_filter, threshold_y, threshold_x);
  rshift_per_layer[0] = rshift_filter;
  float max_bias = 0.0f;
  if (bias) {
    float max_bias = findMaxWeight(bias, oc);
    float rshift_bias = (float)findRShiftForBiasI16(max_bias, threshold_y);
    if (rshift_bias < rshift_filter) {
      LLVM_DEBUG(llvm::errs()
                     << "WARNING: adjust rshift for bias"
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

void quantizeWeightInt8PerChannel(float *filter, float *bias, int64_t oc,
                                  int64_t isz, float threshold_y,
                                  float threshold_x, float *new_filter,
                                  float *new_bias, float *rshift_per_channel) {
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
      float rshift_bias = (float)findRShiftForBiasI32(max_bias[i], threshold_y);
      if (rshift_bias < rshift_filter) {
        LLVM_DEBUG(llvm::errs()
                       << "WARNING: adjust rshift for bias"
                       << ", rshift_filter = " << std::to_string(rshift_filter)
                       << ", rshift_bias = " << std::to_string(rshift_bias)
                       << "\n";);
        rshift_per_channel[i] = rshift_bias;
      }
    }
    LLVM_DEBUG(llvm::errs()
                   << "  max_filter[" << i
                   << "] : " << std::to_string(max_filter[i]) << ", bias[" << i
                   << "] : " << std::to_string(max_bias[i])
                   << ", rshift_per_channel[" << i
                   << "] : " << std::to_string(rshift_per_channel[i]) << "\n";);
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

void quantizeWeightInt8ForFC(float *filter, float *bias, int64_t batch,
                             int64_t N, int64_t K, float threshold_y,
                             float threshold_x, float *new_filter,
                             float *new_bias, float *rshift_per_batch,
                             float *multiplier_per_batch) {

  // find qscale
  std::vector<float> max_filter(batch);
  std::vector<double> qscale(batch);
  int64_t isz = N * K;
  for (int i = 0; i < batch; i++) {
    max_filter[i] = findMaxWeight(filter + i * isz, isz);
    qscale[i] = findQScaleForFilter(max_filter[i], threshold_y, threshold_x);
  }

  std::vector<float> max_bias(batch * N);
  if (bias) {
    for (int i = 0; i < batch; i++) {
      for (int n = 0; n < N; ++n) {
        int index = i * N + n;
        max_bias[index] = fabs(bias[index]);
        double qscale_bias = findQScaleForBiasI32(max_bias[index], threshold_y);
        if (qscale_bias > qscale[i]) {
          LLVM_DEBUG(llvm::errs()
                         << "WARNING: adjust qscale for bias"
                         << ", qscale_filter = " << qscale[i]
                         << ", qscale_bias = " << qscale_bias << "\n";);
          qscale[i] = qscale_bias;
        }
      }
    }
  }
  // decompose qscale into rshift and muliplier
  uint32_t multiplier;
  for (int i = 0; i < batch; i++) {
    rshift_per_batch[i] = (float)findRShiftAndMultiplierFromQScale(
        qscale[i], &multiplier, true, 255);
    multiplier_per_batch[i] = (float)multiplier;
  }

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < isz; ++j) {
      int index = i * isz + j;
      new_filter[index] = (float)quantizeFilterRShiftAndMultiplier(
          filter[index], threshold_y, threshold_x, rshift_per_batch[i],
          multiplier_per_batch[i], true);
    }
  }

  if (bias) {
    for (int i = 0; i < batch; i++) {
      for (int n = 0; n < N; ++n) {
        int index = i * N + n;
        new_bias[index] = (float)quantizeBiasRShiftAndMultiplier(
            bias[index], threshold_y, rshift_per_batch[i],
            multiplier_per_batch[i], true);
      }
    }
  }
}

// ONLY deal with bias w/o weight and plz refer
void quantizeBiasInt8PerLayerMultiplier(float *bias, int64_t oc, int64_t isz,
                                        float threshold_y, float threshold_x,
                                        float *new_filter, float *new_bias,
                                        float *rshift_per_layer,
                                        float *multiplier_per_layer,
                                        double qscale, bool qdm) {

  if (!bias) {
    return;
  }

  auto max_bias = std::vector<float>(oc);

  // find qscale
  float max_filter = 0;

  for (auto i = 0; i < oc; ++i) {
    max_bias[i] = fabs(bias[i]);
    double qscale_bias = findQScaleForBiasI32(max_bias[i], threshold_y);
    if (qscale_bias > qscale) {
      LLVM_DEBUG(llvm::errs() << "WARNING: adjust qscale for bias"
                              << ", qscale_filter = " << qscale
                              << ", qscale_bias = " << qscale_bias << "\n";);
      qscale = qscale_bias;
    }
  }

  // decompose qscale into rshift and muliplier
  uint32_t multiplier;
  rshift_per_layer[0] =
      (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, qdm, 255);
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

  for (int64_t i = 0; i < oc; ++i) {
    new_bias[i] = (float)quantizeBiasRShiftAndMultiplier(
        bias[i], threshold_y, rshift_per_layer[0], multiplier_per_layer[0],
        qdm);
  }
}

void quantizeWeightInt8Multiplier(float *filter, float *bias, int64_t oc,
                                  int64_t isz, float threshold_y,
                                  float threshold_x, float *new_filter,
                                  float *new_bias, float *rshift_per_channel,
                                  float *multiplier_per_channel,
                                  std::vector<float> &filter_threshold, int quant_bitwidth) {

  std::vector<float> max_filter(oc);
  bool use_filter_threshold = filter_threshold.size() > 0;

  auto max_bias = std::vector<float>(oc);
  for (int64_t i = 0; i < oc; ++i) {
    // find qscale

    max_filter[i] = use_filter_threshold ? filter_threshold[i]
                                         : findMaxWeight(&filter[isz * i], isz);
    double qscale =
        findQScaleForFilter(max_filter[i], threshold_y, threshold_x, quant_bitwidth);
    if (qscale >= 1) {
      // Now 1880v2 not support lshift, if qscale > 1, rshift <= 0 not working
      // now we fix threshold_w to limit value qscale = (thr_w * thr_x) / (127.0
      // * thr_y) thr_w = qscale * 127.0 * thr_y / thr_x qscale = 0.99999999
      qscale = 0.999999;
      max_filter[i] = qscale * MAX_QUANT(quant_bitwidth) * threshold_y / threshold_x;
      LLVM_DEBUG(llvm::errs()
                     << "WARNING: adjust threshold_w for qscale"
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
                                << ", qscale_bias = " << qscale_bias << "\n";);
        if (qscale_bias >= 1) {
          // prevent for auto tuning
          LLVM_DEBUG(llvm::errs()
                         << "WARNING:  qscale_bias are valid, keep org qscale"
                         << ", qscale_filter = " << qscale
                         << ", qscale_bias = " << qscale_bias << "\n";);
        } else {
          qscale = qscale_bias;
        }
      }
    }

    // decompose qscale into rshift and muliplier
    uint32_t multiplier;
    rshift_per_channel[i] =
        (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true);
    multiplier_per_channel[i] = (float)multiplier;

    LLVM_DEBUG(llvm::errs()
                   << "  max_filter[" << i
                   << "] : " << std::to_string(max_filter[i]) << ", max_bias["
                   << i << "] : " << std::to_string(max_bias[i])
                   << ", qscale : " << qscale << "  [multiplier : rshift][" << i
                   << "] = [" << std::to_string(multiplier_per_channel[i])
                   << " : " << std::to_string(rshift_per_channel[i]) << "]\n";);
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

static inline int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}


/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerLayerRshift(float *output, float *input,
                                          int64_t size, uint32_t rshift) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (int64_t i = 0; i < size; ++i) {
    output[i] = (float)applyRShiftAndSaturateInt8(input[i], rshift);
  }
}

/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerChannelRShift(float *output, float *input,
                                            int64_t on, int64_t oc, int64_t isz,
                                            float *rshift_per_channel) {
  for (int64_t n = 0; n < on; ++n) {
    for (int64_t i = 0; i < oc; ++i) {
      for (int64_t j = 0; j < isz; ++j) {
        output[n * oc * isz + i * isz + j] = (float)applyRShiftAndSaturateInt8(
            input[n * oc * isz + i * isz + j], rshift_per_channel[i]);
      }
    }
  }
}

/// Quantize an Activation tensor, given per channel mulitplier and rshift
void quantizeActivationInt8PerChannelMultiplierAndRShift(
    float *output, float *input, float *bias, bool do_relu, int64_t on,
    int64_t oc, int64_t isz, float *rshift_per_channel,
    float *multiplier_per_channel) {
#pragma omp parallel for collapse(3)
  for (int64_t n = 0; n < on; ++n) {
    for (int64_t i = 0; i < oc; ++i) {
      for (int64_t j = 0; j < isz; ++j) {
        int32_t v = (int32_t)input[n * oc * isz + i * isz + j];
        if (bias != NULL) {
          v += (int32_t)bias[i];
        }
        v = applyMultiplierAndRShiftAndSaturateInt8(v, rshift_per_channel[i],
                                                    multiplier_per_channel[i],
                                                    true);
        if (do_relu && (v < 0)) {
          v = 0;
        }
        output[n * oc * isz + i * isz + j] = (float)v;
      }
    }
  }
}

bfloat16 F32ToBF16(float src, bool rounding) {
  uint16_t u16_val;
  if (rounding) {
    uint32_t u32_val = *((uint32_t *)(&src));
    uint32_t lsb = (u32_val >> 16) & 1;
    u32_val += (0x7fff + lsb);
    u16_val = ((uint16_t *)(&u32_val))[1];
    /* HW behavior */
    // infinity set to max finite positive value
    u16_val = ((u16_val & 0x7f80) == 0x7f80) ?
              0x7f7f : u16_val;
  } else {
    u16_val = ((uint16_t *)(&src))[1];
  }
  return u16_val;
}

void F32ToBF16(float *src, bfloat16 *dst, size_t size, bool rounding) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = F32ToBF16(src[i], rounding);
  }
}

static int32_t F32ToInt(float v, int round_mode) {
  int32_t i32_val;
  if (round_mode == 1) { // round to zero
    i32_val = (int)v;
  } else { // round to nearest even
    float fraction, integer;
    float abs_v = std::abs(v);
    fraction = std::modf(abs_v, &integer);
    i32_val = (int)integer;
    if (fraction > 0.5) {
      i32_val = i32_val + 1;
    } else if (fraction == 0.5) {
      if (i32_val & 0x01) {
        i32_val = i32_val + 1;
      }
    }
    if (v < 0)
      i32_val = -i32_val;
  }
  return i32_val;
}

int8_t F32ToInt8(float v, int round_mode) {
  int32_t i32_val = F32ToInt(v, round_mode);
  if (i32_val > 127)
    return 127;
  if (i32_val < -128)
    return -128;
  return (int8_t)i32_val;
}

uint8_t F32ToUint8(float v, int round_mode) {
  int32_t i32_val = F32ToInt(v, round_mode);
  if (i32_val > 255)
    return 255;
  if (i32_val < 0)
    return 0;
  return (uint8_t)i32_val;
}

float BF16(float src, bool rounding) {
  float dst = 0;
  uint16_t u16_val = F32ToBF16(src, rounding);
  uint16_t *p = (uint16_t *)(&dst);
  p[1] = u16_val;
  return dst;
}

void BF16(float *src, float *dst, size_t size, bool rounding) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = BF16(src[i], rounding);
  }
}


} // namespace mlir
