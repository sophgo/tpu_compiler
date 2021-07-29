#include "tpuc/MachineInfo.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "tpuc/NativeCpuImplementation.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>
#include <functional>
#include <assert.h>
#include <string>
#include <climits>
#include <functional>
#include <math.h>
// align cmodel
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#define DEBUG_TYPE "lut_func"

typedef float (*activate_func)(float, float);

static float sigmoid_actf(float x, float p) { return 1 / (1 + expf(-x)); }
static float tanh_actf(float x, float p) { return tanh(x); }
static float exp_actf(float x, float p) { return expf(x) + p; }
static float elu_actf(float x, float p) { return (x >=0) ? x : (expf(x) -1); }
static float softplus_actf(float x, float p) { return logf(expf(x) + 1); }
static float swish_actf(float x, float p) { return x / (1 + expf(-x)); }
static float mish_actf(float x, float p) { return x * tanh(softplus_actf(x, p)); }


static inline float BF16(float data) {
  return convert_bf16_fp32(convert_fp32_bf16(data));
}

static void gen_bf16_base_table(float start, float end, int table_hw,
                                float *table, activate_func func, float extra_param) {
  int half = table_hw / 2;
  int range = abs(end - start);
  float interval = (float)range / (float)table_hw;
  float x_value;
  float y_value;
  float offset = (start + end) / 2;
  assert(offset == 0);

  // Set idx [0 , 127] fp32 and bf16 data
  for (int i = 0; i < half; i++) {
    x_value = offset + i * interval;
    y_value = func(x_value, extra_param);
    table[i] = y_value;
  }

  // set idx 129 to 255, 2's complment
  for (int i = half, j = 0; i < table_hw; i++, j++) {
    x_value = start + j * interval;
    y_value = func(x_value, extra_param);
    table[i] = y_value;
  }
}

static void gen_bf16_slope_table(float start, float end, int table_hw, float *table,
                                 float *slope_table, activate_func func, float extra_param) {
  int half = table_hw / 2;
  float scale = ((float)table_hw) / (end - start);

  // positive axis, slope = x(i+1) - x(i)
  for (int i = 0; i < half - 1; i++) {
    auto x0 = table[i];
    auto x1 = table[i + 1];
    if (convert_fp32_bf16(x0) == convert_fp32_bf16(x1)) {
      slope_table[i] = 0;
    } else {
      slope_table[i] = x1 - x0;
    }
  }
  // slope of range end
  slope_table[half - 1] = (func(3 * end, extra_param) - func(end, extra_param)) / (2 * end * scale);

  // negtive axis, slope = x(i - 1) - x(i)
  for (int i = table_hw - 1; i > half; i--) {
    auto x0 = table[i];
    auto x1 = table[i - 1];
    if (convert_fp32_bf16(x0) == convert_fp32_bf16(x1)) {
      slope_table[i] = 0;
    } else {
      slope_table[i] = x0 - x1;
    }
  }
  // slope of range start
  slope_table[half] = (func(3 * start, extra_param) - func(start, extra_param)) / (std::abs(2 * start) * scale);
}

void bf16_gen_base_slope_table(const std::string &name, float *base_table, float *slope_table,
                               float &range_start, float &range_end, float extra_param) {
  activate_func func;
  if (name == "sigmoid") {
    range_start = -1 * SIGMOID_BF16_LUT_RANGE;
    range_end = SIGMOID_BF16_LUT_RANGE;
    func = sigmoid_actf;
  } else if (name == "tanh") {
    range_start = -1 * TANH_BF16_LUT_RANGE;
    range_end = TANH_BF16_LUT_RANGE;
    func = tanh_actf;
  } else if (name == "exp") {
    range_start = -1 * EXP_BF16_LUT_RANGE;
    range_end = EXP_BF16_LUT_RANGE;
    func = exp_actf;
  } else if (name == "elu") {
    range_start = -1 * ELU_BF16_LUT_RANGE;
    range_end = ELU_BF16_LUT_RANGE;
    func = elu_actf;
  } else if (name == "swish") {
    range_start = -1 * SWISH_BF16_LUT_RANGE;
    range_end = SWISH_BF16_LUT_RANGE;
    func = swish_actf;
  } else if (name == "mish") {
    range_start = -1 * MISH_BF16_LUT_RANGE;
    range_end = MISH_BF16_LUT_RANGE;
    func = mish_actf;
  } else if (name == "softplus") {
    range_start = -1 * SOFTPLUS_BF16_LUT_RANGE;
    range_end = SOFTPLUS_BF16_LUT_RANGE;
    func = softplus_actf;
  } else {
    llvm::errs() << "unsupported lookup table func:" << name << "\n";
    llvm_unreachable("Error");
  }
  gen_bf16_base_table(range_start, range_end, 256, base_table, func, extra_param);
  gen_bf16_slope_table(range_start, range_end, 256, base_table, slope_table, func, extra_param);
}

void bf16_lut_slope(const std::string &name, float *input, float *output, int size,
                    const std::vector<float> &base_table,
                    const std::vector<float> &slope_table) {
  float range_start;
  float range_end;
  if (name == "sigmoid") {
    range_start = -1 * SIGMOID_BF16_LUT_RANGE;
    range_end = SIGMOID_BF16_LUT_RANGE;
  } else if (name == "tanh") {
    range_start = -1 * TANH_BF16_LUT_RANGE;
    range_end = TANH_BF16_LUT_RANGE;
  } else if (name == "exp") {
    range_start = -1 * EXP_BF16_LUT_RANGE;
    range_end = EXP_BF16_LUT_RANGE;
  } else if (name == "elu") {
    range_start = -1 * ELU_BF16_LUT_RANGE;
    range_end = ELU_BF16_LUT_RANGE;
  } else if (name == "swish") {
    range_start = -1 * SWISH_BF16_LUT_RANGE;
    range_end = SWISH_BF16_LUT_RANGE;
  } else if (name == "mish") {
    range_start = -1 * MISH_BF16_LUT_RANGE;
    range_end = MISH_BF16_LUT_RANGE;
  } else if (name == "softplus") {
    range_start = -1 * SOFTPLUS_BF16_LUT_RANGE;
    range_end = SOFTPLUS_BF16_LUT_RANGE;
  } else {
    llvm::errs() << "unsupported lookup table func:" << name << "\n";
    llvm_unreachable("Error");
  }

  // interger index range
  // from 16(-8~8)->256(lut index size)
  float scale = BF16(256.0 / (range_end - range_start));
  float offset = BF16((range_end + range_start) / 2);

  for (int i = 0; i < size; ++i) {
    float rescale_bf16_input = BF16(BF16(input[i] - offset) * scale);
    // get interger part
    uint16_t rescale_input_bf16 = convert_fp32_bf16(rescale_bf16_input);
    int rescale_input_i8 = _convert_bf16_s8(rescale_input_bf16, /*int8_rnd_mode=*/1);
    // get delta x (x - x0)
    float delta_x = BF16(rescale_bf16_input - rescale_input_i8);
    // get slope
    auto slope = slope_table[rescale_input_i8 & 0xff];
    // base y0 = f(x0)
    auto base = base_table[rescale_input_i8 & 0xff];
    // result = y0 + delta * slope
    output[i] = BF16(base + delta_x * slope);
  }
}

// <! gen reciprocal f(x) = 1/x
static double _gen_reciprocal(int base, int p) {
  // y = x ^ -1/
  double f = (double) (pow(base, -1 * p));
  return f;
}

void bf16_gen_reciprocal(int start, int end, int table_hw, uint16_t *table_data) {

  int exp_start = start;
  int half = table_hw / 2;
  uint64_t idx = 0;

  // prepare channel 0
  // double s = 0.0;
  // 0^-1 is invalid, use positive/negtive max value: 0x7F7F / 0xFF7F
  table_data[idx] = 0x7F7F; //<! convert to 0x7F7F

  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    float exp = shift;

    double s = _gen_reciprocal(2, exp);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  table_data[idx] = 0xFF7F; //<! convert to 0x7F7F
  idx++;

  // < 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    float exp = shift;

    double s = -1 * _gen_reciprocal(2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_reciprocal_mantissa(int start, int end, int table_hw, uint16_t *table_mantissa) {
  int half = table_hw/2;

  int idx = 0;
  double d;
  for (int i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, -1);
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    //13=2^3x1.625=(2^2)x(2^1x1.625)
    table_mantissa[idx] = convert_fp32_bf16(d);
    idx++;
  }
}

// <! gen invert sqrt
static double _gen_sqrt(int base, int p) {
  // y = x ^ 0.5
  double f = (double) (pow(base, p * 0.5));
  return f;
}

void bf16_gen_sqrt(int start, int table_hw, uint16_t *table_data) {
  //<! 32*8 table, duplicate `channel` times;

  int half = table_hw / 2;
  uint64_t idx = 0;
  assert(half == 128);

  // prepare channel 0
  float s = 0.0;
  table_data[idx] = convert_fp32_bf16(s);
  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half; i++) {
    int shift = (start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = _gen_sqrt(2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_sqrt_mantissa(int table_hw, uint16_t* table_mantissa) {

  uint32_t half = table_hw  / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (uint32_t i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, 0.5);
    table_mantissa[128+idx] = convert_fp32_bf16(d);

    d = 2 * (1 + i * 1 / 128.0);

    d = (double) pow(d, 0.5);
    table_mantissa[idx] = convert_fp32_bf16(d);
    idx++;
  }
}

void bf16_gen_reciprocal_sqrt(int start, int table_hw, uint16_t *table_data) {
  //<! 32*8 table, duplicate `channel` times;

  int half = table_hw / 2;
  uint64_t idx = 0;
  assert(half == 128);

  // prepare channel 0
  float s = 0.0;
  table_data[idx] = convert_fp32_bf16(s);
  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half; i++) {
    int shift = (start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }
    double s = _gen_sqrt(2, exp);
    table_data[idx] = convert_fp32_bf16(1.0 / s);
    idx++;
  }
}

void bf16_gen_reciprocal_sqrt_mantissa(int table_hw, uint16_t* table_mantissa) {

  uint32_t half = table_hw  / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (uint32_t i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, 0.5);
    table_mantissa[128+idx] = convert_fp32_bf16(1.0 / d);
    d = 2 * (1 + i * 1 / 128.0);

    d = (double) pow(d, 0.5);
    table_mantissa[idx] = convert_fp32_bf16(1.0 / d);
    idx++;
  }
}

// gen power exp table
void bf16_gen_power_exp_table(uint16_t *table_data, float beta,
                              int start, int table_hw) {
  int exp_start = start;
  int half = table_hw/2;
  uint64_t idx = 0;

  table_data[idx] = 0x1; // power(0)
  idx++;

  // > 0, exp from -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = (double)(pow(2, (exp*(-beta))));
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  table_data[idx] = 1; // power(-0)
  idx++;

  // < 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = -1 * (double)(pow(2, (exp*(-beta))));
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_power_mantissa_table(uint16_t* table_mantissa, float beta,
                                   int table_hw) {
  int half = table_hw / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (int i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, (0-beta));
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx+128
                            <<"] = " <<table_mantissa[128+idx];);

    //13=2^3x1.625=(2^2)x(2^1x1.625)
    d = 2 * (1 + i * 1 / 128.0);
    d = (double) pow(d, (0-beta));
    table_mantissa[idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx
                            <<"] = " <<table_mantissa[idx];);
    idx++;
  }
}

void bf16_gen_exponent_mantissa_table(const std::string &name, uint16_t *exp_table,
                                      uint16_t *mantissa_table, float extra_param) {
  float range_start = -62;
  float range_end = 63;
  int table_hw = 256;
  if (name == "reciprocal") {
      bf16_gen_reciprocal(range_start, range_end, table_hw, exp_table);
      bf16_gen_reciprocal_mantissa(range_start, range_end, table_hw, mantissa_table);
  } else if (name == "sqrt") {
      bf16_gen_sqrt(range_start, table_hw, exp_table);
      bf16_gen_sqrt_mantissa(table_hw, mantissa_table);
  } else if (name == "reciprocal_sqrt") {
      bf16_gen_reciprocal_sqrt(range_start, table_hw, exp_table);
      bf16_gen_reciprocal_sqrt_mantissa(table_hw, mantissa_table);
  } else if (name == "power") {
      bf16_gen_power_exp_table(exp_table, extra_param, range_start, table_hw);
      bf16_gen_power_mantissa_table(mantissa_table, extra_param, table_hw);
  } else {
    llvm::errs() << "unsupported lookup table func:" << name << "\n";
    llvm_unreachable("Error");
  }
}

void bf16_lut_mantissa(float *input, float *output, int size,
                       const std::vector<float> &bf16_lut,
                       const std::vector<float> &bf16_mantissa_lut) {
  for (int i = 0; i < size; i++) {
    uint16_t bf16InputValue = convert_fp32_bf16(input[i]);
    float input_bf16 = convert_bf16_fp32(bf16InputValue);
    int exponentIndex;
    if (input_bf16 == 0) {
      exponentIndex = 0;
    } else if (input_bf16 >= 0) {
      exponentIndex = floor(log2(input_bf16));
      exponentIndex += 62 + 1; // 62 means start with 2^-62, index from 1
    } else {
      exponentIndex = floor(log2(-1 * input_bf16));
      exponentIndex += 62 + 129; // 62 means start with 2^-62, index from 129
    }
    float exponent = bf16_lut[exponentIndex];
    float mantissa = bf16_mantissa_lut[bf16InputValue & 0xff];
    output[i] = convert_bf16_fp32(convert_fp32_bf16(exponent * mantissa));
  }
}


