#include "tpuc/MachineInfo.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/QuantizationArithmetic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>
#include <functional>
#include <assert.h>
#include <string>
#include <climits>
#include <functional>
#include <math.h>

#define DEBUG_TYPE "lut_func"

typedef float (*activate_func)(float, float, float);

static float sigmoid_actf(float x, float s, float b) { return s / (1 + expf(-x)) + b; }
static float tanh_actf(float x, float p0, float p1) { return tanh(x); }
static float log_actf(float x, float p0, float p1) { return log(x); }
static float exp_actf(float x, float s, float b) { return s * expf(x) + b; }
static float elu_actf(float x, float p0, float p1) { return (x >=0) ? x : (expf(x) -1); }
static float softplus_actf(float x, float s, float b) { return s * logf(expf(x) + 1) + b; }
static float swish_actf(float x, float p0, float p1) { return x / (1 + expf(-x)); }
static float mish_actf(float x, float p0, float p1) { return x * tanh(softplus_actf(x, 1, 0)); }


static void gen_bf16_base_table(float start, float end, int table_hw, float *table,
                                activate_func func, float param0, float param1) {
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
    y_value = func(x_value, param0, param1);
    table[i] = BF16(y_value);
  }

  // set idx 129 to 255, 2's complment
  for (int i = half, j = 0; i < table_hw; i++, j++) {
    x_value = start + j * interval;
    y_value = func(x_value, param0, param1);
    table[i] = BF16(y_value);
  }
}

static void gen_bf16_slope_table(float start, float end, int table_hw,
                                 float *table, float *slope_table,
                                 activate_func func, float param0, float param1) {
  int half = table_hw / 2;
  float scale = ((float)table_hw) / (end - start);

  // positive axis, slope = x(i+1) - x(i)
  for (int i = 0; i < half - 1; i++) {
    auto x0 = table[i];
    auto x1 = table[i + 1];
    if (F32ToBF16(x0, false) == F32ToBF16(x1, false)) {
      slope_table[i] = 0;
    } else {
      slope_table[i] = BF16(x1 - x0);
    }
  }
  // slope of range end
  slope_table[half - 1] = BF16((func(3 * end, param0, param1) -
                                func(end, param0, param1)) / (2 * end * scale));

  // negtive axis, slope = x(i - 1) - x(i)
  for (int i = table_hw - 1; i > half; i--) {
    auto x0 = table[i];
    auto x1 = table[i - 1];
    if (F32ToBF16(x0, false) == F32ToBF16(x1, false)) {
      slope_table[i] = 0;
    } else {
      slope_table[i] = BF16(x0 - x1);
    }
  }
  // slope of range start
  slope_table[half] = BF16((func(3 * start, param0, param1) -
                            func(start, param0, param1)) / (std::abs(2 * start) * scale));
}

void bf16_gen_base_slope_table(const std::string &name, float *base_table, float *slope_table,
                               float &range_start, float &range_end, float param0, float param1) {
  activate_func func;
  if (name == "sigmoid") {
    range_start = -1 * SIGMOID_BF16_LUT_RANGE;
    range_end = SIGMOID_BF16_LUT_RANGE;
    func = sigmoid_actf;
  } else if (name == "tanh") {
    range_start = -1 * TANH_BF16_LUT_RANGE;
    range_end = TANH_BF16_LUT_RANGE;
    func = tanh_actf;
  } else if (name == "log") {
    range_start = -1 * LOG_BF16_LUT_RANGE;
    range_end = LOG_BF16_LUT_RANGE;
    func = log_actf;
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
  gen_bf16_base_table(range_start, range_end, 256, base_table, func, param0, param1);

  gen_bf16_slope_table(range_start, range_end, 256, base_table, slope_table, func, param0, param1);
}

void bf16_lut_slope(const std::string &name, float *input, float *output, int size,
                    float *base_table, float *slope_table) {
  float range_start;
  float range_end;
  if (name == "sigmoid") {
    range_start = -1 * SIGMOID_BF16_LUT_RANGE;
    range_end = SIGMOID_BF16_LUT_RANGE;
  } else if (name == "tanh") {
    range_start = -1 * TANH_BF16_LUT_RANGE;
    range_end = TANH_BF16_LUT_RANGE;
  } else if (name == "log") {
    range_start = -1 * LOG_BF16_LUT_RANGE;
    range_end = LOG_BF16_LUT_RANGE;
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
    float rescale_bf16_input = BF16(BF16(input[i] - offset, (offset != 0)) * scale);
    // get interger part
    int rescale_input_i8 = F32ToInt8(rescale_bf16_input, 1);
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

static void bf16_gen_pow(int start, int table_hw, float coeff, float *table_data) {
  int half = table_hw / 2;
  uint64_t idx = 0;
  assert(half == 128);
  // 0^-1 is invalid, use positive/negtive max value: 0x7F7F / 0xFF7F
  uint32_t max_bf16_val = 0x7F7F0000;
  float max_bf16 = BF16(*((float *)(&max_bf16_val)), false);

  // prepare channel 0
  if (coeff < 0) {
    table_data[idx] = max_bf16;
  } else if (coeff == 0) {
    table_data[idx] = BF16(1.0);
  } else {
    table_data[idx] = BF16(0.0);
  }
  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half-1; i++) {
    int shift = (start + i);
    float exp = shift;
    float s = (float)pow(2, coeff * exp);
    table_data[idx] = BF16(s);
    idx++;
  }
  if (coeff < 0) {
    table_data[idx] = max_bf16;
  } else if (coeff == 0) {
    table_data[idx] = BF16(1.0);
  } else {
    table_data[idx] = BF16(0.0);
  }
  idx++;
  int _signed = -1;
  if (coeff == (int)(coeff)) {
    if ((int)(coeff) % 2 == 0) {
      _signed = 1;
    }
    // < 0, exp from 0 -62 -61 ..  62  63
    for (int i = 0; i < half - 1; i++) {
      int shift = (start + i);
      float exp = shift;
      float s = _signed * (float)pow(2, coeff * exp);
      table_data[idx] = BF16(s);
      idx++;
    }
  }
}

static void bf16_gen_log(int start, int table_hw, float *table_data) {
  int half = table_hw / 2;
  uint64_t idx = 0;
  assert(half == 128);
  // log(0) is invalid, use negtive max value:  0xFF7F
  uint32_t neg_max_bf16_val = 0xFF7F0000;
  float neg_max_bf16 = BF16(*((float *)(&neg_max_bf16_val)), false);
  table_data[0] = neg_max_bf16;
  idx++;
  // all 64 used indicate inf ignore
  // exp from -62 -61 .. 0 .. 62  63
  for (int i = 0; i < half-1; i++) {
    int shift = (start + i);
    float exp = shift;
    float s = (float)(exp * log(2.));
    table_data[idx] = BF16(s);
    // log(neg_value) is invalid
    table_data[128 + idx] = neg_max_bf16;
    idx++;
  }
}

static void bf16_gen_pow_mantissa(int table_hw, float coeff, float *table_mantissa) {
  // lut low 8 bits and we don't care it's normal number or subnormal number
  uint32_t half = table_hw  / 2;
  assert(half == 128);

  int idx = 0;
  for (uint32_t i = 0; i < half; i++) {
    float d = 1 + i * 1 / 128.0;
    d = (float) pow(d, coeff);
    table_mantissa[128 + idx] = BF16(d);
    table_mantissa[idx] = BF16(d);
    idx++;
  }
}

static void bf16_gen_log_mantissa(int table_hw, float *table_mantissa) {
  uint32_t half = table_hw  / 2;
  assert(half == 128);
  int idx = 0;
  for (uint32_t i = 0; i < half; i++) {
    float d = 1 + i * 1 / 128.0;
    d = (float) log(d);
    table_mantissa[idx] = BF16(d);
    table_mantissa[128 + idx] = BF16(d);
    idx++;
  }
}

void bf16_gen_exponent_mantissa_table(const std::string &name, float *exp_table,
                                      float *mantissa_table, float param0,  float param1) {
  (void)param1;
  float range_start = -62;
  int table_hw = 256;
  if (name == "pow") {
      bf16_gen_pow(range_start, table_hw, param0,exp_table);
      bf16_gen_pow_mantissa(table_hw, param0, mantissa_table);
  } else if (name == "log"){
      bf16_gen_log(range_start, table_hw, exp_table);
      bf16_gen_log_mantissa(table_hw, mantissa_table);
  } else {
    llvm::errs() << "unsupported lookup table func:" << name << "\n";
    llvm_unreachable("Error");
  }
}

void bf16_lut_mantissa(float *input, float *output, int size,
                       float *exp_table, float *mantissa_table,
                       const std::string &method) {
  for (int i = 0; i < size; i++) {
    float val = input[i];
    uint16_t bf16_val = F32ToBF16(val, false);
    int exponentIndex;
    if (val == 0) {
      exponentIndex = 0;
    } else if (val >= 0) {
      exponentIndex = floor(log2(val));
      exponentIndex += 62 + 1; // 62 means start with 2^-62, index from 1
    } else {
      exponentIndex = floor(log2(-1 * val));
      exponentIndex += 62 + 129; // 62 means start with 2^-62, index from 129
    }
    float exponent = exp_table[exponentIndex];
    float mantissa = mantissa_table[bf16_val & 0xff];
    if (method == "mantissa")
      output[i] = BF16(exponent * mantissa);
    else if (method == "log")
      output[i] = BF16(exponent + mantissa);
    else{
      llvm::errs() << "unsupported lookup table func:" << method << "\n";
      llvm_unreachable("Error");
    }
  }
}
