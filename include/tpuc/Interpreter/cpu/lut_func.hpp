#ifndef MLIR_LOOKUP_TABLE_FUNC_H_
#define MLIR_LOOKUP_TABLE_FUNC_H_

#include <vector>
#include <string>
#include <stdint.h>

#define SIGMOID_BF16_LUT_RANGE 12
#define TANH_BF16_LUT_RANGE 15
#define EXP_BF16_LUT_RANGE 15
#define ELU_BF16_LUT_RANGE 15
#define MISH_BF16_LUT_RANGE 8
#define SOFTPLUS_BF16_LUT_RANGE 8
#define SWISH_BF16_LUT_RANGE 12


void bf16_gen_base_slope_table(const std::string &name, float *base_table, float *slope_table,
                               float &range_start, float &range_end, float param0, float param1);
void bf16_lut_slope(const std::string &name, float *input, float *output, int size,
                    const std::vector<float> &base_table,
                    const std::vector<float> &slop_table);

void bf16_gen_exponent_mantissa_table(const std::string &name, float *exp_table,
                                      float *mantissa_table, float param0, float param1);

void bf16_lut_mantissa(float *input, float *output, int size,
                       const std::vector<float> &bf16_lut,
                       const std::vector<float> &bf16_mantissa_lut);
#endif
