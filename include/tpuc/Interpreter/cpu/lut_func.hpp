#ifndef MLIR_LOOKUP_TABLE_FUNC_H_
#define MLIR_LOOKUP_TABLE_FUNC_H_

#include <vector>
#include <string>
#include <stdint.h>

#define SIGMOID_BF16_LUT_RANGE 12
#define TANH_BF16_LUT_RANGE 15
#define EXP_BF16_LUT_RANGE 15
#define MISH_BF16_LUT_RANGE 8
#define SOFTPLUS_BF16_LUT_RANGE 8
#define SWISH_BF16_LUT_RANGE 12


void bf16_gen_base_slope_table(const std::string &name, float *base_table, float *slope_table,
                               float &range_start, float &range_end);
void bf16_lut_slope(const std::string &name, float *input, float *output, int size,
                    const std::vector<float> &base_table,
                    const std::vector<float> &slop_table);

void bf16_gen_exponent_mantissa_table(const std::string &name, uint16_t *exp_table,
                                      uint16_t *mantissa_table, float extra_param);


void bf16_gen_reciprocal(int start, int end, int table_hw, uint16_t *table_data);
void bf16_gen_reciprocal_mantissa(int start, int end, int table_hw, uint16_t *table_mantissa);

void bf16_gen_sqrt(int start, int table_hw, uint16_t *table_data);
void bf16_gen_sqrt_mantissa(int table_hw, uint16_t *table_mantissa);

// y = 1/sqrt(x)
void bf16_gen_reciprocal_sqrt(int start, int table_hw, uint16_t *table_data);
void bf16_gen_reciprocal_sqrt_mantissa(int table_hw, uint16_t *table_mantissa);

void bf16_gen_power_exp_table(uint16_t *table_data, float beta,
                              int start, int table_hw);
void bf16_gen_power_mantissa_table(uint16_t* table_mantissa, float beta,
                                   int table_hw);

void bf16_lut_mantissa(float *input, float *output, int size,
                       const std::vector<float> &bf16_lut,
                       const std::vector<float> &bf16_mantissa_lut);
#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
