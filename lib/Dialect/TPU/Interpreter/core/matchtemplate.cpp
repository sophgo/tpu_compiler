#include "tpuc/Interpreter/cpu/matchtemplate.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"


namespace mlir {

MatchTemplateOpKernel::MatchTemplateOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto castOp = cast<tpu::MatchTemplateOp>(op);
  // get tensors
  input_data = this->opdTensors[0];
  template_data = this->opdTensors[1];
  mode = castOp.mode().str();
  output_data = this->resTensor;
  // get table
  lut = this->opdTensors[2];
  mantissa_lut = this->opdTensors[3];

  input_shape = getTensorShape(castOp.input());
  template_shape = getTensorShape(castOp.match());
  assert(input_shape.size() == 2);
  assert(template_shape.size() == 2);
  assert(input_shape[0] >= template_shape[0]);
  assert(input_shape[1] >= template_shape[1]);

  n = input_shape[0] - template_shape[0] + 1;
  c = input_shape[1] - template_shape[1] + 1;
  h = template_shape[0];
  w = template_shape[1];
  stride = input_shape[1];
  outer_size = n * c;
  match_size = h * w;

  if (datatype == DataType::BF16) {
    assert(lut);
    assert(mantissa_lut);
  }
}

void MatchTemplateOpKernel::ccoeff_normed_fp32(float *input, float *tmplate, float *dst, int size) {
  double dividend = 0;
  double divider = 0;
  for (int i = 0; i < size; i++){
    uint32_t offset = i / w * stride + i % w;
    auto inp = input[offset];
    dividend += inp * tmplate[i];
    divider += std::pow(inp, 2);
  }
  *dst = dividend * std::pow(divider, -0.5);
}

void MatchTemplateOpKernel::ccoeff_normed_bf16(float *input, float *tmplate, float *dst, int size) {
  float dividend = 0;
  float divider = 0;
  for (int i = 0; i < size; i++){
    uint32_t offset = i / w * stride + i % w;
    auto inp = input[offset];
    dividend += BF16(BF16(inp) * BF16(tmplate[i]));
    divider +=  BF16(BF16(inp) * BF16(inp));
  }
  dividend = BF16(dividend);
  divider = BF16(divider);
  bf16_lut_mantissa(&divider, &divider, 1, lut->data(), mantissa_lut->data());
  *dst =  BF16(dividend * divider);
}

void MatchTemplateOpKernel::sqiff_fp32(float *input, float *tmplate, float *dst, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++){
    uint32_t offset = i / w * stride + i % w;
    auto inp = input[offset];
    sum += std::pow(inp - tmplate[i], 2);
  }
  *dst = sum;
}

void MatchTemplateOpKernel::sqiff_bf16(float *input, float *tmplate, float *dst, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++){
    uint32_t offset = i / w * stride + i % w;
    auto inp = input[offset];
    sum += BF16(BF16(inp - tmplate[i]) * BF16(inp - tmplate[i]));
  }
  *dst =  BF16(sum);
}

void MatchTemplateOpKernel::invoke() {
  assert(mode == "TM_CCOEFF_NORMED" || mode == "TM_SQIFF");
  switch (datatype) {
  case DataType::FP32:
#pragma omp parallel for schedule(static, omp_schedule(outer_size))
    for (int i = 0; i < outer_size; i++) {
      uint32_t ioffset = i / c * stride + i % c;
      if(mode == "TM_CCOEFF_NORMED")
        ccoeff_normed_fp32(input_data->data() + ioffset,
                           template_data->data(),
                           output_data->data() + i,
                           match_size);
      else
        sqiff_fp32(input_data->data() + ioffset,
                           template_data->data(),
                           output_data->data() + i,
                           match_size);
    }
    return;
  case DataType::BF16:
#pragma omp parallel for schedule(static, omp_schedule(outer_size))
    for (int i = 0; i < outer_size; i++) {
      uint32_t ioffset = i / c * stride + i % c;
      if(mode == "TM_CCOEFF_NORMED")
        ccoeff_normed_bf16(input_data->data() + ioffset,
                           template_data->data(),
                           output_data->data() + i,
                           match_size);
      else
        sqiff_bf16(input_data->data() + ioffset,
                           template_data->data(),
                           output_data->data() + i,
                           match_size);

    }
    // BF16(output_data->data(), output_data->data(), output_data->size());
    return;
  default:
    break;
  }
  llvm_unreachable("unsupport datatype");
}
} // namespace mlir
