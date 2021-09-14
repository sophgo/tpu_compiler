#include "tpuc/Interpreter/cpu/std.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

StdOpKernel::StdOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto stdOp = cast<tpu::StdOp>(op);
  // get tensors
  input_data = this->opdTensors[0];
  lut = this->opdTensors[1];
  mantissa_lut = this->opdTensors[2];
  start_dim = stdOp.start_dim();
  unbiased = stdOp.unbiased();
  output_data = this->resTensor;
  input_shape = getTensorShape(stdOp.input());
  outer_size =
      std::accumulate(input_shape.begin(), input_shape.begin() + start_dim, 1,
                      std::multiplies<int64_t>());
  std_size = std::accumulate(input_shape.begin() + start_dim, input_shape.end(),
                             1, std::multiplies<int64_t>());
  if (datatype == DataType::BF16) {
    assert(lut);
    assert(mantissa_lut);
  }
}

void StdOpKernel::std_fp32(float *src, float *dst, int size) {
  double sum = std::accumulate(src, src + size, 0.0f);
  double mean = sum / size;
  double var = 0;
  for (int i = 0; i < size; i++) {
    var += std::pow(src[i] - mean, 2);
  }
  if (unbiased) {
    var = var / (size - 1);
  } else {
    var = var / size;
  }
  *dst = std::sqrt(var);
}

void StdOpKernel::std_bf16(float *src, float *dst, int size) {
  float mean = 0.0f;
  float var = 0;
  float data;
  float avg_const = BF16(1.0 / size);
  for (int i = 0; i < size; i++) {
    mean += src[i] * avg_const;
  }
  mean = BF16(mean);
  for (int i = 0; i < size; i++) {
    data = BF16(src[i] - mean);
    var += BF16(std::pow(data, 2)) * avg_const;
  }
  if (unbiased) {
    var = BF16(BF16(var) * BF16((float)size / ((float)size - 1.0f)));
  } else {
    var = BF16(var);
  }
  bf16_lut_mantissa(&var, dst, 1, lut->data(), mantissa_lut->data());
}

void StdOpKernel::invoke() {
  switch (datatype) {
  case DataType::FP32:
#pragma omp parallel for schedule(static, omp_schedule(outer_size))
    for (int i = 0; i < outer_size; i++) {
      std_fp32(input_data->data() + i * std_size, output_data->data() + i,
               std_size);
    }
    return;
  case DataType::BF16:
#pragma omp parallel for schedule(static, omp_schedule(outer_size))
    for (int i = 0; i < outer_size; i++) {
      std_bf16(input_data->data() + i * std_size, output_data->data() + i,
               std_size);
    }
    return;
  default:
    break;
  }
  llvm_unreachable("unsupport datatype");
}

} // namespace mlir