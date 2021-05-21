#include "tpuc/Interpreter/cpu/layernorm.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

LayerNormOpKernel::LayerNormOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto lnOp = cast<tpu::LayerNormOp>(op);
  // get tensors
  input_data = this->opdTensors[0];
  scale_data = this->opdTensors[1];
  bias_data = this->opdTensors[2];
  lut = this->opdTensors[3];
  mantissa_lut = this->opdTensors[4];
  eps = lnOp.eps().convertToFloat();
  arrayAttrToVector(lnOp.normalized_shape(), this->normalized_shape);
  output_data = this->resTensor;
  normalized_size = 1;
  batch_size = 1;
  size_t nm_dims = normalized_shape.size();
  size_t input_dims = shape.size();
  assert(nm_dims <= input_dims);
  size_t index = 0;
  for (; index < input_dims - nm_dims; index++) {
    batch_size *= shape[index];
  }
  for (size_t i = 0; index < input_dims; index++, i++) {
    assert(normalized_shape[i] == shape[index]);
    normalized_size *= normalized_shape[i];
  }
  if (scale_data != nullptr && bias_data != nullptr) {
    affine = true;
  }
  if (datatype == DataType::BF16) {
    assert(lut);
    assert(mantissa_lut);
  }
}

void LayerNormOpKernel::normalize_fp32(float *src, float *dst, int size) {
  double sum = std::accumulate(src, src + size, 0.0f);
  double mean = sum / size;
  double var = 0;
  for (int i = 0; i < size; i++) {
    var += std::pow(src[i] - mean, 2);
  }
  var = var / size + eps;
  double div = std::sqrt(var);
  for (int i = 0; i < size; i++) {
    dst[i] = ((src[i] - mean) / div);
  }
  if (affine) {
    for (int i = 0; i < size; i++) {
      dst[i] = dst[i] * scale_data->at(i) + bias_data->at(i);
    }
  }
}

void LayerNormOpKernel::normalize_bf16(float *src, float *dst, int size) {
  float sum = BF16(std::accumulate(src, src + size, 0.0f));
  float mean = BF16(sum / size);
  float var = 0;
  float data;
  for (int i = 0; i < size; i++) {
    data = BF16(src[i] - mean);
    var += std::pow(data, 2);
  }
  var = BF16(var / size) + BF16(eps);
  bf16_lut_mantissa(&var, &var, 1, *lut, *mantissa_lut);
  for (int i = 0; i < size; i++) {
    data = BF16(src[i] - mean);
    dst[i] = BF16(data * var);
  }
  if (affine) {
    for (int i = 0; i < size; i++) {
      dst[i] = BF16(BF16(dst[i] * scale_data->at(i)) + bias_data->at(i));
    }
  }
}

void LayerNormOpKernel::invoke() {
  switch (datatype) {
  case DataType::FP32:
#pragma omp parallel for schedule(static, omp_schedule(batch_size))
    for (int i = 0; i < batch_size; i++) {
      normalize_fp32(input_data->data() + i * normalized_size,
                     output_data->data() + i * normalized_size,
                     normalized_size);
    }
    return;
  case DataType::BF16:
#pragma omp parallel for schedule(static, omp_schedule(batch_size))
    for (int i = 0; i < batch_size; i++) {
      normalize_bf16(input_data->data() + i * normalized_size,
                     output_data->data() + i * normalized_size,
                     normalized_size);
    }
    return;
  default:
    break;
  }
  llvm_unreachable("unsupport datatype");
}

} // namespace mlir