#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

#include <cmath>

namespace mlir {

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

float BF16(float data) { return convert_bf16_fp32(convert_fp32_bf16(data)); }

void relu(float *src, float *dst, size_t size) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
}

void leaky_relu(float *src, float *dst, size_t size, float negative_slope) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : src[i] * negative_slope;
  }
};
} // namespace mlir