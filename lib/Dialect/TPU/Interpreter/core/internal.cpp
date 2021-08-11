#include "internal.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

#include <cmath>

namespace mlir {

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

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

inline int crop_offset(const std::vector<int> &indices, long int *shape) {
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset *= shape[i];
    if ((int)indices.size() > i) {
      offset += indices[i];
    }
  }
  return offset;
}

void crop(float *input, float *output, long int *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < output_shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      crop(input, output, input_shape, output_shape, cur_dim + 1, offsets,
           indices);
    }
  } else {
    std::vector<int> ind_red(cur_dim, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);

    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];

      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];

    std::memcpy(output + crop_offset(ind_red, output_shape),
                input + crop_offset(ind_off, input_shape),
                sizeof(float) * output_shape[cur_dim]);
  }
};

} // namespace mlir