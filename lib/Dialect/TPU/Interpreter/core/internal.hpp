#include <stddef.h>
#include <stdint.h>
#include "tpuc/QuantizationArithmetic.h"

namespace mlir {

void relu(float *src, float *dst, size_t size);
void leaky_relu(float *src, float *dst, size_t size, float negative_slope);
void crop(float *input, float *output, long int *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices);
int omp_schedule(int count);

}