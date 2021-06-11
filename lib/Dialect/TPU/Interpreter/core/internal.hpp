#include <stddef.h>
#include <stdint.h>
namespace mlir {

void relu(float *src, float *dst, size_t size);
void leaky_relu(float *src, float *dst, size_t size, float negative_slope);
int omp_schedule(int count);
float BF16(float data);

}