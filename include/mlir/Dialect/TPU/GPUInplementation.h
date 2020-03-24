#ifndef MLIR_DIALECT_GPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_GPU_IMPLEMENTATION_H_

#ifdef USE_GPU
#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cout << "    Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

#define CUDNN_CALL(f)                                                          \
  {                                                                            \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cout << "    Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
int gpu_conv(float *input, float *weight, float *bias, float *output, int in,
                int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
                int sh, int sw, int dh, int dw, int ph, int pw, int g);

int gpu_relu(float *input, float *output, int n, int c, int h, int w,
             float negative_slope);

#endif
#endif // MLIR_DIALECT_GPU_IMPLEMENTATION_H_