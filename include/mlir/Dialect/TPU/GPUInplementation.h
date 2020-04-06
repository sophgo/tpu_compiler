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
      std::cout << "CUDA CALL Error occurred: " << err << std::endl;           \
      std::exit(1);                                                            \
    }                                                                          \
  }

#define CUDNN_VERSION_MIN(major, minor, patch)                                 \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

inline const char *cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
  case CUDNN_STATUS_SUCCESS:
    return "CUDNN_STATUS_SUCCESS";
  case CUDNN_STATUS_NOT_INITIALIZED:
    return "CUDNN_STATUS_NOT_INITIALIZED";
  case CUDNN_STATUS_ALLOC_FAILED:
    return "CUDNN_STATUS_ALLOC_FAILED";
  case CUDNN_STATUS_BAD_PARAM:
    return "CUDNN_STATUS_BAD_PARAM";
  case CUDNN_STATUS_INTERNAL_ERROR:
    return "CUDNN_STATUS_INTERNAL_ERROR";
  case CUDNN_STATUS_INVALID_VALUE:
    return "CUDNN_STATUS_INVALID_VALUE";
  case CUDNN_STATUS_ARCH_MISMATCH:
    return "CUDNN_STATUS_ARCH_MISMATCH";
  case CUDNN_STATUS_MAPPING_ERROR:
    return "CUDNN_STATUS_MAPPING_ERROR";
  case CUDNN_STATUS_EXECUTION_FAILED:
    return "CUDNN_STATUS_EXECUTION_FAILED";
  case CUDNN_STATUS_NOT_SUPPORTED:
    return "CUDNN_STATUS_NOT_SUPPORTED";
  case CUDNN_STATUS_LICENSE_ERROR:
    return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)
  case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
    return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
  case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
    return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
  case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
    return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
  }
  return "Unknown cudnn status";
}

#define CUDNN_CALL(f)                                                          \
  {                                                                            \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cout << "cudnn Error occurred: " << cudnnGetErrorString(err)        \
                << std::endl;                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  }

#define CUDNN_CHECK(condition)                                                 \
  do {                                                                         \
    cudnnStatus_t status = condition;                                          \
    std::cout << "cudnn check " << cudnnGetErrorString(status) << std::endl;                     \
  } while (0)

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