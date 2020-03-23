#include "cuda_runtime.h"
#include "mlir/Dialect/TPU/GPUInplementation.h"

__global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(i, n){
    out[i] = in[i] * ((in[i] > 0) + (in[i] <= 0) * negative_slope);
  }
}

int gpu_relu(float *input, float *output, int n, int c, int h, int w,
             float negative_slope) {
  int size = n * c * h * w;
  float *gi, *go;
  cudaMallocManaged(&gi, size * sizeof(float));
  cudaMallocManaged(&go, size * sizeof(float));

  cudaMemcpy(gi, input, size * sizeof(float), cudaMemcpyDefault);
  ReLUForward<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(size, gi, go,
                                                      negative_slope);
  cudaDeviceSynchronize();
  cudaMemcpy(output, go, size * sizeof(float), cudaMemcpyDefault);
  cudaFree(gi);
  cudaFree(go);
  return 0;
}