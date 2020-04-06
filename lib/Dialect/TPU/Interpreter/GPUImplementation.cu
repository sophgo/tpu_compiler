#include "cuda_runtime.h"
#include "mlir/Dialect/TPU/GPUInplementation.h"

__global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(i, n){
    out[i] = in[i] * ((in[i] > 0) + (in[i] <= 0) * negative_slope);
  }
}


int gpu_conv(float *input, float *weight, float *bias,
    float *output, int in, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int dh, int dw, int ph, int pw, int g){
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
    // input
    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));

    float *in_data;
    CUDA_CALL(cudaMallocManaged(&in_data, in * ic * ih * iw * sizeof(float)));
    CUDA_CALL(cudaMemcpy(in_data, input, in * ic * ih * iw * sizeof(float),
               cudaMemcpyDefault));
    // filter
    cudnnFilterDescriptor_t filt_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, oc, ic, kh, kw));


    float *filt_data;
    CUDA_CALL(cudaMallocManaged(&filt_data, oc * ic * kh * kw * sizeof(float)));
    CUDA_CALL(cudaMemcpy(filt_data, weight, oc * ic * kh * kw * sizeof(float),
               cudaMemcpyDefault));

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, ph, pw, sh, sw, dh,
                                                dw, CUDNN_CONVOLUTION,
                                                CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, in, oc, oh, ow));
    float *out_data;
    CUDA_CALL(cudaMallocManaged(&out_data, in * oc * oh * ow * sizeof(float)));

    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        cudnn, in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
    float *ws_data;
    CUDA_CALL(cudaMallocManaged(&ws_data, ws_size));
    float alpha = 1, beta = 0;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, algo,
        ws_data, ws_size, &beta, out_desc, out_data));

    if (bias) {
      beta = 1.0f;
      cudnnTensorDescriptor_t bias_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, 1, oc, 1, 1));

      float *biasdata;
      CUDA_CALL(cudaMallocManaged(&biasdata, oc * sizeof(float)));
      CUDA_CALL(cudaMemcpy(biasdata, bias, oc * sizeof(float),
                           cudaMemcpyDefault));
    
      // add bias
      CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, bias_desc, biasdata, &beta,
                                out_desc, out_data));

      CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc));
      CUDA_CALL(cudaFree(biasdata));
    }

    CUDA_CALL(cudaMemcpy(output, out_data, in * oc * oh * ow * sizeof(float),
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();
    // release
    CUDA_CALL(cudaFree(ws_data));
    CUDA_CALL(cudaFree(out_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDA_CALL(cudaFree(filt_data));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDA_CALL(cudaFree(in_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));
    return 0;
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