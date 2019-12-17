#ifndef MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_

int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw, int g);

int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw,
    bool is_avg);

int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose);

int my_relu(float *input, float *output,
    int n, int c, int h, int w, float negative_slope);

int my_prelu(float *input, float *output, int n, int c, int h, int w,
            float *negative_slope);

int my_bn(float *input, float *mean, float *variance, float *scale,
    float *output, int n, int c, int h, int w);

int my_scale(float *input, float *scale, float *bias,
    float *output, int n, int c, int h, int w);

int my_softmax(float *input, float *output, int n, int c);

int my_eltwise(float *input_1, float *input_2, float *output,
    int n, int c, int h, int w, int op);

#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
