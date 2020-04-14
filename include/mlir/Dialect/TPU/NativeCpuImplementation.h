#ifndef MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_

int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int dh,int dw,int ph, int pw, int g);

int mkldnn_deconv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw, int g);

int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
    bool is_avg);

int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose);
int my_avg_pooling(float *input, float *output, int n, int c, int ih, int iw,
                   int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                   int pb, int pl, int pr);
int my_sigmoid(float *input, float *output, int n, int c, int h, int w);
int my_crop(float *input, float *output, long int *shape1, int *shape2, long int *top_shape,  int cur_dim, int *offsets, int *indices);
int my_relu(float *input, float *output,
    int n, int c, int h, int w, float negative_slope);

int my_prelu(float *input, float *output, int n, int c, int h, int w,
            float *negative_slope);

int my_bn(float *input, float *mean, float *variance, float *scale, float variance_epsilon,
    float *output, int n, int c, int h, int w);

int my_lrn_one(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size, float alpha, float beta, float k);
int my_lrn_two(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size, float alpha, float beta, float k);
int my_lrn_three(float *input, float *output, int n, int c, int h, int w,
                 unsigned int local_size, float alpha, float beta, float k);
int my_lrn_main(float *input, float *scale, float *output, int n, int c, int h, int w,
                unsigned int local_size, float alpha, float beta, float k);
int my_lrn_int8(float *input, float *output, int n, int c, int h, int w,
                unsigned int local_size, float *sqr_table, float *power_table,
                int quant0, int quant1);

int my_shuffle_channel(float *input, float *output, unsigned int group,
    int n, int c,  int frame_size);

int my_scale(float *input, float *scale, float *bias,
    float *output, int n, int c, int h, int w);

int my_swap_channel(float *input, float *output, int n, int c,  int h, int w, int * order);

int my_pixelshuffle(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    int upscale_factor);

int my_clip(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    float min, float max);

int my_div(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    float divisor);

int my_upsample(float *input, float *output, int n, int c, int ih, int iw,
                int scale);

int my_softmax2D(float *input, float *output, int n, int c);
int my_softmax4D(float *input, float *output, int axis, const std::vector<int64_t>& shape);
int my_softmax3D(float *input, float *output, int axis, const std::vector<int64_t>& shape);

int my_tanh(float *input, float *output,
    int n, int c, int h, int w);
int my_eltwise(float *input_1, float *input_2, float *output,
    int n, int c, int h, int w, int op);

int my_permute(float *input, float *output, const int input_shape_size,
    int in, int ic, int ih, int iw,int on, int oc, int oh, int ow,
    int order0,int order1,int order2,int order3);

int my_normalize(float *input,float *scale, float *output,
    bool across_spatial,bool channel_shared,
    int n, int c, int h, int w);

int my_slice(float *input, float *output, int axis, int offset,
  std::vector<int64_t> input_shape, std::vector<int64_t> output_shape);

int my_power(float *input, float *output,
    int n, int c, int h, int w, float scale, float shift, float power);

int my_preprocess(float *input, float *output,
                  int n, int c, int h, int w,
                  const std::vector<int>& channel_order,
                  const std::vector<float>& mean,
                  float raw_scale, float input_scale);

int my_transpose(float *input, float *output, int n, int c, int h, int w);

#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
