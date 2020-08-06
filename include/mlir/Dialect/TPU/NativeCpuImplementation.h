#ifndef MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_

int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int dh,int dw, int pt, int pb, int pl, int pr, int g);

int mkldnn_deconv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr, int g);

int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
    bool is_avg, bool count_include_pad);

int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose);

int my_gru(float *input, float *output,
    float *weight, float *recurrence, float *bias, float *initial_h,
    int seq_len, int batch_size, int input_size, int hidden_size,
    bool b_bidirectional=false, bool b_linear_before_reset=true);

int my_avg_pooling(float *input, float *output, int n, int c, int ih, int iw,
                   int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                   int pb, int pl, int pr);

int my_sigmoid(float *input, float *output, int n, int c, int h, int w, bool is_bf16 = false);
int my_crop(float *input, float *output, long int *input_shape, long int *output_shape,  int cur_dim, int *offsets, int *indices);

int calc_dilute_hw (int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t);
void my_dilateActivation (float* input, float* output,
    int pad_h_t, int pad_h_b,
    int ins_h,   int ins_h_l,
    int pad_w_l, int pad_w_r,
    int ins_w,   int ins_w_l,
    int n, int c, int h, int w, int fill_constant = 0);

int my_relu(float *input, float *output,
    int n, int c, int h, int w, float negative_slope);

int my_prelu(float *input, float *output, int n, int c, int h, int w,
            float *negative_slope);

int my_bn(float *input, float *mean, float *variance, float *scale, float variance_epsilon,
    float *output, int n, int c, int h, int w);

void my_interp(const int channels,
    const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

int my_interptile(float *input, float *output, int n, int c, int h, int w,
    int _ih, int _iw);

int my_lrn_one(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size, float alpha);
int my_lrn_two(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size);
int my_lrn_three(float *input, float *output, int n, int c, int h, int w,
                 float beta, float k);
int my_lrn_main(float *input, float *scale, float *output, int n, int c, int h,
                int w);
int my_lrn_int8(float *input, float *output, int n, int c, int h, int w,
                unsigned int local_size, float *sqr_lut, float *power_lut,
                int sum_rshift, int lrn_rshit, int quant0, int quant1);

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

int my_permute(float *input, float *output, const int input_shape_size,
    int in, int ic, int ih, int iw,int on, int oc, int oh, int ow,
    int order0,int order1,int order2,int order3);

float my_mish_caffe(float x_val, float mish_threshold = 20.0);
int my_mish(float *input, float *output, int n, int c, int h, int w, bool is_bf16 = false, float mish_threshold = 20.0);

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
                  const std::vector<float>& std,
                  float raw_scale, float input_scale);

int my_transpose(float *input, float *output, int n, int c, int h, int w);

int my_reorg(float *input, float *output, uint32_t stride, int n, int c, int h, int w);

int my_pad_constant(float *input, float *output,
                    std::vector<int64_t> &input_shape,
                    std::vector<int> &pads, float const_val);

void gen_bf16_table(int start, int end, int table_hw, float *table,
                           double (*activate_func)(double));

void gen_bf16_slope_table(int start, int end, int table_hw,
                                         float *table,
                                         float *slope_table, double (*activate_func)(double));
int my_reduce_mean(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes);

int my_reduce_max(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes);

int my_roipooling(float *data, float *rois, float *output, int pooled_h, int pooled_w,
                  float spatial_scale, int batch, int num_rois, int channel, int height, int width);

int my_tile(float *input, float *output, std::vector<int64_t> &input_shape,
            std::vector<int64_t> &output_shape, std::vector<int32_t> &resp);
#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
