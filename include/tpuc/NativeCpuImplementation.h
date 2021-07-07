#ifndef MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_

//
// mkldnn functions
//
int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int dh,int dw, int pt, int pb, int pl, int pr, int g, int pad_value);

int mkldnn_deconv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr, int g);

int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
    bool is_avg, bool count_include_pad, int pad_value=0);

int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose);

void mkldnn_conv3d(float *input, float *weight, float *bias, float *output,
  int batch, int ic, int id, int ih, int iw, int oc, int od, int oh, int ow,
  int g, int kd, int kh, int kw, int sd, int sh, int sw, int dd, int dh, int dw,
  int pd0, int pt, int pb, int pd1, int pl, int pr);

//
// native cpu functions
//
int my_abs(float *input, float *output,
    int n, int c, int h, int w);

int my_avg_pooling(float *input, float *output, int n, int c, int ih, int iw,
                   int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                   int pb, int pl, int pr);
int my_lut_interpolation(float *input, float *output, int n, int c, int h, int w,
                         bool is_bf16, double (*activate_func)(double),
                         float thresh_min, float thresh_max, bool isExpFunc);
int my_exp(float *input, float *output, int n, int c, int h, int w, bool is_bf16 = false);
int my_reciprocal(float *input, float *output, int n, int c, int h, int w, bool is_bf16 = false);
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

int my_interp_linear(float *input, float* output, int n, int c, int ih, int iw, int oh, int ow);

int my_scale(float *input, float *scale, float *bias,
    float *output, int n, int c, int h, int w);

int my_pixelshuffle(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    int upscale_factor, bool dcr_mode=false);

int my_clip(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    float min, float max);

int my_div(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    float divisor);

int my_upsample(float *input, float *output, int n, int c, int ih, int iw,
                int scale_h, int scale_w);

int my_permute(float *input, float *output, int in, int ic, int ih, int iw,
               int order0, int order1, int order2, int order3);

float my_mish_caffe(float x_val, float mish_threshold = 20.0);

float my_mish_caffe_tanh_part(float x_val, float mish_threshold = 20.0);

int my_normalize(float *input,float *scale, float *output,
    bool across_spatial,bool channel_shared,
    int n, int c, int h, int w);

int my_slice(float *input, float *output, int axis, int offset,
  std::vector<int64_t> input_shape, std::vector<int64_t> output_shape);

int my_power(float *input, float *output,
    int n, int c, int h, int w, float scale, float shift, float power);

int my_transpose(float *input, float *output, int n, int c, int h, int w);

int my_reorg(float *input, float *output, uint32_t stride, int n, int c, int h, int w);

int my_pad_constant(float *input, float *output,
                    std::vector<int64_t> &input_shape,
                    std::vector<int> &pads, float const_val);

int my_reduce_l2(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes);

int my_reduce_mean(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes);

int my_reduce_mean_int8(float *input, float *output,
                        std::vector<int64_t> &org_input_shape,
                        std::vector<int> &axes,
                        int avg_const, int rshift);

int my_reduce_max(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes);

int my_roipooling(float *data, float *rois, float *output, int pooled_h, int pooled_w,
                  float spatial_scale, int batch, int num_rois, int channel, int height, int width);

// type = 0:fp32,=1:u8,=2:bf16
void my_yuv420_csc(float *input, float *output, int n, int c, int h, int w,
                   std::vector<int> order, int type = 0);

void conv3d_float_ref(float *input, float *weight, float *bias, float *output,
                      int n, int input_c, int input_d, int input_h, int input_w,
                      int output_c, int output_d, int output_h, int output_w,
                      int kernel_d, int kernel_h, int kernel_w,
                      int stride_d, int stride_h, int stride_w,
                      int dilation_d, int dilation_h, int dilation_w,
                      int pad_d0, int pad_top, int pad_bottom,
                      int pad_d1, int pad_left, int pad_right);

void pool3d_float_ref(float *input, float *output,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d0, int pad_d1,
    int pad_top, int pad_bot, int pad_left, int pad_right);

float softplus_activate (float x, float threshold = 20);

//
// bf16 table functions
//
void gen_bf16_table(int start, int end, int table_hw, float *table,
                           double (*activate_func)(double));

void gen_bf16_slope_table(int start, int end, int table_hw,
                                         float *table,
                                         float *slope_table, double (*activate_func)(double));

void bf16_gen_reciprocal(int start, int end, int table_hw, uint16_t *table_data);
void bf16_gen_reciprocal_mantissa(int start, int end, int table_hw, uint16_t *table_mantissa);

void bf16_gen_sqrt(int start, int table_hw, uint16_t *table_data);
void bf16_gen_sqrt_mantissa(int table_hw, uint16_t *table_mantissa);

// y = 1/sqrt(x)
void bf16_gen_reciprocal_sqrt(int start, int table_hw, uint16_t *table_data);
void bf16_gen_reciprocal_sqrt_mantissa(int table_hw, uint16_t *table_mantissa);

void bf16_gen_power_exp_table(uint16_t *table_data, float beta,
                              int start, int table_hw);
void bf16_gen_power_mantissa_table(uint16_t* table_mantissa, float beta,
                                   int table_hw);

void bf16_lut_mantissa(float *input, float *output, int size,
                       const std::vector<float> &bf16_lut,
                       const std::vector<float> &bf16_mantissa_lut);
void bf16_lut_slope(float *input, float *output, int size,
                    const std::vector<float> &bf16_lut,
                    const std::vector<float> &bf16_slope_lut,
                    int bf16_table_start, int bf16_table_end);
#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
