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

int calc_dilute_hw (int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t);
void my_dilateActivation (float* input, float* output,
    int pad_h_t, int pad_h_b,
    int ins_h,   int ins_h_l,
    int pad_w_l, int pad_w_r,
    int ins_w,   int ins_w_l,
    int n, int c, int h, int w, int fill_constant = 0);

void my_interp(const int channels,
    const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

int my_scale(float *input, float *scale, float *bias,
    float *output, int n, int c, int h, int w);

int my_pixelshuffle(float *input, float *output, int in, int ic,
                    int ih, int iw, int on, int oc, int oh, int ow,
                    int upscale_factor, bool dcr_mode=false);

int my_upsample(float *input, float *output, int n, int c, int ih, int iw,
                int scale_h, int scale_w);

int my_permute(float *input, float *output, int in, int ic, int ih, int iw,
               int order0, int order1, int order2, int order3);

float my_mish_activate(float x_val);

int my_pad_constant(float *input, float *output,
                    std::vector<int64_t> &input_shape,
                    std::vector<int> &pads, float const_val);

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

#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
