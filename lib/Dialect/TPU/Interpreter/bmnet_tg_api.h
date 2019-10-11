#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"

void bmnet_conv_parallel_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx,
    u32 stream_id,
    u32 inst_id,
    u32 layer_id,
    const u32 *depends,
    u32 depends_len,
    gaddr_t ga_ifmap,
    gaddr_t ga_ofmap,
    gaddr_t ga_weight,
    gaddr_t ga_bias,
    gaddr_t ga_bn_mean,
    gaddr_t ga_bn_variance,
    gaddr_t ga_scale,
    gaddr_t ga_scale_bias,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int groups,
    int output_c,
    u16 kh,
    u16 kw,
    u16 dilation_h,
    u16 dilation_w,
    u8 pad_top,
    u8 pad_bottom,
    u8 pad_left,
    u8 pad_right,
    u8 stride_h,
    u8 stride_w,
    int result_add,
    int do_bias,
    int do_bn,
    int do_scale,
    int do_scale_bias,
    int do_activation,
    float bn_scale,
    float bn_eps,
    int activation_method,
    float activation_arg[],
    gaddr_t activation_ga_slope,
    bool activation_channel_shared,
    int activation_gt_scale,
    int activation_gt_rshift,
    int activation_le_scale,  // slope, TODO
    int activation_le_rshift,
    int right_shift_width,
    int bn_right_shift_width,
    int scale_right_shift_width,
    bool use_winograd);

void bmnet_fc_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx,
    u32 stream_id,
    u32 inst_id,
    u32 layer_id,
    const u32 *depends,
    u32 depends_len,
    gaddr_t bottom_data_gaddr,
    gaddr_t weight_data_gaddr,
    gaddr_t bias_data_gaddr,
    gaddr_t top_data_gaddr,
    int in_row,
    int in_col,
    int out_col,
    int have_bias,
    int do_activation,
    int activation_method,
    gaddr_t activation_ga_slope,
    int activation_channel_shared,
    int activation_gt_scale,
    int activation_gt_rshift,
    int activation_le_scale,
    int activation_le_rshift,
    bool weight_tp,
    int left_shift_width,
    int right_shift_width);

void bmnet_pooling_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx,
    u32 stream_id,
    u32 inst_id,
    u32 layer_id,
    const u32 *depends,
    u32 depends_len,
    gaddr_t ifmap_gaddr,
    gaddr_t ofmap_gaddr,
    gaddr_t index_gaddr,
    gaddr_t o_findex_gaddr,
    int n,
    int c,
    int h,
    int w,
    int kh,
    int kw,
    int pad_top,
    int pad_bot,
    int pad_left,
    int pad_right,
    int stride_h,
    int stride_w,
    int is_avg_pooling,
    float avg_const,  // default(passing 0.0f) is 1/kh*kw
    int do_relu,
    int right_shift_width,
    const int *threshold_x_quantized,
    const bool ceil_mode);

void bmnet_relu_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx,
    u32 stream_id,
    u32 inst_id,
    u32 layer_id,
    const u32 *depends,
    u32 depends_len,
    u64 bottom_gaddr,
    u64 top_gaddr,
    float negative_slope,
    int input_n,
    int input_c,
    int input_h,
    int input_w);

void bmnet_eltwise_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx,
    u32 stream_id,
    u32 inst_id,
    u32 layer_id,
    const u32 *depends,
    u32 depends_len,
    gaddr_t ga_input[],
    gaddr_t ga_output,
    int input_size,
    int op,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    bool do_relu,
    float relu_slope,
    int right_shift_width,
    const int *threshold_x_quantized,
    const int *coeffs);
