/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_tg_api.h
 * Description: backend tensor global interface.
 */

#ifndef CVI_BACKEND_TG_API
#define CVI_BACKEND_TG_API

#include <backend/backend_common.h>

void *cvi_backend_get_cvk_ctx(const CviBackendContext &ctx);

/////////////// fixed kernel API /////////////
/*
 * - do_ic_alignment
 *   when do_ic_alignment is set
 *   weight shape has been changed to even (ic is the most inner dim)
 *   input shape (input_c) is the original ic
 */
void cvi_backend_tg_fixed_conv_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean,
    gaddr_t ga_bn_variance, gaddr_t ga_scale, gaddr_t ga_scale_bias,
    int input_n, int input_c, int input_h, int input_w, int groups,
    int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom,
    uint8_t pad_left, uint8_t pad_right, uint8_t insert_h, uint8_t insert_w,
    uint8_t stride_h, uint8_t stride_w, int do_bias,
    int do_bn, int do_scale, int do_scale_bias, int do_activation,
    float bn_scale, float bn_eps, int activation_method, float activation_arg[],
    gaddr_t activation_ga_slope, bool activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, int right_shift_width, int bn_right_shift_width,
    int scale_right_shift_width, bool do_chl_quan, bool do_ic_alignment,
    bool store_compr_act, bool load_compr_act);

void cvi_backend_tg_fixed_concat_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddrs[], gaddr_t output_gaddr, int input_dims[],
    int input_num, int concat_axis, int output_dim_size, int *output_dim,
    bool do_relu, const int need_quantize_num, const int *right_shift_width,
    const int *threshold_x_quantized);

void cvi_backend_tg_fixed_fc_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_data_gaddr, gaddr_t weight_data_gaddr, gaddr_t bias_data_gaddr,
    gaddr_t top_data_gaddr, int in_row, int in_col, int out_col, int have_bias, int do_activation,
    int activation_method, gaddr_t activation_ga_slope, int activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, bool weight_tp, int left_shift_width, int right_shift_width,
    int threshold_x_quantized_len, const int *threshold_x_quantized, const int *right_shift_array,
    bool compressed_weight, std::vector<int> compr_weight_poss);

void cvi_backend_tg_fixed_max_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w,
    bool do_relu, bool ceil_mode);

void cvi_backend_tg_fixed_avg_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w,
    bool do_relu, int rshift, int multiplier, bool ceil_mode);

void cvi_backend_tg_fixed_relu_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, uint64_t bottom_gaddr, uint64_t top_gaddr,
    float negative_slope, int input_n, int input_c, int input_h,
    int input_w, int threshold_x_quantized_len,
    const int *threshold_x_quantized,
    const int *right_shift_array, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_leakyrelu_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, uint64_t input_gaddr, uint64_t output_gaddr,
    int input_n, int input_c, int input_h,
    int input_w, int GT_right_shift_width, int LE_right_shift_width, int GT_scale, int LE_scale,
    int threshold_x_quantized_len, const int *threshold_x_quantized, const int *right_shift_array);

void cvi_backend_tg_fixed_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    uint64_t bottom_gaddr, uint64_t top_gaddr, uint64_t negative_scope_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    int threshold_x_quantized_len,
    const int *threshold_x_quantized,
    const int *right_shift_array, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    uint64_t bottom_gaddr, uint64_t top_gaddr, uint64_t negative_scope_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    int GT_right_shift_width, int GT_scale,
    int LE_right_shift_width, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_power_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr, gaddr_t output_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    const int power, const gaddr_t scale_gaddr,
    const gaddr_t shift_gaddr, int right_shift_width,
    gaddr_t mulpy_offset, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_scale_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w, int scale_dim,
    int inner_dim, bool is_scale_const, int const_scale, int right_shift_width, int do_activation,
    int activation_method, float activation_arg[], const int *i8_multiplier, bool do_bias,
    bool second_is_blob);

// wrapper for quantize for int 8, INT8_PER_LAYER
void cvi_backend_tg_fixed_scale_qi8_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w, int scale_dim,
    int inner_dim, bool is_scale_const, int const_scale, int right_shift_width, int do_activation,
    int activation_method, float activation_arg[],
    const int *i8_multiplier,  // INT8_PER_LAYER
    bool do_bias,
    bool second_is_blob  // true means second comes from weight, otherwise comes from another input
);

// wrapper for quantize for int 32, INT8_32_MULTIPLER
void cvi_backend_tg_fixed_scale_qi32_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w, int scale_dim,
    int inner_dim, bool is_scale_const, int const_scale, int do_activation, int activation_method,
    float activation_arg[], bool do_bias,
    bool second_is_blob  // true means second comes from weight, otherwise comes from another input
);

void cvi_backend_tg_fixed_lut_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr, gaddr_t sg_lut_gaddr,
    int input_n, int input_c, int input_h, int input_w, cvk_fmt_t fmt);

// y = x * multiplier + const_val
void cvi_backend_tg_fixed_mac_const_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_n, int input_c,
    int input_h, int input_w, int multiplier, int const_val, bool do_relu);

void cvi_backend_tg_fixed_crop_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr, int *input1_dim,
    int *input2_dim, int *output_dim, int *offsets, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_dilate_kernel(
    const CviBackendContext &ctx,
    uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int n, int c, int ih, int iw,
    int oh, int ow, int fill_constant,
    int ins_h, int ins_w,
    cvk_fmt_t fmt);

void cvi_backend_tg_fixed_deconv_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int output_h,
    int output_w, int kh, int kw, int dh, int dw, int pad_h_top, int pad_h_bottom, int pad_w_left,
    int pad_w_right, int stride_h, int stride_w, bool do_bias, bool result_add, bool do_relu,
    int right_shift_width, bool use_winograd, int right_shift_array_len, gaddr_t ga_per_channel);

void cvi_backend_tg_fixed_premute_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_n,
    int input_c, int input_h, int input_w, int output_n, int output_c,
    int output_h, int output_w, int order_n, int order_c, int order_h,
    int order_w, bool need_permute_);

void cvi_backend_tg_fixed_upsample_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, int input_n, int input_c, int input_h, int input_w,
    int h_factor, int w_factor);

void cvi_backend_tg_fixed_lrn_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr,
    gaddr_t output_gaddr, gaddr_t sqr_lut_gaddr,
    gaddr_t power_lut_gaddr, int input_n, int input_c,
    int input_h, int input_w, int local_size,
    int sum_right_shift_width,
    int lrn_right_shift_width, int quant_data0,
    int quant_data1);

void cvi_backend_tg_fixed_reorg_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int height, int width,
    int stride);

void cvi_backend_tg_fixed_eltwise_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_max_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_min_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);


//////////////// bf16 kernel API /////////////////
void cvi_backend_tg_bf16_conv_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean,
    gaddr_t ga_bn_variance, gaddr_t ga_scale, gaddr_t ga_scale_bias,
    int input_n, int input_c, int input_h, int input_w, int groups,
    int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t stride_h, uint8_t stride_w, int do_bias,
    int do_bn, int do_scale, int do_scale_bias, int do_activation,
    float bn_scale, float bn_eps, int activation_method, float activation_arg[],
    gaddr_t activation_ga_slope);

void cvi_backend_tg_bf16_concat_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddrs[], gaddr_t output_gaddr, int input_dims[],
    int input_num, int concat_axis, int output_dim_size, int *output_dim,
    bool do_relu, const int need_quantize_num, const int *threshold_x_quantized);

void cvi_backend_tg_bf16_eltwise_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input[], gaddr_t ga_output, int input_size, int op,
    int input_n, int input_c, int input_h, int input_w,
    bool do_relu, float relu_slope,
    bool do_early_stride, int stride_h, int stride_w,
    const float coeffs[]);

void cvi_backend_tg_bf16_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr, gaddr_t index_gaddr,
    gaddr_t o_findex_gaddr, int n, int c, int h, int w, int kh, int kw,
    int pad_top, int pad_bot, int pad_left, int pad_right,
    int stride_h, int stride_w, int is_avg_pooling, float avg_const,
    int do_relu, const bool ceil_mode);

void cvi_backend_tg_bf16_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t bottom_data_gaddr, gaddr_t weight_data_gaddr,
    gaddr_t bias_data_gaddr, gaddr_t top_data_gaddr, int in_row, int in_col,
    int out_col, int have_bias, int do_activation, int activation_method);

void cvi_backend_tg_bf16_leakyrelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_bottom, gaddr_t ga_top, float ga_negative_slope,
    int input_n, int input_c, int input_h, int input_w);

void cvi_backend_tg_bf16_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_bottom,
    gaddr_t ga_top, gaddr_t ga_negative_slope, int input_n, int input_c,
    int input_h, int input_w);

void cvi_backend_tg_bf16_scale_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h,
    int input_w, int scale_dim, int inner_dim, bool is_scale_const,
    int const_scale, int do_activation, int activation_method,
    float activation_arg[], bool do_bias, bool second_is_load_weight);

void cvi_backend_tg_bf16_lut_scientific_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    gaddr_t exp_lut_table, gaddr_t mantissa_lut_table, int input_n, int input_c,
    int input_h, int input_w, cvk_fmt_t fmt) ;

void cvi_backend_tg_bf16_lut_interpolation_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t bottom_gaddr,
    gaddr_t top_gaddr, gaddr_t y0_table_gaddr, gaddr_t slope_gaddr, int input_n,
    int input_c, int input_h, int input_w, float range_min, float range_max,
    float scale);

void cvi_backend_tg_bf16_premute_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr, gaddr_t output_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    int output_n, int output_c, int output_h, int output_w,
    int order_n, int order_c, int order_h, int order_w,
    bool need_permute_);

void cvi_backend_tg_bf16_upsample_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c,
    int input_h, int input_w, int h_factor, int w_factor);

void cvi_backend_tg_fixed_pixel_shuffle_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, int input_n, int input_c, int input_h, int input_w, int factor);

void cvi_backend_tg_bf16_pixel_shuffle_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c,
    int input_h, int input_w, int factor);

void cvi_backend_tg_bf16_gru_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_recurrence,
    gaddr_t ga_bias, gaddr_t ga_initial_h,
    gaddr_t ga_sigmoid_table_data_lut, gaddr_t ga_sigmoid_slope_table_data_lut,
    gaddr_t ga_tanh_table_data_lut, gaddr_t ga_tanh_slope_table_data_lut,
    gaddr_t ga_output,
    int seq_len, int batch_size, int input_size, int hidden_size,
    bool do_bias, bool is_linear_before_reset, bool is_bidirectional);

void cvi_backend_tg_bf16_lstm_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_recurrence,
    gaddr_t ga_bias, gaddr_t ga_initial_h, gaddr_t ga_initial_c,
    gaddr_t ga_sigmoid_table_data_lut, gaddr_t ga_sigmoid_slope_table_data_lut,
    gaddr_t ga_tanh_table_data_lut, gaddr_t ga_tanh_slope_table_data_lut,
    gaddr_t ga_output,
    int seq_len, int batch_size, int input_size, int hidden_size,
    bool do_bias, bool is_bidirectional);

void cvi_backend_tg_bf16_softmax_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input,
    gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
    gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
    gaddr_t ga_output,
    int64_t* shape, int axis, int dimension);


////////////// fixed & bf16 kernel api ////////////////

// shuffle channel, batch = shape[0], channel = shape[1], frame_size = shape[2]
// * shape[3] * ... suppor int8 and bf16
void cvi_backend_tg_shuffle_channel_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int frame_size,
    int group, cvk_fmt_t fmt);

// slice, refer to tg op, support int8 and bf16
void cvi_backend_tg_slice_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int input_dim_size,
    int *input_dim, int axis, int offset, int length,
    cvk_fmt_t fmt);

// tile h/w, plz refer np.tile for more details
int cvi_backend_tg_tile_kernel(
    const CviBackendContext &ctx,
    gaddr_t input_gaddr, int input_n, int input_c, int input_h, int input_w,
    cvk_fmt_t input_fmt, gaddr_t output_gaddr,
    int output_n, int output_c, int output_h, int output_w,
    cvk_fmt_t output_fmt, int* tile_factors, int tile_factors_len,
    uint32_t layer_id);

void cvi_backend_tg_pad_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int *pads, float const_val,
    cvk_fmt_t fmt);

// swap channel by order
// e.g. if order is 2/1/0，then swap c[0]/c[1]/c[2] to c[2]/c[1]/c[0]
void cvi_backend_tg_swap_channel_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int input_dim_size,
    int *input_dim, int * channel_order, cvk_fmt_t fmt);


////////// data format convert API (need to clear up later) /////////////////
void mixed_precision_quant(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    cvk_fmt_t from, cvk_fmt_t to,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    float const_scale);

void mixed_precision_dequant(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    cvk_fmt_t from, cvk_fmt_t to,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    float const_scale);

void mixed_precision_tg_bf16_s8(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale);

void mixed_precision_tg_s8_bf16(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale);

void cvi_backend_tg_bf16_reorg_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int height, int width, int stride);

void convert_fp32_bf16_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int height, int width);

void convert_bf16_fp32_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int height, int width);


#endif /* CVI_BACKEND_TG_API */
