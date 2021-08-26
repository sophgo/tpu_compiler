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
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, int input_n,
    int input_c, int input_h, int input_w, int groups, int output_c,
    uint16_t kh, uint16_t kw, uint16_t dilation_h, uint16_t dilation_w,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t insert_h, uint8_t insert_w, uint8_t stride_h, uint8_t stride_w,
    int do_bias, int do_activation, float activation_arg[],
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, int right_shift_width, bool do_chl_quan,
    bool do_ic_alignment, int store_cmpr_act, int load_cmpr_act,
    bool do_cmpr_wgt, int store_cmpr_act_c_step, int load_cmpr_act_c_step,
    int store_cmpr_act_h_step, int load_cmpr_act_h_step,
    int pad_value = 0, gaddr_t ga_scale_lut = GA_INVALID);

void cvi_backend_tg_fixed_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
    bool have_bias, bool do_relu, std::vector<int> rshift_width,
    std::vector<int> multiplier, std::vector<int> compressed_pos,
    int batch_high = 1, int batch_low = 1, bool lstride = false,
    bool rstride = false, bool ostride = false);

void cvi_backend_tg_fixed_max_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w,
    bool do_relu, bool ceil_mode,
    int store_cmpr_act = 0, int load_cmpr_act = 0,
    int store_cmpr_act_c_step = 0, int load_cmpr_act_c_step = 0);

void cvi_backend_tg_fixed_avg_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w,
    bool do_relu, int rshift, int multiplier, bool ceil_mode);

void cvi_backend_tg_fixed_leakyrelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    uint64_t input_gaddr, uint64_t output_gaddr,
    int input_n, int input_c, int input_h,
    int input_w, int GT_right_shift_width, int LE_right_shift_width, int GT_scale, int LE_scale,
    int input_offset, int output_offset);

void cvi_backend_tg_fixed_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    uint64_t bottom_gaddr, uint64_t top_gaddr, uint64_t negative_scope_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    int GT_right_shift_width, int GT_scale, int LE_right_shift_width);

void cvi_backend_tg_crop_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr, int *input_dim,
    int *output_dim, int *offsets, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_dilate_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int n, int c, int ih, int iw,
    int oh, int ow, int fill_constant,
    int ins_h, int ins_w,
    cvk_fmt_t fmt);

void cvi_backend_tg_fixed_lrn_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, gaddr_t ga_sqr_lut, gaddr_t ga_power_lut,
    int input_n, int input_c, int input_h, int input_w, int local_size,
    int sum_right_shift_width, int lrn_right_shift_width, int quant_data0,
    int quant_data1);

void cvi_backend_tg_fixed_eltwise_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs,
    int32_t *inputs_offset = nullptr,
    int32_t output_offset = 0,
    int32_t store_cmpr_act = 0, int32_t load_cmpr_act = 0,
    int32_t store_cmpr_act_c_step = 0, int32_t load_cmpr_act_c_step = 0);

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

void cvi_backend_tg_fixed_reduce_max_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int axes[], int num_axes);

void cvi_backend_tg_fixed_reduce_mean_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int rshift, int multiplier,
    int axes[], int num_axes);

void cvi_backend_tg_int8_broadcast_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t bn, int32_t bc, int32_t bh, int32_t bw,
    bool do_relu, const int32_t rshift,
    const int32_t *multipliers);

void cvi_backend_tg_int8_broadcast_sub_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t bn, int32_t bc, int32_t bh, int32_t bw,
    bool do_relu, const int32_t rshift,
    const int32_t *multipliers);

void cvi_backend_tg_int8_broadcast_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t bn, int32_t bc, int32_t bh, int32_t bw,
    bool do_relu, const int32_t rshift,
    const int32_t *multipliers);

//////////////// bf16 kernel API /////////////////
void cvi_backend_tg_bf16_conv_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_h, int input_w, int groups,
    int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t ins_h, uint8_t ins_w,
    uint8_t stride_h, uint8_t stride_w, int do_bias,
    int do_activation, bool fp32_output,
    int store_cmpr_act, int load_cmpr_act, bool do_cmpr_wgt,
    int store_cmpr_act_c_step, int load_cmpr_act_c_step,
    int store_cmpr_act_h_step, int load_cmpr_act_h_step);

void cvi_backend_tg_bf16_conv3d_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_c, int output_d, int output_h, int output_w,
    uint16_t kd, uint16_t kh, uint16_t kw,
    uint16_t dilation_d, uint16_t dilation_h, uint16_t dilation_w,
    uint8_t pad_d0, uint8_t pad_d1,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t stride_d, uint8_t stride_h, uint8_t stride_w,
    bool has_bias, bool do_relu);

void cvi_backend_tg_bf16_pooling_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr, gaddr_t index_gaddr,
    gaddr_t o_findex_gaddr, int n, int c, int h, int w, int kh, int kw,
    int pad_top, int pad_bot, int pad_left, int pad_right,
    int stride_h, int stride_w, int is_avg_pooling, float avg_const,
    int do_relu, const bool ceil_mode);

void cvi_backend_tg_bf16_max_pooling3d_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kd, int kh, int kw,
    int pad_d0, int pad_d1,
    int pad_top, int pad_bot, int pad_left, int pad_right,
    int stride_d, int stride_h, int stride_w,
    bool do_relu, const bool ceil_mode);

void cvi_backend_tg_bf16_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
    bool do_bias, bool do_relu, std::vector<int> compr_weight_poss,
    int batch_high = 1, int batch_low = 1, bool lstride = false,
    bool rstride = false, bool ostride = false);

void cvi_backend_tg_bf16_leakyrelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, float negative_slope, int n, int c, int h, int w);

void cvi_backend_tg_bf16_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_bottom,
    gaddr_t ga_top, gaddr_t ga_negative_slope, int input_n, int input_c,
    int input_h, int input_w);

void cvi_backend_tg_bf16_layernorm_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_table, gaddr_t ga_mantissa_table, gaddr_t ga_scale, gaddr_t ga_bias,
    gaddr_t ga_output, int batch_size, int normalized_size, float eps, bool affine = false);

void cvi_backend_tg_bf16_lut_mantissa_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t bottom_gaddr,
    gaddr_t top_gaddr, gaddr_t exp_lut_table, gaddr_t mantissa_lut_table,
    int input_n, int input_c, int input_h, int input_w);

void cvi_backend_tg_bf16_lut_slope_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t bottom_gaddr,
    gaddr_t top_gaddr, gaddr_t y0_table_gaddr, gaddr_t slope_gaddr, int input_n,
    int input_c, int input_h, int input_w, float range_min, float range_max);

void cvi_backend_tg_fixed_pixel_shuffle_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, int input_n, int input_c, int input_h, int input_w, int factor,
	bool isDCR);

void cvi_backend_tg_bf16_pixel_shuffle_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c,
    int input_h, int input_w, int factor,
	bool isDCR);

void cvi_backend_tg_bf16_gru_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_recurrence, gaddr_t ga_bias, gaddr_t ga_initial_h,
    gaddr_t ga_sigmoid_table_data_lut, gaddr_t ga_sigmoid_slope_table_data_lut,
    gaddr_t ga_tanh_table_data_lut, gaddr_t ga_tanh_slope_table_data_lut,
    gaddr_t ga_output, int seq_len, int num_dir, int batch_size,
    int hidden_size, bool do_bias, bool with_initial_h,
    bool is_linear_before_reset, bool is_bidirectional, bool only_last = false);

void cvi_backend_tg_bf16_lstm_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_recurrence, gaddr_t ga_bias, gaddr_t ga_initial_h, gaddr_t ga_inital_c,
    gaddr_t ga_sigmoid_table_data_lut, gaddr_t ga_sigmoid_slope_table_data_lut,
    gaddr_t ga_tanh_table_data_lut, gaddr_t ga_tanh_slope_table_data_lut,
    gaddr_t ga_output, int seq_len, int num_dir, int batch_size,
    int hidden_size, bool do_bias, bool with_initial_h, bool with_initial_c,
    bool is_bidirectional);

void cvi_backend_tg_bf16_softmax_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input,
    gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
    gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
    gaddr_t ga_output,
    int64_t* shape, int axis, int dimension);

void cvi_backend_tg_bf16_eltwise_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w,
    const float *coeffs,
    int32_t store_cmpr_act = 0, int32_t load_cmpr_act = 0,
    int32_t store_cmpr_act_c_step = 0, int32_t load_cmpr_act_c_step = 0);

void cvi_backend_tg_bf16_eltwise_max_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w,
    const float *coeffs);

void cvi_backend_tg_bf16_eltwise_min_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w,
    const float *coeffs);

void cvi_backend_tg_bf16_eltwise_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w,
    const float *coeffs);

void cvi_backend_tg_bf16_square_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, bool do_relu);

void cvi_backend_tg_bf16_eltwise_min_max_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w,
    const float *coeffs);

void cvi_backend_tg_eltwise_abs_kernel(const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num,
    int32_t n, int32_t c, int32_t h,
    int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs, cvk_fmt_t fmt);

void cvi_backend_tg_bf16_broadcast_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output, int n, int c,
    int h, int w, int bn, int bc, int bh, int bw, bool do_relu);

void cvi_backend_tg_bf16_broadcast_sub_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output, int n, int c,
    int h, int w, int bn, int bc, int bh, int bw, bool do_relu);

void cvi_backend_tg_bf16_broadcast_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output, int n, int c,
    int h, int w, int bn, int bc, int bh, int bw, bool do_relu);

void cvi_backend_tg_bf16_reduce_max_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int axes[], int num_axes);

void cvi_backend_tg_bf16_reduce_mean_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int axes[], int num_axes);

void cvi_backend_tg_bf16_lrn_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, gaddr_t exp_table_gaddr, gaddr_t mantissa_table_gaddr,
    int input_n, int input_c, int input_h, int input_w, int local_size,
    float alpha, float k);

////////////// fixed & bf16 kernel api ////////////////
void cvi_backend_tg_argmax_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int w_tile_size, cvk_fmt_t fmt);

void cvi_backend_tg_concat_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    int input_num, gaddr_t input_gaddrs[], gaddr_t output_gaddr,
    int axis_dims[], int concat_axis, int output_dim_size, int *output_dim,
    bool do_relu, const int *right_shift_width,
    const int *threshold_x_quantized, cvk_fmt_t fmt);

void cvi_backend_tg_lut_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                               gaddr_t bottom_gaddr, gaddr_t top_gaddr,
                               gaddr_t sg_lut_gaddr, int input_n, int input_c,
                               int input_h, int input_w, cvk_fmt_t fmt);

void cvi_backend_tg_relu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    uint64_t ga_input, uint64_t ga_output, int n,
    int c, int h, int w, cvk_fmt_t fmt);

void cvi_backend_tg_reorg_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t output_gaddr,
    int batch, int channel, int height, int width, int stride, cvk_fmt_t fmt);

void cvi_backend_tg_permute_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_n,
    int input_c, int input_h, int input_w, int order_n, int order_c, int order_h,
    int order_w, cvk_fmt_t fmt);

void cvi_backend_tg_pool_mask_kernel(const CviBackendContext &ctx,
                                     uint32_t layer_id, gaddr_t input_gaddr,
                                     gaddr_t output_gaddr, int n, int c, int h,
                                     int w, int scale, cvk_fmt_t fmt);

void cvi_backend_tg_quant_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    cvk_fmt_t from, cvk_fmt_t to,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    float const_scale = 1.0, int offset=0,
    int load_cmpr_act = 0, int load_cmpr_act_c_step = 0);

void cvi_backend_tg_requant_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t bottom_gaddr,
                                   gaddr_t top_gaddr, int input_n, int input_c,
                                   int input_h, int input_w, int input_offset,
                                   int output_offset, float scale);

void cvi_backend_tg_reverse_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int axis, cvk_fmt_t fmt);

void cvi_backend_tg_scale_lut_kernel(const CviBackendContext &ctx,
                                       uint32_t layer_id, gaddr_t bottom_gaddr,
                                       gaddr_t top_gaddr, gaddr_t table_gaddr,
                                       int input_n, int input_c, int input_h,
                                       int input_w, cvk_fmt_t fmt);

// slice, refer to tg op, support int8 and bf16
void cvi_backend_tg_slice_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_dim_size,
    int *input_dim, int axis, int offset, int length, cvk_fmt_t fmt);

void cvi_backend_tg_scale_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h,
    int input_w, int scale_dim, int inner_dim, bool is_scale_const,
    int const_scale, int do_relu, bool do_bias, cvk_fmt_t fmt);

void cvi_backend_tg_swap_channel_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_dim_size,
    int *input_dim, int * channel_order, cvk_fmt_t fmt);

// only support tile axis dim
void cvi_backend_tg_tile_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t input_gaddr, gaddr_t output_gaddr,
                                int n, int c, int h, int w, int axis,
                                int factor, cvk_fmt_t fmt);

void cvi_backend_tg_pad_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int *pads, float const_val,
    const char* mode, cvk_fmt_t fmt);

void cvi_backend_tg_reflectionpad_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, gaddr_t ga_left, gaddr_t ga_right, int outer_size,
    int working_size, std::vector<int> &pads, cvk_fmt_t fmt);

void cvi_backend_tg_fill_const_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ofmap, int output_n,
    int output_c, int output_h, int output_w, float const_val,
    cvk_fmt_t fmt);

void cvi_backend_tg_upsample_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c,
    uint32_t input_h, uint32_t input_w, uint32_t h_factor, uint32_t w_factor,
    cvk_fmt_t fmt);

void cvi_backend_tg_yuv420_csc_kernel(const CviBackendContext &ctx,
                                      uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, int n, int c, int h,
                                      int w, const std::vector<int> &order,
                                      cvk_fmt_t fmt);

#endif /* CVI_BACKEND_TG_API */
