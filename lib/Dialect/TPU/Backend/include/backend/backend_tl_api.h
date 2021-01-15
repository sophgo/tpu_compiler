/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_tl_api.h
 * Description:
 */

#ifndef CVI_BACKEND_TL_API
#define CVI_BACKEND_TL_API

#include <backend/backend_common.h>

class CviBackendContext;

void cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap, cvk_fmt_t fmt,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw);

void cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw,
    bool doDecompress);

void cvi_backend_tl_store(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ofmap, gaddr_t ga_ofmap, cvk_fmt_t fmt,
    uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow);

void cvi_backend_tl_to_tensor(
    const CviBackendContext &ctx,
    cvk_tl_t *tensor,
    laddr_t la,
    uint32_t tensor_n, uint32_t tensor_c, uint32_t tensor_h, uint32_t tensor_w,
    cvk_fmt_t fmt, uint8_t eu_align);

void cvi_backend_tl_load_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H,
    int Local_W, int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to,
    bool DoDecompressed);

void cvi_backend_tl_load_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to);

void cvi_backend_tl_load_compressed(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H,
    int Local_W, int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to,
    int h_step, int step_size, int c_step);

void cvi_backend_tl_store_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H,
    int Local_W, int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to);

void cvi_backend_tl_store_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H,
    int Local_W, int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to, bool DoCompress);

void cvi_backend_tl_store_compressed(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to,
    int h_step, int step_size, int c_step,
    bool DoIntraCmdParal = false);

void cvi_backend_tl_copy(
    const CviBackendContext &ctx, uint32_t layer_id,
    int la_src, int la_dst,
    int n, int c, int h, int w,
    bool align, cvk_fmt_t fmt);

///
/// Self Load/Store All
///   This assumes input/output neuron no need to do slicing/tiling
///   Weight will be sliced into lane number size
///   Function handles load/store input/output and load weight internally
///
void cvi_backend_tl_conv_LA(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    gaddr_t ga_filter, gaddr_t ga_perchannel,
    uint32_t input_n, uint32_t input_c, uint32_t input_h, uint32_t input_w,
    uint32_t groups, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint16_t kh, uint16_t kw, uint8_t dilation_h, uint8_t dilation_w,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t stride_h, uint8_t stride_w,
    bool result_add, bool with_bias, bool do_relu, bool do_ic_alignment);

///
/// Self Load Weight
///   This assumes input/output neuron no need to do slicing/tiling
///   Weight will be sliced into lane number size
///   Function handles load weight internally, but NOT input/output load/store
///
void cvi_backend_tl_conv_LW(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_working,
    gaddr_t ga_filter, gaddr_t ga_perchannel,
    uint32_t input_n, uint32_t input_c, uint32_t input_h, uint32_t input_w,
    uint32_t groups, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint16_t kh, uint16_t kw, uint8_t dilation_h, uint8_t dilation_w,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t stride_h, uint8_t stride_w,
    bool result_add, bool with_bias, bool do_relu,
    bool do_store, gaddr_t ga_ofmap,
    bool do_leaky_relu,
    int8_t rshift_pos, int8_t m_i8_pos,
    int8_t rshift_neg, int8_t m_i8_neg, bool do_ic_alignment,
    bool compressed_weight);

///
/// For both Self Load and No Load
///   This assumes input/output neuron no need to do slicing/tiling
///   if la addresses are provided, use la addresses
///   otherwise, alloc tl memory internally (and do load/store)
///   if do_load is set, load input from ga_input
///   if do_store is set, store output to ga_output
///   addend always do loading internally
///
void cvi_backend_tl_eltwise_op(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    gaddr_t ga_input, gaddr_t ga_output, gaddr_t ga_addend,
    int op_code, int n, int c, int h, int w, bool do_relu,
    bool do_early_stride, int stride_h, int stride_w,
    int8_t rshift, int8_t m_i8_input, int8_t m_i8_addend, int32_t i32Multiplier,
    bool do_load, bool do_store);

// layer group added cvi_backend_tl_xxxx
void cvi_backend_tl_conv(
  const CviBackendContext& ctx, uint32_t layer_id,
  laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
  laddr_t la_working, laddr_t la_perchannel,
  int input_n, int input_c, int input_h, int input_w,
  int group, int output_c, int output_h, int output_w,
  uint32_t kh, uint32_t kw, uint32_t dilation_h, uint32_t dilation_w,
  uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left, uint32_t pad_w_right,
  uint32_t stride_h, uint32_t stride_w,
  uint32_t result_add, uint32_t ctrl, bool do_bias,
  bool do_relu, float neg_slope,
  int rshift, int rshift_len,
  int8_t rshift_pos, int8_t rshift_neg, int8_t m_i8_pos, int8_t m_i8_neg,
  bool do_ic_alignment);

void cvi_backend_bf16_tl_conv(
  const CviBackendContext& ctx, uint32_t layer_id,
  laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
  laddr_t la_working, laddr_t la_bias,
  int input_n, int input_c, int input_h, int input_w,
  int group, int output_c, int output_h, int output_w,
  uint32_t kh, uint32_t kw, uint32_t dilation_h, uint32_t dilation_w,
  uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left, uint32_t pad_w_right,
  uint32_t stride_h, uint32_t stride_w,
  bool do_bias, bool do_relu);

void cvi_backend_tl_eltwise(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t *la_input, laddr_t la_output, laddr_t la_working,
    int input_n, int input_c, int input_h,
    int input_w, int input_size, int op,
    int8_t rshift_i8, const int8_t *m_i8,
    bool use_default_coeff,
    bool do_relu, float relu_slope, const int *coeffs, const int i32Multiplier,
    bool do_early_stride, int stride_h, int stride_w);

void cvi_backend_bf16_tl_eltwise(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t *la_input, laddr_t la_output, laddr_t la_working,
    int input_n, int input_c, int input_h, int input_w,
    int input_size, int op,
    bool use_default_coeff,
    bool do_relu, float relu_slope,
    const int *coeffs,
    bool do_early_stride,
    int stride_h, int stride_w);

void cvi_backend_tl_pooling(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr,
    int input_n, int input_c, int input_h, int input_w,
    int output_n, int output_c, int output_h, int output_w,
    uint32_t kh, uint32_t kw, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left, uint32_t pad_w_right,
    bool is_avg_pooling,
    int8_t rshift_i8, int8_t m_i8);

void cvi_backend_tl_bf16_pooling(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr,
    int input_n, int input_c, int input_h, int input_w,
    int output_n, int output_c, int output_h, int output_w,
    uint32_t kh, uint32_t kw, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left, uint32_t pad_w_right,
    bool is_avg_pooling);

void cvi_backend_tl_lrn(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, laddr_t sqr_lut_laddr,
    laddr_t power_lut_laddr, laddr_t working_laddr,
    int input_n, int input_c, int input_h, int input_w, int size,
    int8_t sum_rshift_i8, int8_t lrn_rshift_i8,
    int8_t *m_i8);

void cvi_backend_bf16_tl_lrn(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, laddr_t power_exp_table,
    laddr_t power_mantissa_table, laddr_t working_laddr,
    int input_n, int input_c, int input_h, int input_w, int size,
    float alpha, float k);

void cvi_backend_tl_lut_LA(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    gaddr_t ga_input, gaddr_t ga_output, gaddr_t sg_lut_gaddr,
    int n, int c, int h, int w,
    bool do_load, bool do_store);

void cvi_backend_tl_lut(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_y_table, laddr_t la_slope_lut,
    int thresh_min, int thresh_max,
    int n, int c, int h, int w);

void cvi_backend_bf16_tl_lut(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_y_table, laddr_t la_slope_lut,
    int thresh_min, int thresh_max, bool added_offset,
    int n, int c, int h, int w, int method);

void cvi_backend_tl_lut_exponential_mul_mantissa(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_exponential_table, laddr_t la_mantissa_lut,
    int n, int c, int h, int w);

void cvi_backend_bf16_tl_lut_slope_method(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_y_table, laddr_t la_slope_table,
    int thresh_min, int thresh_max, bool added_offset, int n, int c, int h, int w);

void cvi_backend_tl_prelu(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output,
    laddr_t la_slope,
    int input_n, int input_c,
    int input_h, int input_w,
    int8_t r_i8_pos, int8_t m_i8_pos,
    int8_t r_i8_neg);

void cvi_backend_tl_bf16_prelu(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    laddr_t la_input,
    laddr_t la_output,
    laddr_t la_slope,
    int n, int c,
    int h, int w);

void cvi_backend_tl_scale_qi32(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_working,
    gaddr_t scale_gaddr,
    gaddr_t bias_gaddr,  int input_n,
    int input_c, int input_h, int input_w, int scale_dim,
    int inner_dim, bool is_scale_const, int const_scale,
    int do_activation,
    int activation_method,
    float activation_arg[],
    bool do_bias,
    bool is2ndSrcFromWeight); // true means second comes from weight, otherwise comes from another input

void cvi_backend_tl_concat(
    const CviBackendContext &ctx, uint32_t layer_id,
    int *input_dim_c, int input_size, int *output_dim,
    laddr_t *la_input, laddr_t la_output, bool do_relu,
    int8_t rshift, int32_t * m_i8);

void cvi_backend_tl_bf16_concat(
    const CviBackendContext &ctx, uint32_t layer_id,
    int *input_dim_c, int input_size, int *output_dim,
    laddr_t *la_input, laddr_t la_output, bool do_relu);

void cvi_backend_tl_depthwise_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_working, laddr_t la_perchannel,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu,
    int rshift, int rshift_len);

void cvi_backend_tl_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_perchannel,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu,
    int rshift, int rshift_len,
    bool do_ic_alignment);

void cvi_backend_tl_bf16_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_bias,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu);

void cvi_backend_tl_broadcast_mul(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t input_laddr, laddr_t scale_laddr,
    laddr_t bias_laddr, laddr_t output_laddr, int input_n,
    int input_c, int input_h, int input_w, int scale_dim,
    int inner_dim, bool is_scale_const, int const_scale,
    int right_shift_width,
    int do_activation,
    int activation_method,
    float activation_arg[],
    const int *i8_multiplier, // INT8_PER_LAYER
    bool do_bias);

void cvi_backend_bf16_tl_broadcast_mul(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t input_laddr, laddr_t scale_laddr,
    laddr_t bias_laddr, laddr_t output_laddr, int input_n,
    int input_c, int input_h, int input_w,
    int do_activation,
    int activation_method,
    float activation_arg[],
    bool do_bias);

void cvi_backend_tl_upsample(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t input_laddr,
    laddr_t output_laddr, int input_n,
    int input_c, int input_h, int input_w,
    int scale_h, int scale_w);

void cvi_backend_tl_leaky_relu(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t input_laddr,
    laddr_t output_laddr, int input_n, int input_c,
    int input_h, int input_w,
    int GT_right_shift_width, int LE_right_shift_width,
    int GT_scale, int LE_scale);

void cvi_backend_bf16_tl_leaky_relu(
    const CviBackendContext &ctx,uint32_t layer_id,
    laddr_t input_laddr, laddr_t output_laddr,
    int input_n, int input_c,
    int input_h, int input_w,
    float neg_slope);

void cvi_backend_tl_pad(
    const CviBackendContext &ctx, uint32_t layer_id,
    int64_t *input_dim, int64_t *output_dim,
    laddr_t la_input, laddr_t la_output,
    float const_val, int32_t * pads);

void cvi_backend_tl_bf16_pad(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    int64_t *input_dim, int64_t *output_dim,
    laddr_t la_input, laddr_t la_output,
    float const_val, int32_t * pads);

void cvi_backend_tl_crop(
     const CviBackendContext &ctx, uint32_t layer_id,
     int64_t *input_dim, int64_t *output_dim,
     laddr_t la_input, laddr_t la_output,
     int *offsets);

void cvi_backend_tl_bf16_crop(
    const CviBackendContext &ctx, uint32_t layer_id,
    int64_t *input_dim, int64_t *output_dim,
    laddr_t la_input, laddr_t la_output,
    int *offsets);

void cvi_backend_tl_relu(
     const CviBackendContext &ctx, uint32_t layer_id,
     int n, int c, int h, int w,
     laddr_t la_input, laddr_t la_output);

void cvi_backend_tl_bf16_relu(
     const CviBackendContext &ctx, uint32_t layer_id,
     int n, int c, int h, int w,
     laddr_t la_input, laddr_t la_output);

void cvi_backend_tl_quant(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    cvk_fmt_t from, cvk_fmt_t to,
    float const_scale,
    int n, int c, int h, int w,
    bool bExtraInput);

void cvi_backend_tl_slice(
     const CviBackendContext &ctx, uint32_t layer_id,
     int64_t *input_dim, int64_t *output_dim,
     laddr_t la_input, laddr_t la_output,
     int axis, int offset);

void cvi_backend_tl_bf16_slice(
     const CviBackendContext &ctx, uint32_t layer_id,
     int64_t *input_dim, int64_t *output_dim,
     laddr_t la_input, laddr_t la_output,
     int axis, int offset);

void cvi_backend_tl_pixel_shuffle_LA(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, gaddr_t ga_ifmap,
    uint32_t input_n, uint32_t input_c, uint32_t input_h, uint32_t input_w,
    uint32_t factor);

// output = input * multiplier + const_val
void cvi_backend_tl_mac_const(const CviBackendContext &ctx, uint32_t layer_id,
                              gaddr_t input_addr, gaddr_t output_addr,
                              gaddr_t working_addr, int n, int c, int h, int w,
                              int multiplier, int const_val, bool do_relu);

void cvi_backend_bf16_tl_mac_const(const CviBackendContext &ctx,
                                   uint32_t layer_id, laddr_t input_addr,
                                   laddr_t output_addr, int n, int c, int h,
                                   int w, float multiplier, float const_val,
                                   bool do_relu);

void cvi_backend_tl_bf16_ps32_to_fp32(const CviBackendContext &ctx,
                                      uint32_t layer_id, laddr_t la_addr,
                                      int n, int c, int h, int w);

void cvi_backend_tl_store_fp32(const CviBackendContext &ctx,
                               uint32_t layer_id, gaddr_t ga_dst,
                               laddr_t la_src,
                               int n, int c, int h, int w);

void cvi_backend_ml_load_stride(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_src, laddr_t la_dst,
                                int Local_R, int Local_C,
                                int Global_C,
                                bool DoTranspose, bool DoAligned,
                                cvk_fmt_t from, cvk_fmt_t to,
                                bool DoDecompress);

#endif /* CVI_BACKEND_TL_API */
