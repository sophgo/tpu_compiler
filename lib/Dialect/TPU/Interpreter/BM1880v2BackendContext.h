/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef _BM1880v2_BACKEND_CONTEXT_H_
#define _BM1880v2_BACKEND_CONTEXT_H_

#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <bmkernel/bm1880v2/compression.h>
#include "BackendContext.h"
#include <vector>


#define BM_CHIP_UNKNOWN -1
#define BM_CHIP_BM1680   0
#define BM_CHIP_BM1682   1
#define BM_CHIP_BM1684   2
#define BM_CHIP_BM1880   3
#define BM_CHIP_BM1882   4
#define BM_CHIP_BM1880v2 5
#define BM_CHIP_INVALID  6
#define INVALID_GLOBAL_ADDR 0xFFFFFFFFFFFFFFFF


// namespace bmnet {

class BM1880v2BackendContext : public BM188xBackendContext {
 public:
  BM1880v2BackendContext(int chip_ver, int nodechip_num, std::vector<int8_t> &weight);
  ~BM1880v2BackendContext() override;

  void submit();

  void parallel_enable() const override { return bmk1880v2_parallel_enable(bmk_); }

  void parallel_disable() const override { return bmk1880v2_parallel_disable(bmk_); }

  void set_layer_id(u16 layer_id) const override { bmk1880v2_set_layer_id(bmk_, layer_id); }

  int layer_id() const override { return bmk1880v2_layer_id(bmk_); }

  void create_streams(int nr_streams) const { return bmk1880v2_create_streams(bmk_, nr_streams); }

  void destroy_streams() const { return bmk1880v2_destroy_streams(bmk_); }

  void set_stream(int i) const { return bmk1880v2_set_stream(bmk_, i); }

  void add_dependency(bmk1880v2_op_t *before, bmk1880v2_op_t *after) const {
    return bmk1880v2_add_dependency(bmk_, before, after);
  }

  bmk1880v2_op_t *tdma_l2l_tensor_copy(const bmk1880v2_tdma_l2l_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_l2l_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2l_tensor_lrn_shift(
      const bmk1880v2_tdma_l2l_tensor_lrn_shift_param_t *p) const {
    return bmk1880v2_tdma_l2l_tensor_lrn_shift(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_tensor_copy(const bmk1880v2_tdma_l2tg_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_l2t_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_tensor_copy(const bmk1880v2_tdma_l2tg_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_l2g_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_tensor_copy_nc_transposed(
      const bmk1880v2_tdma_l2tg_tensor_copy_nc_transposed_param_t *p) const {
    return bmk1880v2_tdma_l2t_tensor_copy_nc_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_tensor_copy_nc_transposed(
      const bmk1880v2_tdma_l2tg_tensor_copy_nc_transposed_param_t *p) const {
    return bmk1880v2_tdma_l2g_tensor_copy_nc_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_tensor_copy_cw_transposed(
      const bmk1880v2_tdma_l2tg_tensor_copy_cw_transposed_param_t *p) const {
    return bmk1880v2_tdma_l2t_tensor_copy_cw_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_tensor_copy_cw_transposed(
      const bmk1880v2_tdma_l2tg_tensor_copy_cw_transposed_param_t *p) const {
    return bmk1880v2_tdma_l2g_tensor_copy_cw_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_tensor_copy_compressed(
      const bmk1880v2_tdma_l2tg_tensor_copy_compressed_param_t *p) const {
    return bmk1880v2_tdma_l2t_tensor_copy_compressed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_tensor_copy_compressed(
      const bmk1880v2_tdma_l2tg_tensor_copy_compressed_param_t *p) const {
    return bmk1880v2_tdma_l2g_tensor_copy_compressed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_tensor_fill_constant(
      const bmk1880v2_tdma_l2tg_tensor_fill_constant_param_t *p) const {
    return bmk1880v2_tdma_l2t_tensor_fill_constant(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_tensor_fill_constant(
      const bmk1880v2_tdma_l2tg_tensor_fill_constant_param_t *p) const {
    return bmk1880v2_tdma_l2g_tensor_fill_constant(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_matrix_copy(const bmk1880v2_tdma_l2tg_matrix_copy_param_t *p) const {
    return bmk1880v2_tdma_l2t_matrix_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_matrix_copy(const bmk1880v2_tdma_l2tg_matrix_copy_param_t *p) const {
    return bmk1880v2_tdma_l2g_matrix_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2t_general_copy(const bmk1880v2_tdma_l2tg_general_copy_param_t *p) const {
    return bmk1880v2_tdma_l2t_general_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_l2g_general_copy(const bmk1880v2_tdma_l2tg_general_copy_param_t *p) const {
    return bmk1880v2_tdma_l2g_general_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_tensor_copy(const bmk1880v2_tdma_tg2l_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_t2l_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_tensor_copy(const bmk1880v2_tdma_tg2l_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_g2l_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_tensor_copy_nc_transposed(
      const bmk1880v2_tdma_tg2l_tensor_copy_nc_transposed_param_t *p) const {
    return bmk1880v2_tdma_t2l_tensor_copy_nc_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_tensor_copy_nc_transposed(
      const bmk1880v2_tdma_tg2l_tensor_copy_nc_transposed_param_t *p) const {
    return bmk1880v2_tdma_g2l_tensor_copy_nc_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_tensor_copy_chw_rotated(
      const bmk1880v2_tdma_tg2l_tensor_copy_chw_rotated_param_t *p) const {
    return bmk1880v2_tdma_t2l_tensor_copy_chw_rotated(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_tensor_copy_chw_rotated(
      const bmk1880v2_tdma_tg2l_tensor_copy_chw_rotated_param_t *p) const {
    return bmk1880v2_tdma_g2l_tensor_copy_chw_rotated(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_tensor_copy_decompressed(
      const bmk1880v2_tdma_tg2l_tensor_copy_decompressed_param_t *p) const {
    return bmk1880v2_tdma_t2l_tensor_copy_decompressed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_tensor_copy_decompressed(
      const bmk1880v2_tdma_tg2l_tensor_copy_decompressed_param_t *p) const {
    return bmk1880v2_tdma_g2l_tensor_copy_decompressed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_tg2l_tensor_fill_constant(
      const bmk1880v2_tdma_tg2l_tensor_fill_constant_param_t *p) const {
    return bmk1880v2_tdma_tg2l_tensor_fill_constant(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_matrix_copy(const bmk1880v2_tdma_tg2l_matrix_copy_param_t *p) const {
    return bmk1880v2_tdma_t2l_matrix_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_matrix_copy(const bmk1880v2_tdma_tg2l_matrix_copy_param_t *p) const {
    return bmk1880v2_tdma_g2l_matrix_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_matrix_copy_row_col_transposed(
      const bmk1880v2_tdma_tg2l_matrix_copy_row_col_transposed_param_t *p) const {
    return bmk1880v2_tdma_t2l_matrix_copy_row_col_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_matrix_copy_row_col_transposed(
      const bmk1880v2_tdma_tg2l_matrix_copy_row_col_transposed_param_t *p) const {
    return bmk1880v2_tdma_g2l_matrix_copy_row_col_transposed(bmk_, p);
  }

  bmk1880v2_op_t *tdma_t2l_general_copy(const bmk1880v2_tdma_tg2l_general_copy_param_t *p) const {
    return bmk1880v2_tdma_t2l_general_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2l_general_copy(const bmk1880v2_tdma_tg2l_general_copy_param_t *p) const {
    return bmk1880v2_tdma_g2l_general_copy(bmk_, p);
  }

  bmk1880v2_op_t *tdma_g2g_tensor_copy(const bmk1880v2_tdma_tg2tg_tensor_copy_param_t *p) const {
    return bmk1880v2_tdma_tg2tg_tensor_copy(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_mul(const bmk1880v2_tiu_element_wise_mul_param_t *p) const {
    return bmk1880v2_tiu_element_wise_mul(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_mac(const bmk1880v2_tiu_element_wise_mac_param_t *p) const {
    return bmk1880v2_tiu_element_wise_mac(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_add(const bmk1880v2_tiu_element_wise_add_param_t *p) const {
    return bmk1880v2_tiu_element_wise_add(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_sub(const bmk1880v2_tiu_element_wise_sub_param_t *p) const {
    return bmk1880v2_tiu_element_wise_sub(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_max(const bmk1880v2_tiu_element_wise_max_param_t *p) const {
    return bmk1880v2_tiu_element_wise_max(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_min(const bmk1880v2_tiu_element_wise_min_param_t *p) const {
    return bmk1880v2_tiu_element_wise_min(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_arith_shift(
      const bmk1880v2_tiu_element_wise_arith_shift_param_t *p) const {
    return bmk1880v2_tiu_element_wise_arith_shift(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_and_int8(
      const bmk1880v2_tiu_element_wise_and_int8_param_t *p) const {
    return bmk1880v2_tiu_element_wise_and_int8(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_and_int16(
      const bmk1880v2_tiu_element_wise_and_int16_param_t *p) const {
    return bmk1880v2_tiu_element_wise_and_int16(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_or_int8(
      const bmk1880v2_tiu_element_wise_or_int8_param_t *p) const {
    return bmk1880v2_tiu_element_wise_or_int8(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_or_int16(
      const bmk1880v2_tiu_element_wise_or_int16_param_t *p) const {
    return bmk1880v2_tiu_element_wise_or_int16(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_xor_int8(
      const bmk1880v2_tiu_element_wise_xor_int8_param_t *p) const {
    return bmk1880v2_tiu_element_wise_xor_int8(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_xor_int16(
      const bmk1880v2_tiu_element_wise_xor_int16_param_t *p) const {
    return bmk1880v2_tiu_element_wise_xor_int16(bmk_, p);
  }

  bmk1880v2_op_t *tiu_element_wise_copy(const bmk1880v2_tiu_element_wise_copy_param_t *p) const {
    return bmk1880v2_tiu_element_wise_copy(bmk_, p);
  }

  bmk1880v2_op_t *tiu_mdsum(const bmk1880v2_tiu_mdsum_param_t *p) const {
    return bmk1880v2_tiu_mdsum(bmk_, p);
  }

  bmk1880v2_op_t *tiu_lookup_table(const bmk1880v2_tiu_lookup_table_param_t *p) const {
    return bmk1880v2_tiu_lookup_table(bmk_, p);
  }

  bmk1880v2_op_t *tiu_convolution(const bmk1880v2_tiu_convolution_param_t *p) const {
    return bmk1880v2_tiu_convolution(bmk_, p);
  }

  bmk1880v2_op_t *tiu_max_pooling(const bmk1880v2_tiu_max_pooling_param_t *p) const {
    return bmk1880v2_tiu_max_pooling(bmk_, p);
  }

  bmk1880v2_op_t *tiu_average_pooling(const bmk1880v2_tiu_average_pooling_param_t *p) const {
    return bmk1880v2_tiu_average_pooling(bmk_, p);
  }

  bmk1880v2_op_t *tiu_depthwise_convolution(
      const bmk1880v2_tiu_depthwise_convolution_param_t *p) const {
    return bmk1880v2_tiu_depthwise_convolution(bmk_, p);
  }

  bmk1880v2_op_t *tiu_matrix_multiplication(
      const bmk1880v2_tiu_matrix_multiplication_param_t *p) const {
    return bmk1880v2_tiu_matrix_multiplication(bmk_, p);
  }

  bmk1880v2_tensor_lmem_stride_t tensor_lmem_default_stride(bmk1880v2_tensor_lmem_shape_t s,
                                                            int eu_align) const {
    return bmk1880v2_tensor_lmem_default_stride(bmk_, s, eu_align);
  }

  bmk1880v2_tensor_tgmem_stride_t tensor_tgmem_default_stride(
      bmk1880v2_tensor_tgmem_shape_t s) const {
    return bmk1880v2_tensor_tgmem_default_stride(s);
  }

  bmk1880v2_matrix_lmem_shape_t matrix_lmem_default_shape(u32 row, u32 col) const {
    return bmk1880v2_matrix_lmem_default_shape(bmk_, row, col);
  }

  bmk1880v2_matrix_lmem_shape_t matrix_lmem_shape_t1(u32 len) const {
    return bmk1880v2_matrix_lmem_shape_t1(bmk_, len);
  }

  bmk1880v2_matrix_lmem_stride_t matrix_lmem_default_stride(bmk1880v2_matrix_lmem_shape_t s) const {
    return bmk1880v2_matrix_lmem_default_stride(bmk_, s);
  }

  bmk1880v2_tensor_lmem_t *lmem_alloc_tensor(bmk1880v2_tensor_lmem_shape_t s, fmt_t fmt,
                                             int eu_align) const {
    return bmk1880v2_lmem_alloc_tensor(bmk_, s, fmt, eu_align);
  }

  u32 lmem_tensor_to_size(bmk1880v2_tensor_lmem_shape_t s, int eu_align) const {
    return bmk1880v2_lmem_tensor_to_size(bmk_, s, eu_align);
  }

  void lmem_free_tensor(const bmk1880v2_tensor_lmem_t *t) const {
    return bmk1880v2_lmem_free_tensor(bmk_, t);
  }

  bmk1880v2_matrix_lmem_t *lmem_alloc_matrix(bmk1880v2_matrix_lmem_shape_t s, fmt_t fmt,
                                             int eu_align) const {
    return bmk1880v2_lmem_alloc_matrix(bmk_, s, fmt, eu_align);
  }

  void lmem_free_matrix(const bmk1880v2_matrix_lmem_t *t) const {
    return bmk1880v2_lmem_free_matrix(bmk_, t);
  }

  //
  // BM kernel legacy API
  //
  u32 tl_shape_to_size(bmk1880v2_tensor_lmem_shape_t shape, bool aligned, fmt_t fmt) const;

  inline u32 tl_address(const bmk1880v2_tensor_lmem_t &tlp) const { return tlp.start_address; }
  inline u32 tl_address(const bmk1880v2_tensor_lmem_t *tlp) const { return tlp->start_address; }

  inline bmk1880v2_tensor_lmem_shape_t shape_t4(int n, int c, int h, int w) const {
    return {static_cast<u32>(n), static_cast<u32>(c), static_cast<u32>(h), static_cast<u32>(w)};
  }

  inline bmk1880v2_tensor_lmem_t tl_prealloc_align(u32 la_addr, bmk1880v2_tensor_lmem_shape_t shape,
                                                   fmt_t fmt) const {
    bmk1880v2_tensor_lmem_t tl_data = {0};
    tl_data.start_address = la_addr;
    tl_data.fmt = fmt;
    tl_data.shape = shape;
    tl_data.stride = this->tensor_lmem_default_stride(tl_data.shape, /*eu_aligned=*/1);

    return tl_data;
  }

  inline bmk1880v2_tensor_lmem_t tl_prealloc(u32 la_addr, bmk1880v2_tensor_lmem_shape_t shape,
                                             fmt_t fmt) const {
    bmk1880v2_tensor_lmem_t tl_data = {0};
    tl_data.start_address = la_addr;
    tl_data.fmt = fmt;
    tl_data.shape = shape;
    tl_data.stride = this->tensor_lmem_default_stride(tl_data.shape, /*eu_aligned=*/0);

    return tl_data;
  }

  //
  // TDMA legacy API
  //
  void tdma_load_stride(bmk1880v2_tensor_lmem_t *tlp, u64 ga_src,
                        bmk1880v2_tensor_tgmem_stride_t ts_stride, ctrl_t ctrl) const;

  void tdma_load(bmk1880v2_tensor_lmem_t *tlp, u64 ga_src, ctrl_t ctrl) const;

  void tdma_store_stride(bmk1880v2_tensor_lmem_t *tlp, u64 ga_dst,
                         bmk1880v2_tensor_tgmem_stride_t ts_stride, ctrl_t ctrl) const;

  void tdma_store(bmk1880v2_tensor_lmem_t *tlp, u64 ga_dst, ctrl_t ctrl) const;

  void tdma_tg_copy(bmk1880v2_tensor_tgmem_t *dst, bmk1880v2_tensor_tgmem_t *src,
                    ctrl_t ctrl) const;

  void _tdma_tg_copy(bmk1880v2_tensor_tgmem_t *dst, bmk1880v2_tensor_tgmem_t *src) const;
  bool _tdma_tg_copy_no_split(bmk1880v2_tensor_tgmem_t *dst, bmk1880v2_tensor_tgmem_t *src) const;
  bool _tdma_tg_copy_split_nh(bmk1880v2_tensor_tgmem_t *dst, bmk1880v2_tensor_tgmem_t *src) const;

  // matrix format
  void tdma_load_stride(bmk1880v2_matrix_lmem_t *tlp, u64 ga_src,
                        bmk1880v2_matrix_tgmem_stride_t ts_stride, ctrl_t ctrl) const;

  bool is_compress(int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, u64 ga_src,
                   ctrl_t ctrl) const;
  void tdma_load(bmk1880v2_matrix_lmem_t *tlp, u64 ga_src, ctrl_t ctrl) const;

  void tdma_store_stride(bmk1880v2_matrix_lmem_t *tlp, u64 ga_dst,
                         bmk1880v2_matrix_tgmem_stride_t ts_stride, ctrl_t ctrl) const;

  void tdma_store(bmk1880v2_matrix_lmem_t *tlp, u64 ga_dst, ctrl_t ctrl) const;

  enum { NEURON_MEMORY = 0, WEIGHT_MEMORY = 1 };

  bmk1880v2_context_t *bm_get_bmk() const { return bmk_; }

 private:
  bmk1880v2_context_t *bmk_;
};

//}  // namespace bmnet

#endif /* _BM1880v2_BACKEND_CONTEXT_H_ */
