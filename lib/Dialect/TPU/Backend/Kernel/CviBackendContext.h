/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: CviBackendContext.h
 * Description:
 */

#ifndef _BM18XX_BACKEND_CONTEXT_H_
#define _BM18XX_BACKEND_CONTEXT_H_

#include <vector>
#include <cstring>
#include <assert.h>
#include <cvikernel/cvikernel.h>
#include <backend/backend_tg_api.h>

#define MAX_CONV_IC (4095 - 32)
#define MAX_TIU_CHL (4095 - 32)
#define MAX_CHANNEL (4095 - 32)
#define MAX_HEIGHT (4095 - 32)
#define MAX_WIDTH (4095 - 32)
#define MAX_ROW (4095 - 32)
#define MAX_COL (4095 - 32)

#define NPU_NUM ctx.cvi_chip_info_context(CVI_CHIP_LANE_NUM)
#define EU_NUM ctx.cvi_chip_info_context(CVI_CHIP_EU_NUM)
#define LOCAL_MEM_SIZE ctx.cvi_chip_info_context(CVI_CHIP_LMEM_SIZE)
#define LOCAL_MEM_BANKS ctx.cvi_chip_info_context(CVI_CHIP_LMEM_BANK)

class CviBackendContext {
public:
  CviBackendContext(const char *runchip);
  ~CviBackendContext();

  void write_cmdbuf(const void *cmdbuf, uint32_t size);
  void read_cmdbuf(std::vector<uint8_t> &out_cmdbuf);
  void dmabuf_convert(std::vector<uint8_t> &dmabuf);
  void submit();

  int cvi_chip_info_context(CVI_CHIP_INFO_E cvi_chip_info_e) const;

  void parallel_enable() const { cvk_ctx_->ops->parallel_enable(cvk_ctx_); }
  void parallel_disable() const { cvk_ctx_->ops->parallel_disable(cvk_ctx_); }
  void set_layer_id(uint16_t layer_id) const {
    cvk_ctx_->ops->set_layer_id(cvk_ctx_, layer_id);
  }

public:
  // ####################################################
  // cvikernel adapter
  // ####################################################

  //
  // tdma kernel api, support i8/u8/bf16
  //
  void tdma_l2l_tensor_copy(cvk_tdma_l2l_tensor_copy_param_t *param) const {
    // compatible for i8/u8/bf16
    cvk_ctx_->ops->tdma_l2l_bf16_tensor_copy(cvk_ctx_, param);
  }

  void tdma_l2l_tensor_lrn_shift(cvk_tdma_l2l_tensor_lrn_shift_param_t *param) const {
    cvk_ctx_->ops->tdma_l2l_tensor_lrn_shift(cvk_ctx_, param);
  }

  void tdma_l2g_tensor_copy(cvk_tdma_l2g_tensor_copy_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_bf16_tensor_copy(cvk_ctx_, param);
  }

  void tdma_l2g_tensor_copy_nc_transposed(
      cvk_tdma_l2g_tensor_copy_nc_transposed_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_bf16_tensor_copy_nc_transposed(cvk_ctx_, param);
  }

  void tdma_l2g_tensor_copy_cw_transposed(
      cvk_tdma_l2g_tensor_copy_cw_transposed_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_bf16_tensor_copy_cw_transposed(cvk_ctx_, param);
  }

  void
  tdma_l2g_tensor_copy_compressed(cvk_tdma_l2g_tensor_copy_compressed_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_tensor_copy_compressed(cvk_ctx_, param);
  }

  void tdma_l2g_tensor_fill_constant(cvk_tdma_l2g_tensor_fill_constant_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_tensor_fill_constant(cvk_ctx_, param);
  }

  void tdma_l2g_matrix_copy(cvk_tdma_l2g_matrix_copy_param_t *param) const {
    cvk_ctx_->ops->tdma_l2g_bf16_matrix_copy(cvk_ctx_, param);
  }

  void tdma_g2l_tensor_copy(cvk_tdma_g2l_tensor_copy_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_bf16_tensor_copy(cvk_ctx_, param);
  }

  void tdma_g2l_tensor_copy_nc_transposed(
      cvk_tdma_g2l_tensor_copy_nc_transposed_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_bf16_tensor_copy_nc_transposed(cvk_ctx_, param);
  }

  void
  tdma_g2l_tensor_copy_chw_rotated(cvk_tdma_g2l_tensor_copy_chw_rotated_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_tensor_copy_chw_rotated(cvk_ctx_, param);
  }

  void tdma_g2l_tensor_copy_decompressed(
      cvk_tdma_g2l_tensor_copy_decompressed_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_tensor_copy_decompressed(cvk_ctx_, param);
  }

  void tdma_g2l_tensor_fill_constant(cvk_tdma_g2l_tensor_fill_constant_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_bf16_tensor_fill_constant(cvk_ctx_, param);
  }

  void tdma_g2l_matrix_copy_decompressed(
      cvk_tdma_g2l_matrix_copy_decompressed_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_matrix_copy_decompressed(cvk_ctx_, param);
  }

  void tdma_g2l_matrix_copy(cvk_tdma_g2l_matrix_copy_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_bf16_matrix_copy(cvk_ctx_, param);
  }

  void tdma_g2l_matrix_copy_row_col_transposed(
      cvk_tdma_g2l_matrix_copy_row_col_transposed_param_t *param) const {
    cvk_ctx_->ops->tdma_g2l_matrix_copy_row_col_transposed(cvk_ctx_, param);
  }

  void tdma_g2g_tensor_copy(cvk_tdma_g2g_tensor_copy_param_t *param) const {
    cvk_ctx_->ops->tdma_g2g_bf16_tensor_copy(cvk_ctx_, param);
  }

  //
  // tiu kernel api
  //
  // per tensor
  void tiu_mul(const cvk_tiu_mul_param_t *param) const {
    cvk_ctx_->ops->tiu_mul(cvk_ctx_, param);
  }

  void tiu_mac(const cvk_tiu_mac_param_t *param) const {
    cvk_ctx_->ops->tiu_mac(cvk_ctx_, param);
  }

  void tiu_add(const cvk_tiu_add_param_t *param) const {
    cvk_ctx_->ops->tiu_add(cvk_ctx_, param);
  }

  void tiu_sub(const cvk_tiu_sub_param_t *param) const {
    cvk_ctx_->ops->tiu_sub(cvk_ctx_, param);
  }

  void tiu_max(const cvk_tiu_max_param_t *param) const {
    cvk_ctx_->ops->tiu_max(cvk_ctx_, param);
  }

  void tiu_min(const cvk_tiu_min_param_t *param) const {
    cvk_ctx_->ops->tiu_min(cvk_ctx_, param);
  }

  void tiu_arith_shift(const cvk_tiu_arith_shift_param_t *param) const {
    cvk_ctx_->ops->tiu_arith_shift(cvk_ctx_, param);
  }

  void tiu_and_int8(const cvk_tiu_and_int8_param_t *param) const {
    cvk_ctx_->ops->tiu_and_int8(cvk_ctx_, param);
  }

  void tiu_and_int16(const cvk_tiu_and_int16_param_t *param) const {
    cvk_ctx_->ops->tiu_and_int16(cvk_ctx_, param);
  }

  void tiu_or_int8(const cvk_tiu_or_int8_param_t *param) const {
    cvk_ctx_->ops->tiu_or_int8(cvk_ctx_, param);
  }

  void tiu_or_int16(const cvk_tiu_or_int16_param_t *param) const {
    cvk_ctx_->ops->tiu_or_int16(cvk_ctx_, param);
  }

  void tiu_xor_int8(const cvk_tiu_xor_int8_param_t *param) const {
    cvk_ctx_->ops->tiu_xor_int8(cvk_ctx_, param);
  }

  void tiu_xor_int16(const cvk_tiu_xor_int16_param_t *param) const {
    cvk_ctx_->ops->tiu_xor_int16(cvk_ctx_, param);
  }

  void tiu_copy(const cvk_tiu_copy_param_t *param) const {
    cvk_ctx_->ops->tiu_copy(cvk_ctx_, param);
  }

  void tiu_lookup_table(const cvk_tiu_lookup_table_param_t *param) const {
    cvk_ctx_->ops->tiu_lookup_table(cvk_ctx_, param);
  }

  void tiu_bf16_lookup_interp_table(cvk_tiu_bf16_lookup_interp_table_param_t *param) const {
    cvk_ctx_->ops->tiu_bf16_lookup_interp_table(cvk_ctx_, param);
  }

  void tiu_pt_convolution(const cvk_tiu_pt_convolution_param_t *param) const {
    cvk_ctx_->ops->tiu_pt_convolution(cvk_ctx_, param);
  }

  void tiu_max_pooling(const cvk_tiu_max_pooling_param_t *param) const {
    cvk_ctx_->ops->tiu_max_pooling(cvk_ctx_, param);
  }

  void tiu_average_pooling(const cvk_tiu_average_pooling_param_t *param) const {
    cvk_ctx_->ops->tiu_average_pooling(cvk_ctx_, param);
  }

  void tiu_pt_depthwise_convolution(cvk_tiu_depthwise_pt_convolution_param_t *param) const {
    cvk_ctx_->ops->tiu_pt_depthwise_convolution(cvk_ctx_, param);
  }

  void tiu_matrix_multiplication(cvk_tiu_matrix_multiplication_param_t *param) const {
    cvk_ctx_->ops->tiu_matrix_multiplication(cvk_ctx_, param);
  }

  // per-channel
  void tiu_convolution(const cvk_tiu_convolution_param_t *param) const {
    cvk_ctx_->ops->tiu_convolution(cvk_ctx_, param);
  }

  void tiu_depthwise_convolution(cvk_tiu_depthwise_convolution_param_t *param) const {
    cvk_ctx_->ops->tiu_depthwise_convolution(cvk_ctx_, param);
  }

  void tiu_matrix_multiplication_qm(cvk_tiu_matrix_multiplication_qm_param_t *param) const {
    cvk_ctx_->ops->tiu_matrix_multiplication_qm(cvk_ctx_, param);
  }

  void tiu_mul_qm(const cvk_tiu_mul_qm_param_t *param) const {
    cvk_ctx_->ops->tiu_mul_qm(cvk_ctx_, param);
  }

  // helper
  cvk_tl_stride_t tl_default_stride(cvk_tl_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->tl_default_stride(cvk_ctx_, shape, fmt, eu_align);
  }

  cvk_tg_stride_t tg_default_stride(cvk_tg_shape_t shape, cvk_fmt_t fmt) const {
    return cvk_ctx_->ops->tg_default_stride(cvk_ctx_, shape, fmt);
  }

  cvk_ml_shape_t ml_shape_t1(uint32_t len, cvk_fmt_t fmt) const {
    return cvk_ctx_->ops->ml_shape_t1(cvk_ctx_, len, fmt);
  }

  cvk_ml_shape_t ml_default_shape(uint32_t row, uint32_t col, cvk_fmt_t fmt) const {
    return cvk_ctx_->ops->ml_default_shape(cvk_ctx_, row, col, fmt);
  }

  cvk_ml_stride_t ml_default_stride(cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->ml_default_stride(cvk_ctx_, shape, fmt, eu_align);
  }

  // lmem kernel
  cvk_tl_t *lmem_alloc_tensor(cvk_tl_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_alloc_tensor(cvk_ctx_, shape, fmt, eu_align);
  }

  void lmem_free_tensor(const cvk_tl_t *tl) const {
    cvk_ctx_->ops->lmem_free_tensor(cvk_ctx_, tl);
  }

  cvk_ml_t *lmem_alloc_matrix(cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_alloc_matrix(cvk_ctx_, shape, fmt, eu_align);
  }

  cvk_ml_t *lmem_alloc_ps32_matrix(cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_alloc_ps32_matrix(cvk_ctx_, shape, fmt, eu_align);
  }

  void lmem_free_matrix(const cvk_ml_t *ml) const {
    cvk_ctx_->ops->lmem_free_matrix(cvk_ctx_, ml);
  }

  void lmem_init_tensor(cvk_tl_t *tl, cvk_tl_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    cvk_ctx_->ops->lmem_init_tensor(cvk_ctx_, tl, shape, fmt, eu_align);
  }

  void lmem_init_matrix(cvk_ml_t *ml, cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    cvk_ctx_->ops->lmem_init_matrix(cvk_ctx_, ml, shape, fmt, eu_align);
  }

  uint32_t lmem_tensor_to_size(cvk_tl_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_tensor_to_size(cvk_ctx_, shape, fmt, eu_align);
  }

  uint32_t lmem_matrix_to_size(cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_matrix_to_size(cvk_ctx_, shape, fmt, eu_align);
  }

  uint32_t lmem_ps32_matrix_to_size(cvk_ml_shape_t shape, cvk_fmt_t fmt, int eu_align) const {
    return cvk_ctx_->ops->lmem_ps32_matrix_to_size(cvk_ctx_, shape, fmt, eu_align);
  }

  void bf16_table_shape(cvk_tl_shape_t *shape) const {
    cvk_ctx_->misc_ops->bf16_table_shape(cvk_ctx_, shape);
  }

  void gmem_init_tensor(cvk_tg_t *tg, cvk_tg_shape_t shape, cvk_fmt_t fmt) const {
    cvk_ctx_->ops->gmem_init_tensor(cvk_ctx_, tg, shape, fmt);
  }

  inline uint16_t convert_fp32_to_bf16(float fp32) const {
    return cvk_ctx_->misc_ops->float_to_bfloat16(cvk_ctx_, fp32);
  }

  //
  // tdma simple api
  //
  void tdma_load(cvk_tl_t *tlp, uint64_t ga_src, uint8_t do_transpose = 0) const;
  void tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src, cvk_tg_stride_t ts_stride,
                        bool do_transpose = false, bool do_decompress = false) const;
  void tdma_store(cvk_tl_t *tlp, uint64_t ga_dst, uint8_t do_transpose = 0) const;
  void tdma_store_stride(cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
                         bool do_transpose = false, bool do_compress = false) const;
  // matrix format
  void tdma_load(cvk_ml_t *tlp, uint64_t ga_src, uint8_t do_transpose = 0) const;
  void tdma_load_stride(cvk_ml_t *tlp, uint64_t ga_src, cvk_mg_stride_t ts_stride,
                        uint8_t do_transpose = 0) const;
  void tdma_store(cvk_ml_t *tlp, uint64_t ga_dst, uint8_t do_transpose = 0) const;
  void tdma_store_stride(cvk_ml_t *tlp, uint64_t ga_dst, cvk_mg_stride_t ts_stride,
                         uint8_t do_transpose = 0) const;
  void tdma_g2g_tensor_copy(uint64_t src_addr, cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride, cvk_fmt_t src_fmt,
                            uint64_t dst_addr, cvk_tg_shape_t dst_shape, cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt) const;

  uint32_t ga_cmpr_offset(int n, int c, int h, int w, int n_pos, int c_pos,
                          int h_pos, int c_step, int step_size) const;
  uint32_t addr_after_right_shift(int addr, uint32_t step, int c_str) const;
  uint32_t tl_cmpr_c_stride(int n, int c, int h, int w, cvk_fmt_t fmt) const;

public:
  // ####################################################
  // backend common api
  // ####################################################

  //
  // shape/size/fmt functions
  //
  void assert_support_fmt(cvk_fmt_t fmt) const;

  int bitsize_of_fmt(uint32_t fmt) const;

  inline int bytesize_of_fmt(cvk_fmt_t fmt) const {
    return bitsize_of_fmt(fmt) / 8; // byte
  }

  inline cvk_tl_shape_t tl_shape_t4(int n, int c, int h, int w) const {
    return {static_cast<uint32_t>(n), static_cast<uint32_t>(c),
            static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
  }

  inline cvk_tg_shape_t tg_shape_t4(int n, int c, int h, int w) const {
    return {static_cast<uint32_t>(n), static_cast<uint32_t>(c),
            static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
  }

  inline cvk_tg_stride_t tg_default_stride(int c, int h, int w,
                                           cvk_fmt_t fmt) const {
    return tg_default_stride(tg_shape_t4(1, c, h, w), fmt);
  }

  inline int tensor_size(int n, int c, int h, int w, cvk_fmt_t fmt) const {
    return n * c * h * w * bytesize_of_fmt(fmt);
  }

  inline uint32_t lmem_tensor_to_size(int n, int c, int h, int w,
                                      cvk_fmt_t fmt = CVK_FMT_I8,
                                      int eu_align = 1) const {
    return lmem_tensor_to_size(tl_shape_t4(n, c, h, w), fmt, eu_align);
  }

  inline uint32_t tiu_eu_num(cvk_fmt_t fmt) const {
    return cvi_chip_info_context(CVI_CHIP_EU_NUM) / (fmt == CVK_FMT_BF16 ? 2 : 1);
  }
  // tiu simple api
  void tiu_zeros(uint16_t layer_id, cvk_tl_t * tl_mem) const;

  //
  // tiling functions
  //

  typedef enum TilingDim {
    TilingAll = 0, // reshape(NxCxHxW) and tiling to [1, NPU_NUM, x, EU_NUM]
    TilingNHW,     // keep c, tiling n,h,w
    TilingNCH,     // keep w, tiling n,c,h
    TilingNH,     // keep cw, tiling n,h
    TilingNCHW,    // tiling n,c,h,w
  } tiling_mode_t;

  typedef struct tiling_info {
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
    int32_t pos_n;
    int32_t pos_c;
    int32_t pos_h;
    int32_t pos_w;
    uint64_t offset; // gmem offset
  } tiling_info_t;

  bool size_to_hw(int size, int&h, int&w) const;
  void tiling_packing(std::vector<tiling_info_t> &tiling_result, int n, int c,
                      int h, int w, cvk_fmt_t fmt, int blob_num = 1,
                      uint32_t reserved_lmem = 0,
                      tiling_mode_t mode = TilingNCHW) const;
  void tiling_packing(std::vector<tiling_info_t> &tiling_result,
                      cvk_tg_shape_t shape, cvk_fmt_t fmt, int blob_num = 1,
                      uint32_t reserved_lmem = 0,
                      tiling_mode_t mode = TilingNCHW) const;

  //
  // Hardware feature
  //
  const cvk_tl_shape_t &lut_table_shape(cvk_fmt_t fmt) const;

  bool has_cmd_pre_exe() const {
    return (cvk_ctx_->info.features & CVK_HWF_CMD_PRE_EXE) ? true : false;
  }

  inline uint32_t chan_quan_param_size(bool do_bias) const {
    // bias(4B) + multiplier(4B) + right_shift(1B)
    // multiplier(4B) + right_shift(1B)
    return do_bias ? 9 : 5;
  }

  // get TDMA base gmem selection from gaddr.
  uint8_t getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const;

  enum GlobalMemoryRegion {
    SHARED_MEMORY = 0,
    WEIGHT_MEMORY = 1,
    PRIVATE_MEMORY = 2,
    IO_MEMORY_0 = 3,
    IO_MEMORY_1 = 4,
    IO_MEMORY_2 = 5,
    IO_MEMORY_3 = 6,
    IO_MEMORY_4 = 7,
    MAX_GLOBAL_MEMORY_REGION = 8
  };

  enum QuantizeMode {
    INT8_PER_LAYER = 1, // 1880 mode, scale + rightshift
    INT8_PER_CHANNEL = 2,
    INT8_32_MULTIPLER = 3, // 1880v2,32bit multipliers(channel align) product tensor
    INT8_NOTSUPPORT // not support, should be assert it
  };

  void *get_cvk_ctx() const { return cvk_ctx_; }

private:
  void tiling_all(std::vector<tiling_info_t> &tiling_result, int64_t total,
                  cvk_fmt_t fmt, int blob_num, uint32_t lmem_size) const;
  void tiling_nchw(std::vector<tiling_info_t> &tiling_result, int n, int c,
                   int h, int w, cvk_fmt_t fmt, int blob_num,
                   uint32_t lmem_size, tiling_mode_t mode) const;

private:
  // Mapping between tdma base selection and global memory region.
  // Allowed to assign unique tdma base selection for each global memory region.
  uint8_t tdmaBaseSelects[MAX_GLOBAL_MEMORY_REGION];

  std::vector<uint8_t> cmdbuf_;

  std::vector<uint8_t> cvk_cmd_buf_;
  cvk_context_t *cvk_ctx_;
};

#endif /* _BM18XX_BACKEND_CONTEXT_H_ */
