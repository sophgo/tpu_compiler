/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: dma_legacy_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

#define DEBUG_TYPE "bmnet_bm1880v2_dma_legacy"


void CviBackendContext::tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src,
                                         cvk_tg_stride_t ts_stride,
                                         uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  //
  // Constraint:
  //   assert_tl_tg_same_size()
  //   Global_N == Local_N
  //
  // 1. Global channel != local channel
  //    Eg.
  //     alexnet: src (, 256, 1, 1), dst (2, 128, 1, 1)
  //
  // 2. Global shape != local shape
  //    Eg.
  //     alexnet conv5 relu
  //     src (, 384, 13, 13), dst (1, 384, 8, 13)

  // tensor in system memory
  // Global shape use local shape
  cvk_tg_t ts_data;
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_g2l_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy(&p1);
  }
}

//
void CviBackendContext::tdma_load_stride_bf16(
    cvk_tl_t *tlp, uint64_t ga_src, cvk_tg_stride_t ts_stride,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  cvk_tg_t ts_data;
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_g2l_tensor_copy_nc_transposed_param_t p1 = {0};
    ts_data.shape = {tlp->shape.c, tlp->shape.n, tlp->shape.h, tlp->shape.w};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_bf16_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_bf16_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_load, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void CviBackendContext::tdma_load(
    cvk_tl_t *tlp, uint64_t ga_src, uint8_t do_transpose)
    const {
  assert(tlp != nullptr);

  cvk_tg_t ts_data = {0};
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = this->tg_default_stride(ts_data.shape, ts_data.fmt);
  tdma_load_stride(tlp, ga_src, ts_data.stride);
}

void CviBackendContext::tdma_load_bf16(
    cvk_tl_t *tlp, uint64_t ga_src,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_tg_t ts_data = {0};
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = this->tg_default_stride(ts_data.shape, ts_data.fmt);
  this->tdma_load_stride_bf16(tlp, ga_src, ts_data.stride);
}

//
// Implement 1880 gdma_store_stride, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void CviBackendContext::tdma_store_stride(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy(&p1);
  }
}

void CviBackendContext::tdma_store_stride_bf16(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_tg_t ts_data;
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_bf16_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_bf16_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_store, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void CviBackendContext::tdma_store(
    cvk_tl_t *tlp, uint64_t ga_dst,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = this->tg_default_stride(ts_data.shape, ts_data.fmt);

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy(&p1);
  }
}

void CviBackendContext::tdma_store_bf16(
    cvk_tl_t *tlp, uint64_t ga_dst,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_tg_t ts_data;
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = this->tg_default_stride(ts_data.shape, ts_data.fmt);

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_bf16_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_bf16_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_tg_copy, tensor format
//   No split
//
//   Condition:
//     size <= local LANE size
//
bool CviBackendContext::_tdma_tg_copy_no_split(cvk_tg_t *tg_dst,
                                                    cvk_tg_t *tg_src) const {
  cvk_fmt_t cal_fmt = CVK_FMT_I8;

  if (is_bf16(tg_src->fmt) || is_bf16(tg_dst->fmt)) {
    // TODO: more general
    cal_fmt = CVK_FMT_BF16;
  }

  cvk_tl_t tl_data;
  tl_data.start_address = 0;  // Start of LMEM
  tl_data.fmt = cal_fmt;       // don't care
  tl_data.shape = {tg_src->shape.n, tg_src->shape.c, tg_src->shape.h, tg_src->shape.w};
  tl_data.stride = this->tl_default_stride(tl_data.shape, cal_fmt, /*eu_align=*/0);

  // Check required lane size
  uint32_t required_lane_size = tl_data.shape.n * tl_data.stride.n;
  if (required_lane_size > (uint32_t)cvi_chip_info_context(CVI_CHIP_LMEM_SIZE)) {
    return false;
  }

  // G2L
  cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
  p1.src = tg_src;
  p1.dst = &tl_data;
  this->_tdma_g2l_tensor_copy(&p1);

  // L2G
  cvk_tdma_l2g_tensor_copy_param_t p2 = {0};
  p2.src = &tl_data;
  p2.dst = tg_dst;
  this->_tdma_l2g_tensor_copy(&p2);

  return true;
}

//
// Implement 1880 gdma_tg_copy, tensor format
//   Split along N, H axis.
//
//   Condition:
//     src shape == dst shape.
//     H*W > one lane size.
//     C close to lane number.
//
bool CviBackendContext::_tdma_tg_copy_split_nh(cvk_tg_t *tg_dst,
                                                    cvk_tg_t *tg_src) const {
  uint64_t input_gaddr = tg_src->start_address;
  uint32_t input_n = tg_src->shape.n;
  uint32_t input_c = tg_src->shape.c;
  uint32_t input_h = tg_src->shape.h;
  uint32_t input_w = tg_src->shape.w;
  uint64_t output_gaddr = tg_dst->start_address;
  uint32_t output_n = tg_dst->shape.n;
  uint32_t output_c = tg_dst->shape.c;
  uint32_t output_h = tg_dst->shape.h;
  uint32_t output_w = tg_dst->shape.w;

  LLVM_DEBUG(llvm::errs() << llvm::format("_tg_copy_split_nh:\n"
                                        "    src shape (%d, %d, %d, %d), stride (%d, %d, %d)\n"
                                        "    dst shape (%d, %d, %d, %d), stride (%d, %d, %d)\n",
                                        tg_src->shape.n, tg_src->shape.c, tg_src->shape.h,
                                        tg_src->shape.w, tg_src->stride.n, tg_src->stride.c,
                                        tg_src->stride.h, tg_dst->shape.n, tg_dst->shape.c,
                                        tg_dst->shape.h, tg_dst->shape.w, tg_dst->stride.n,
                                        tg_dst->stride.c, tg_dst->stride.h));

  // Shape must be equal
  if (input_n != output_n || input_c != output_c || input_h != output_h || input_w != output_w) {
    LLVM_DEBUG(llvm::errs() << "<= _tg_copy_split_nh, shape not equal" << "\n");
    return false;
  }

  // Split along H based on LMEM size
  uint32_t lmem_size = this->cvi_chip_info_context(CVI_CHIP_LMEM_SIZE) * this->cvi_chip_info_context(CVI_CHIP_LANE_NUM);
  uint32_t chw_size = input_c * input_h * input_w;
  uint32_t max_tiled_h = (chw_size <= lmem_size) ? input_h : lmem_size / input_c / input_w;

  // Find max tiled_h to fit local memory
  do {
    cvk_fmt_t cal_fmt = CVK_FMT_I8;

    if (is_bf16(tg_src->fmt) || is_bf16(tg_dst->fmt)) {
      // TODO: more general
      cal_fmt = CVK_FMT_BF16;
    }

    cvk_tl_shape_t shape = {1, input_c, max_tiled_h, input_w};
    cvk_tl_stride_t stride =
      this->tl_default_stride(shape, cal_fmt, /*eu_align=*/0);
    uint32_t required_lane_size = stride.n;

    if (required_lane_size <= (uint32_t)cvi_chip_info_context(CVI_CHIP_LMEM_SIZE)) {
      break;
    }
  } while (--max_tiled_h);

  // Something wrong !
  if (!max_tiled_h) {
    LLVM_DEBUG(llvm::errs() << "<= _tg_copy_split_nh, unable to split h" << "\n");
    return false;
  }

  // Split along N axis
  for (uint32_t i = 0; i < input_n; i++) {
    // Split along H axis
    for (uint32_t h_offset = 0; h_offset < input_h; h_offset += max_tiled_h) {
      uint32_t tiled_h = (h_offset + max_tiled_h) <= input_h ? max_tiled_h : (input_h - h_offset);

      // Calculate offset with stride(original shape), position (ni, 0, h_offset, 0)
      uint64_t input_offset = i * tg_src->stride.n + h_offset * tg_src->stride.h;
      uint64_t output_offset = i * tg_dst->stride.n + h_offset * tg_dst->stride.h;

      LLVM_DEBUG(
          llvm::errs() << llvm::format("    [%d] h_offset %d, src tiled shape(1, %d, %d, %d)\n", i,
                                    h_offset, input_c, tiled_h, input_w));

      // G2L tensor copy
      {
        // Tiled shape and original stride
        cvk_tg_t src;
        src.base_reg_index = tg_src->base_reg_index;
        src.start_address = input_gaddr + input_offset;
        src.fmt = tg_src->fmt;  // don't care
        src.shape = {1, input_c, tiled_h, input_w};
        src.stride = tg_src->stride;

        // Follow source shape
        cvk_tl_t dst;
        dst.start_address = 0;  // start of lmem
        dst.fmt = tg_dst->fmt;
        dst.shape = {1, input_c, tiled_h, input_w};
        dst.stride = this->tl_default_stride(dst.shape, dst.fmt, /*eu_align=*/0);

        cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
        p1.src = &src;
        p1.dst = &dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] G2L tensor copy: h_offset %d, tiled_h %d\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                        i, h_offset, tiled_h, p1.src->start_address, p1.src->shape.n,
                        p1.src->shape.c, p1.src->shape.h, p1.src->shape.w, p1.src->stride.n,
                        p1.src->stride.c, p1.src->stride.h, p1.dst->start_address, p1.dst->shape.n,
                        p1.dst->shape.c, p1.dst->shape.h, p1.dst->shape.w, p1.dst->stride.n,
                        p1.dst->stride.c, p1.dst->stride.h, p1.dst->stride.w));

        this->_tdma_g2l_tensor_copy(&p1);
      }

      // L2G tensor copy
      {
        cvk_tl_t src;
        src.start_address = 0;
        src.fmt = tg_src->fmt;  // don't care
        src.shape = {1, input_c, tiled_h, input_w};
        src.stride = this->tl_default_stride(src.shape, src.fmt, /*eu_align=*/0);

        // tilted shape and original stride
        cvk_tg_t dst;
        dst.base_reg_index = tg_dst->base_reg_index;
        dst.start_address = output_gaddr + output_offset;
        dst.fmt = tg_dst->fmt;  // don't care
        dst.shape = {1, output_c, tiled_h, output_w};
        dst.stride = this->tg_default_stride(dst.shape, dst.fmt);

        cvk_tdma_l2g_tensor_copy_param_t p2 = {0};
        p2.src = &src;
        p2.dst = &dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] L2G: h_offset %d, tiled_h %d\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, h_offset, tiled_h, p2.src->start_address, p2.src->shape.n,
                        p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                        p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                        p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                        p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h));

        this->_tdma_l2g_tensor_copy(&p2);
      }

    }  // for (uint32_t h_offset = 0; h_offset < input_h; h_offset+=max_tiled_h)
  }    // for (uint32_t i = 0; i < input_n; i++

  LLVM_DEBUG(llvm::errs() << "<= _tg_copy_split_nh" << "\n");
  return true;
}

//
// Implement 1880 gdma_tg_copy, tensor format
//
void CviBackendContext::_tdma_tg_copy(cvk_tg_t *dst,
                                           cvk_tg_t *src) const {
  bool result = false;

  // Case 1: total size <= TPU LMEM size, no need to split N or C
  result = this->_tdma_tg_copy_no_split(dst, src);
  if (result) {
    return;
  }

  // Case 2: split along N, H axis
  result = this->_tdma_tg_copy_split_nh(dst, src);
  if (result) {
    return;
  }

  // Shoul not reach here
  LLVM_DEBUG(llvm::errs() << "tdma_tg_copy: fail to split";);
}

//
// Implement 1880 gdma_tg_copy, tensor format
//
void CviBackendContext::tdma_tg_copy(
    cvk_tg_t *dst, cvk_tg_t *src,
    uint8_t do_transpose) const {
  assert(dst != nullptr && src != nullptr);

  dst->base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(dst->start_address);
  src->base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(src->start_address);

  if (do_transpose) {
    // TDMA does not support G2G nc transpose

    // DO NOT know how to tile
    assert((uint32_t)(src->shape.n * src->shape.c * src->shape.h * src->shape.w) <=
           (uint32_t)(cvi_chip_info_context(CVI_CHIP_LMEM_SIZE) * cvi_chip_info_context(CVI_CHIP_LANE_NUM)));

    // 1. G2L tensor copy
    cvk_tl_t tl_data;
    tl_data.start_address = 0;
    tl_data.fmt = src->fmt;
    tl_data.shape = {src->shape.n, src->shape.c, src->shape.h, src->shape.w};
    tl_data.stride = this->tl_default_stride(tl_data.shape, tl_data.fmt, /*eu_align=*/0);

    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = src;
    p1.dst = &tl_data;
    this->_tdma_g2l_tensor_copy(&p1);

    // 2. L2G nc transpose
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p2 = {0};
    p2.src = &tl_data;
    p2.dst = dst;
    this->_tdma_l2g_tensor_copy_nc_transposed(&p2);
  } else {
    if (0) {
      cvk_tdma_g2g_tensor_copy_param_t p = {0};
      p.src = src;
      p.dst = dst;
      this->tdma_g2g_tensor_copy(&p);
    } else {
      this->_tdma_tg_copy(dst, src);
    }
  }
}

//
// Implement 1880 gdma_load_stride, matrix format
//
void CviBackendContext::tdma_load_stride(
    cvk_ml_t *tlp, uint64_t ga_src, cvk_mg_stride_t ts_stride,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // Global memory from reshaped local memory
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.start_address = ga_src;
  ts_data.fmt = tlp->fmt;
  ts_data.stride = ts_stride;

  if (do_transpose) {
    ts_data.shape = {tlp->shape.col, tlp->shape.n};  // Explicit transpose shape !!!

    cvk_tdma_g2l_matrix_copy_row_col_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;

    LLVM_DEBUG(llvm::errs() << llvm::format(
                    "tdma_load_stride(matrix): src (%d, %d), dst(n=%d, c=%d, w=%d,col= %d)\n",
                    p1.src->shape.row, p1.src->shape.col, p1.dst->shape.n, p1.dst->shape.c,
                    p1.dst->shape.w, p1.dst->shape.col));

    this->tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    ts_data.shape = {tlp->shape.n, tlp->shape.col};

    cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_matrix_copy(&p1);
  }
}

void CviBackendContext::tdma_load_stride_bf16(
    cvk_ml_t *tlp, uint64_t ga_src, cvk_mg_stride_t ts_stride,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // Global memory from reshaped local memory
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.start_address = ga_src;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  // BM1880v2 tdma does not support transposed matrix load
  assert(!do_transpose);

  cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
  p1.src = &ts_data;
  p1.dst = tlp;
  this->tdma_g2l_bf16_matrix_copy(&p1);
}

//
// Implement 1880 gdma_load, matrix format
//
void CviBackendContext::tdma_load(
    cvk_ml_t *tlp, uint64_t ga_src,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_mg_t ts_data = {0};
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  tdma_load_stride(tlp, ga_src, ts_data.stride);
}

void CviBackendContext::tdma_load_bf16(
    cvk_ml_t *tlp, uint64_t ga_src,
    uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_mg_t ts_data = {0};
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  tdma_load_stride_bf16(tlp, ga_src, ts_data.stride);
}

//
// Implement 1880 gdma_store, matrix format
//
void CviBackendContext::tdma_store(
    cvk_ml_t *tlp, uint64_t ga_dst,
    uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_matrix_copy(&p1);
}

void CviBackendContext::tdma_store_bf16(
    cvk_ml_t *tlp, uint64_t ga_dst,
    uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_bf16_matrix_copy(&p1);
}

//
// Implement 1880 gdma_store_stride, matrix format
//
void CviBackendContext::tdma_store_stride(
    cvk_ml_t *tlp, uint64_t ga_dst, cvk_mg_stride_t ts_stride,
    uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_matrix_copy(&p1);
}

void CviBackendContext::tdma_store_stride_bf16(
    cvk_ml_t *tlp, uint64_t ga_dst, cvk_mg_stride_t ts_stride,
    uint8_t do_transpose) const {

  assert(do_transpose == false);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_bf16_matrix_copy(&p1);
}

//}  // namespace bmnet
