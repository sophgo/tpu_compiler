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
    cvk_tg_stride_t ts_stride, bool do_transpose, bool do_decompress) const {
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
  } else if (do_decompress) {
    cvk_cmpr_tg_t cmpr_ts_data = {0};
    cmpr_ts_data.t = ts_data;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &cmpr_ts_data;
    param.dst = tlp;
    this->tdma_g2l_tensor_copy_decompressed(&param);
  } else {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy(&p1);
  }
}

void CviBackendContext::tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src,
                                         cvk_tg_stride_t ts_stride,
                                         bool do_transpose) const {
  this->tdma_load_stride(tlp, ga_src, ts_stride, do_transpose,
                         /*do_decompress=*/false);
}

void CviBackendContext::tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src,
                                         cvk_tg_stride_t ts_stride) const {
  this->tdma_load_stride(tlp, ga_src, ts_stride, /*do_transpose=*/false,
                         /*do_decompress=*/false);
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
    bool do_transpose, bool do_compress) const {
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
  } else if (do_compress) {
    cvk_cmpr_tg_t cmpr_dst = {0};
    cmpr_dst.t = ts_data;

    cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
    param.src = tlp;
    param.dst = &cmpr_dst;
    this->tdma_l2g_tensor_copy_compressed(&param);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy(&p1);
  }
}

void CviBackendContext::tdma_store_stride(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
    bool do_transpose) const {
  tdma_store_stride(tlp, ga_dst, ts_stride, do_transpose,
                    /*do_compress=*/false);
}

void CviBackendContext::tdma_store_stride(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride) const {
  tdma_store_stride(tlp, ga_dst, ts_stride, /*do_transpose=*/false,
                    /*do_compress=*/false);
}

void CviBackendContext::tdma_store_stride_bf16(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
    bool do_transpose, bool do_compress) const {
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
  } else if (do_compress) {
    assert(0 && "bf16 tdma store does not suppport compress yet");
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_bf16_tensor_copy(&p1);
  }
}

void CviBackendContext::tdma_store_stride_bf16(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride,
    bool do_transpose) const {
  this->tdma_store_stride_bf16(tlp, ga_dst, ts_stride, do_transpose,
                              /*do_compress=*/false);
}

void CviBackendContext::tdma_store_stride_bf16(
    cvk_tl_t *tlp, uint64_t ga_dst, cvk_tg_stride_t ts_stride) const {
  this->tdma_store_stride_bf16(tlp, ga_dst, ts_stride, /*do_transpose=*/false,
                              /*do_compress=*/false);
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
