/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */

#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "CviBackendContext"

CviBackendContext::CviBackendContext(const char *runchip) {
  // New kernel API
  cvk_reg_info_t req_info;
  strncpy(req_info.chip_ver_str, runchip, sizeof(req_info.chip_ver_str) - 1);
  req_info.cmdbuf_size = 0x10000000;
  req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
  cvk_ctx_ = cvikernel_register(&req_info);

  // Default mapping between tdma base selection
  // and global memory region.
  tdmaBaseSelects[NEURON_MEMORY] = 0;
  tdmaBaseSelects[WEIGHT_MEMORY] = 1;
  tdmaBaseSelects[INPUT_MEMORY] = 2;
  tdmaBaseSelects[OUTPUT_MEMORY] = 2;

  LLVM_DEBUG(llvm::errs() << "register " << runchip << " done\n";);
}

CviBackendContext::~CviBackendContext() {
  cvk_ctx_->ops->cleanup(cvk_ctx_);
}

void CviBackendContext::write_cmdbuf(const void* cmdbuf, uint32_t size) {
  cmdbuf_.resize(size);
  memcpy(&cmdbuf_[0], cmdbuf, size);
}

void CviBackendContext::read_cmdbuf(std::vector<uint8_t>& out_cmdbuf) {
  out_cmdbuf.assign(cmdbuf_.begin(), cmdbuf_.end());
}

void CviBackendContext::submit() {
  uint32_t size;
  uint8_t *cmdbuf = cvk_ctx_->ops->acquire_cmdbuf(cvk_ctx_, &size);
  write_cmdbuf(cmdbuf, size);
  cvk_ctx_->ops->reset(cvk_ctx_);
}

int CviBackendContext::cvi_chip_info_context(CVI_CHIP_INFO_E cvi_chip_info_e) const {
  if (cvi_chip_info_e == CVI_CHIP_VERSION)
    return cvk_ctx_->info.version;
  else if (cvi_chip_info_e == CVI_CHIP_NODE_SHIFT)
    return cvk_ctx_->info.node_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_NUM)
    return cvk_ctx_->info.npu_num;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_SHIFT)
    return cvk_ctx_->info.npu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_EU_NUM)
    return cvk_ctx_->info.eu_num;
  else if (cvi_chip_info_e == CVI_CHIP_EU_SHIFT)
    return cvk_ctx_->info.eu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_SIZE)
    return cvk_ctx_->info.lmem_size;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_BANK)
    return cvk_ctx_->info.lmem_banks;
  else
    assert(0);
}

uint8_t CviBackendContext::getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const {
  // we store memory region value in bits (40 ~ 41) of gaddr;
  uint32_t memoryRegion = ((((uint64_t)gaddr) >> 40) & 0x03);
  if (memoryRegion < MAX_GLOBAL_MEMORY_REGION) {
    return tdmaBaseSelects[memoryRegion];
  }
  return 0;
}

CviBackendContext *cvi_backend_create_context(const char *runchip) {
  CviBackendContext *ctx = new CviBackendContext(runchip);
  return ctx;
}

void cvi_backend_submit(CviBackendContext *ctx) {
  ctx->submit();
}

void cvi_backend_get_cmdbuf(CviBackendContext *ctx,
    std::vector<uint8_t> &cmdbuf) {
  ctx->read_cmdbuf(cmdbuf);
}

void cvi_backend_parallel_enable(CviBackendContext *ctx) {
  ctx->parallel_enable();
}

void cvi_backend_parallel_disable(CviBackendContext *ctx) {
  ctx->parallel_disable();
}

int cvi_backend_chip_context(CviBackendContext *ctx, CVI_CHIP_INFO_E cvi_chip_info_e) {
  return ctx->cvi_chip_info_context(cvi_chip_info_e);
}

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id) {
  ctx->set_layer_id(layer_id);
}

// tdma api

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

//
// Implement 1880 gdma_store, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_store(cvk_tl_t *tlp, uint64_t ga_dst,
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

void CviBackendContext::tdma_store_bf16(cvk_tl_t *tlp, uint64_t ga_dst,
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
void CviBackendContext::tdma_load_stride(cvk_ml_t *tlp, uint64_t ga_src,
                                         cvk_mg_stride_t ts_stride,
                                         uint8_t do_transpose) const {
  assert(tlp != nullptr);

  // Global memory from reshaped local memory
  cvk_mg_t ts_data = {0};
  ts_data.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.start_address = ga_src;
  ts_data.fmt = tlp->fmt;
  ts_data.stride = ts_stride;

  if (do_transpose) {
    ts_data.shape = {tlp->shape.col,
                     tlp->shape.n}; // Explicit transpose shape !!!

    cvk_tdma_g2l_matrix_copy_row_col_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;

    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "tdma_load_stride(matrix): src (%d, %d), dst(n=%d, c=%d, "
                   "w=%d,col= %d)\n",
                   p1.src->shape.row, p1.src->shape.col, p1.dst->shape.n,
                   p1.dst->shape.c, p1.dst->shape.w, p1.dst->shape.col));

    this->tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    ts_data.shape = {tlp->shape.n, tlp->shape.col};

    cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_matrix_copy(&p1);
  }
}

void CviBackendContext::tdma_load_stride_bf16(cvk_ml_t *tlp, uint64_t ga_src,
                                              cvk_mg_stride_t ts_stride,
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
void CviBackendContext::tdma_load(cvk_ml_t *tlp, uint64_t ga_src,
                                  uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_mg_t ts_data = {0};
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  tdma_load_stride(tlp, ga_src, ts_data.stride);
}

void CviBackendContext::tdma_load_bf16(cvk_ml_t *tlp, uint64_t ga_src,
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
void CviBackendContext::tdma_store(cvk_ml_t *tlp, uint64_t ga_dst,
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

void CviBackendContext::tdma_store_bf16(cvk_ml_t *tlp, uint64_t ga_dst,
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
void CviBackendContext::tdma_store_stride(cvk_ml_t *tlp, uint64_t ga_dst,
                                          cvk_mg_stride_t ts_stride,
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

void CviBackendContext::tdma_store_stride_bf16(cvk_ml_t *tlp, uint64_t ga_dst,
                                               cvk_mg_stride_t ts_stride,
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

void CviBackendContext::tdma_g2g_tensor_copy(
    uint64_t src_addr, cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    cvk_fmt_t src_fmt, uint64_t dst_addr, cvk_tg_shape_t dst_shape,
    cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt, cvk_fmt_t g2g_fmt) const {
  cvk_tg_t src = {0};
  src.start_address = src_addr;
  src.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(src_addr);
  src.fmt = src_fmt;
  src.shape = src_shape;
  src.stride = src_stride;
  src.int8_rnd_mode = 0;

  cvk_tg_t dst = {0};
  dst.start_address = dst_addr;
  dst.base_reg_index = this->getTdmaBaseSelectIndexFromGaddr(dst_addr);
  dst.fmt = dst_fmt;
  dst.shape = dst_shape;
  dst.stride = dst_stride;
  dst.int8_rnd_mode = 0;
  cvk_tdma_g2g_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = &dst;
  if (g2g_fmt == CVK_FMT_BF16) {
    this->tdma_g2g_bf16_tensor_copy(&p);
  } else if (g2g_fmt == CVK_FMT_I8 || g2g_fmt == CVK_FMT_U8) {
    this->tdma_g2g_tensor_copy(&p);
  }
}

void CviBackendContext::tdma_g2g_tensor_copy(
    uint64_t src_addr, cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    uint64_t dst_addr, cvk_tg_shape_t dst_shape, cvk_tg_stride_t dst_stride,
    cvk_fmt_t g2g_fmt) const {
  this->tdma_g2g_tensor_copy(src_addr, src_shape, src_stride, g2g_fmt, dst_addr,
                             dst_shape, dst_stride, g2g_fmt, g2g_fmt);
}