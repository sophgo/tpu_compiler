/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */

#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "CviBackendContext"

// use local
#undef NPU_NUM
#undef EU_NUM
#undef LOCAL_MEM_SIZE
#undef LOCAL_MEM_BANKS
#define NPU_NUM cvi_chip_info_context(CVI_CHIP_LANE_NUM)
#define EU_NUM cvi_chip_info_context(CVI_CHIP_EU_NUM)
#define LOCAL_MEM_SIZE cvi_chip_info_context(CVI_CHIP_LMEM_SIZE)
#define LOCAL_MEM_BANKS cvi_chip_info_context(CVI_CHIP_LMEM_BANK)

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

CviBackendContext::~CviBackendContext() { cvk_ctx_->ops->cleanup(cvk_ctx_); }

void CviBackendContext::write_cmdbuf(const void *cmdbuf, uint32_t size) {
  cmdbuf_.resize(size);
  memcpy(&cmdbuf_[0], cmdbuf, size);
}

void CviBackendContext::read_cmdbuf(std::vector<uint8_t> &out_cmdbuf) {
  out_cmdbuf.assign(cmdbuf_.begin(), cmdbuf_.end());
}

void CviBackendContext::submit() {
  uint32_t size;
  uint8_t *cmdbuf = cvk_ctx_->ops->acquire_cmdbuf(cvk_ctx_, &size);
  write_cmdbuf(cmdbuf, size);
  cvk_ctx_->ops->reset(cvk_ctx_);
}

int CviBackendContext::cvi_chip_info_context(
    CVI_CHIP_INFO_E cvi_chip_info_e) const {
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

uint8_t
CviBackendContext::getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const {
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

void cvi_backend_submit(CviBackendContext *ctx) { ctx->submit(); }

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

int cvi_backend_chip_context(CviBackendContext *ctx,
                             CVI_CHIP_INFO_E cvi_chip_info_e) {
  return ctx->cvi_chip_info_context(cvi_chip_info_e);
}

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id) {
  ctx->set_layer_id(layer_id);
}

// tdma api

void CviBackendContext::tdma_load_stride(cvk_tl_t *tlp, uint64_t ga_src,
                                         cvk_tg_stride_t ts_stride,
                                         bool do_transpose,
                                         bool do_decompress) const {
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
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_src);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_g2l_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_tensor_copy_nc_transposed(&p1);
  } else if (do_decompress) {
    cvk_cmpr_tg_t cmpr_ts_data = {0};
    cmpr_ts_data.t = ts_data;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &cmpr_ts_data;
    param.dst = tlp;
    tdma_g2l_tensor_copy_decompressed(&param);
  } else {
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_load, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_load(cvk_tl_t *tlp, uint64_t ga_src,
                                  uint8_t do_transpose) const {
  assert(tlp != nullptr);

  cvk_tg_t ts_data = {0};
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = tg_default_stride(ts_data.shape, ts_data.fmt);
  tdma_load_stride(tlp, ga_src, ts_data.stride);
}

//
// Implement 1880 gdma_store_stride, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride
//   info.
//
void CviBackendContext::tdma_store_stride(cvk_tl_t *tlp, uint64_t ga_dst,
                                          cvk_tg_stride_t ts_stride,
                                          bool do_transpose,
                                          bool do_compress) const {
  assert(tlp != nullptr);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else if (do_compress) {
    assert(ts_data.fmt != CVK_FMT_BF16 &&
           "bf16 tdma store does not suppport compress yet");
    cvk_cmpr_tg_t cmpr_dst = {0};
    cmpr_dst.t = ts_data;

    cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
    param.src = tlp;
    param.dst = &cmpr_dst;
    tdma_l2g_tensor_copy_compressed(&param);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy(&p1);
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
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = tg_default_stride(ts_data.shape, ts_data.fmt);

  if (do_transpose) {
    cvk_tdma_l2g_tensor_copy_nc_transposed_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tlp;
    p1.dst = &ts_data;
    tdma_l2g_tensor_copy(&p1);
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
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_src);
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

    tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    ts_data.shape = {tlp->shape.n, tlp->shape.col};

    cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
    p1.src = &ts_data;
    p1.dst = tlp;
    tdma_g2l_matrix_copy(&p1);
  }
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
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  tdma_l2g_matrix_copy(&p1);
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
  ts_data.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_dst);
  ts_data.start_address = ga_dst;
  ts_data.fmt = tlp->fmt;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
  p1.src = tlp;
  p1.dst = &ts_data;
  tdma_l2g_matrix_copy(&p1);
}

void CviBackendContext::tdma_g2g_tensor_copy(
    uint64_t src_addr, cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    cvk_fmt_t src_fmt, uint64_t dst_addr, cvk_tg_shape_t dst_shape,
    cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt) const {
  cvk_tg_t src = {0};
  src.start_address = src_addr;
  src.base_reg_index = getTdmaBaseSelectIndexFromGaddr(src_addr);
  src.fmt = src_fmt;
  src.shape = src_shape;
  src.stride = src_stride;
  src.int8_rnd_mode = 0;

  cvk_tg_t dst = {0};
  dst.start_address = dst_addr;
  dst.base_reg_index = getTdmaBaseSelectIndexFromGaddr(dst_addr);
  dst.fmt = dst_fmt;
  dst.shape = dst_shape;
  dst.stride = dst_stride;
  dst.int8_rnd_mode = 0;
  cvk_tdma_g2g_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = &dst;
  tdma_g2g_tensor_copy(&p);
}

int CviBackendContext::bitsize_of_fmt(uint32_t fmt) const {
  switch (fmt) {
  case CVK_FMT_F32:
  case CVK_FMT_I32:
    return 32;
  case CVK_FMT_BF16:
  case CVK_FMT_F16:
  case CVK_FMT_I16:
  case CVK_FMT_U16:
    return 16;
  case CVK_FMT_I8:
  case CVK_FMT_U8:
    return 8;
  case CVK_FMT_I4:
    return 4;
  case CVK_FMT_I2:
    return 2;
  case CVK_FMT_I1:
    return 1;
  default:
    assert(0);
    return -1;
  }
}

const cvk_tl_shape_t &CviBackendContext::lut_table_shape(cvk_fmt_t fmt) const {
  static const cvk_tl_shape_t table_fixed = tl_shape_t4(1, NPU_NUM, 16, 16);
  static const cvk_tl_shape_t table_bf16 = tl_shape_t4(1, NPU_NUM, 32, 8);
  assert_support_fmt(fmt);
  if (fmt == CVK_FMT_BF16) {
    return table_bf16;
  }
  return table_fixed;
}

void CviBackendContext::tiling_packing(
    int require_shape, int coeff_lane_shape, int blob_num, cvk_fmt_t fmt,
    std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> *tiling_info,
    enum TilingDim tiling_along, cvk_tg_shape_t *shape) const {

  assert_support_fmt(fmt);
  assert(blob_num > 0 && "blob number should >= 1(contain itself)");

  int data_type_size = bytesize_of_fmt(fmt); // byte
  int coeff_lane_size = coeff_lane_shape * data_type_size;
  gaddr_t gaddr_offset = 0;

  if (tiling_along == TilingDimNH) {
    int input_n = shape->n;
    int input_c = shape->c;
    int input_h = shape->h;
    int input_w = shape->w;
    int nsecs = 1, hsecs = 1;

    uint32_t global_Nstride =
        static_cast<uint32_t>(input_c) * input_h * input_w;

    if (fmt == CVK_FMT_BF16) {
      blob_num *= 2; // bf16 takes twice size than int8
    }

    split_nh(input_n, input_c, input_h, input_w, blob_num, coeff_lane_shape,
             &nsecs, &hsecs);

    int nslice = input_n / nsecs;
    int hslice = input_h / hsecs;
    int nresidual = input_n - nslice * nsecs;
    int hresidual = input_h - hslice * hsecs;

    for (int nidx = 0, nstart = 0; nidx < nsecs; nidx++) {
      int sec_len_n = nslice + (nidx < nresidual);
      for (int hidx = 0, hstart = 0; hidx < hsecs; hidx++) {
        int sec_len_h = hslice + (hidx < hresidual);
        uint64_t offset =
            (nstart * global_Nstride + hstart * input_w) * data_type_size;

        tiling_info->push_back(std::make_pair(
            tl_shape_t4(sec_len_n, input_c, sec_len_h, input_w), offset));

        hstart += sec_len_h;
      }
      nstart += sec_len_n;
    }
  } else if (tiling_along == TilingDimAll) {
    // available size per lane = LOCAL_MEM_SIZE - coeff_lane_size
    int height = require_shape / (NPU_NUM * EU_NUM);
    if (require_shape + coeff_lane_shape * NPU_NUM >= (NPU_NUM * EU_NUM) &&
        height) {
      do {
        // Find height
        height = require_shape / (NPU_NUM * EU_NUM);
        do {
          cvk_tl_shape_t tmp_shape = tl_shape_t4(1, 1, height, EU_NUM);
          int required_size =
              blob_num * lmem_tensor_to_size(tmp_shape, fmt, /*eu_align=*/1);

          if (required_size <= LOCAL_MEM_SIZE - coeff_lane_size)
            break;
        } while (--height);

        int step_shape = height * NPU_NUM * EU_NUM;

        LLVM_DEBUG(
            llvm::errs() << llvm::format(
                "    step_shape %d, require_shape %d, gaddr_offset 0x%lx\n",
                step_shape, require_shape, gaddr_offset););

        tiling_info->push_back(std::make_pair(
            tl_shape_t4(1, NPU_NUM, height, EU_NUM), gaddr_offset));

        // Update step
        require_shape -= step_shape;
        gaddr_offset += step_shape * data_type_size;
      } while (require_shape >= ((NPU_NUM * EU_NUM)));
    }

    // Use one lane to handle remaining
    if (require_shape) {
      int step_shape = require_shape;

      tiling_info->push_back(
          std::make_pair(tl_shape_t4(1, 1, 1, step_shape), gaddr_offset));

      LLVM_DEBUG(
          llvm::errs() << llvm::format(
              "    (r)step_shape %d, require_shape %d, gaddr_offset 0x%lx\n",
              step_shape, require_shape, gaddr_offset));
      require_shape -= step_shape;
      gaddr_offset += step_shape * data_type_size;
    }
  }
  assert(tiling_info->size());
}

void CviBackendContext::split_nh(int n, int c, int h, int w, int blob_num,
                                 uint32_t reserved, int *n_slices,
                                 int *h_slices) const {
  *h_slices = 1;
  *n_slices = 1;
  uint32_t total_lmem_needs = blob_num * get_lmem_usage(n, c, h, w);

  LLVM_DEBUG(
      llvm::errs() << llvm::format("<%d,%d,%d,%d>, reserved:%u, total:%u\n", n,
                                   c, h, w, reserved, total_lmem_needs););

  if (total_lmem_needs + reserved <= (uint32_t)LOCAL_MEM_SIZE) {
    return;
  }

  // split h if lmem usage per image is larger than LOCAL_MEM_SIZE
  if (n == 1 || total_lmem_needs > (uint32_t)(LOCAL_MEM_SIZE * n)) {
    *n_slices = n;
    total_lmem_needs = total_lmem_needs / n;

    *h_slices = (total_lmem_needs + LOCAL_MEM_SIZE - 1) / LOCAL_MEM_SIZE;
    int h_units_per_slice = (h + *h_slices - 1) / *h_slices;
    LLVM_DEBUG(llvm::errs() << "h_units_per_slice is " << h_units_per_slice
                            << ", h_slices is " << *h_slices << "\n";);

    while (blob_num * get_lmem_usage(1, c, h_units_per_slice, w) + reserved >
               (uint32_t)LOCAL_MEM_SIZE ||
           h_units_per_slice > (4095 - 32)) {
      *h_slices += 1;
      h_units_per_slice = (h + *h_slices - 1) / *h_slices;

      LLVM_DEBUG(llvm::errs() << "h_units_per_slice is " << h_units_per_slice
                              << ", h_slices is " << *h_slices << "\n";);
    }
  } else { // split n if local memory can store more than on image
    *n_slices = (total_lmem_needs + LOCAL_MEM_SIZE - 1) / LOCAL_MEM_SIZE;
    int n_units_per_slice = (n + *n_slices - 1) / *n_slices;

    while (blob_num * get_lmem_usage(n_units_per_slice, c, h, w) + reserved >
           (uint32_t)LOCAL_MEM_SIZE) {
      *n_slices += 1;
      n_units_per_slice = (n + *n_slices - 1) / *n_slices;
    }
  }
}

void CviBackendContext::split_cnh(int n, int c, int h, int w, int blob_num,
                                  uint32_t reserved, int *c_slices,
                                  int *n_slices, int *h_slices) const {
  *c_slices = 1;
  *n_slices = 1;
  *h_slices = 1;
  uint32_t total_lmem_needs = blob_num * get_lmem_usage(n, c, h, w) + reserved;

  if (total_lmem_needs > (uint32_t)LOCAL_MEM_SIZE) {
    if (c > NPU_NUM) {
      int c_units_per_npu = (c + NPU_NUM - 1) / NPU_NUM;

      if (total_lmem_needs / c_units_per_npu <= LOCAL_MEM_SIZE - reserved) {
        int _c = (c + *c_slices - 1) / *c_slices;

        while (blob_num * get_lmem_usage(n, _c, h, w) + reserved >
               (uint32_t)LOCAL_MEM_SIZE) {
          *c_slices += 1;
          _c = (c + *c_slices - 1) / *c_slices;
        }
        return;
      }
    }
    split_nh(n, c, h, w, blob_num, reserved, n_slices, h_slices);
  }
}

int CviBackendContext::split(int blob_num, int count) const {
  int slice_num = 1;
  int W_param = EU_NUM;
  int C_param = (count + EU_NUM - 1) / EU_NUM;
  int aligned_csize = get_csize_local(1, W_param);
  int c_per_npu = (C_param + NPU_NUM - 1) / NPU_NUM;
  int local_mem_usage = c_per_npu * aligned_csize * blob_num;
  const int local_mem_size = LOCAL_MEM_SIZE;
  int proportion_mem_usage =
      (local_mem_usage + local_mem_size - 1) / local_mem_size;

  if (proportion_mem_usage == 1 && c_per_npu < 0x1000) {
    return slice_num;
  } else {
    slice_num = proportion_mem_usage;
  }

  while (true) {
    int count_slice = count / slice_num + 1;
    C_param = (count_slice + EU_NUM - 1) / EU_NUM;
    c_per_npu = (C_param + NPU_NUM - 1) / NPU_NUM;
    local_mem_usage = c_per_npu * aligned_csize * blob_num;
    if (local_mem_usage <= local_mem_size && c_per_npu < 0x1000) {
      return slice_num;
    } else if (slice_num < count) {
      slice_num++;
    } else {
      assert(0);
    }
  }
}

/**
 * \brief load bias(16bytes) or 32 bit multiplier, cus multiplier store int
 * 'bias' \tl_bias set nullptr once INT8_PER_LAYER && !do_bias
 */
void CviBackendContext::load_bias_multiplier(int oc_step, // output channel
                                             bool do_bias, gaddr_t bias_gaddr,
                                             int qmode,
                                             cvk_tl_t **tl_bias) const {

  if (qmode == QuantizeMode::INT8_32_MULTIPLER) {
    load_32byte_multiplier(oc_step, do_bias, bias_gaddr, tl_bias);
  } else if (qmode == QuantizeMode::INT8_PER_LAYER) {
    if (do_bias) {
      load_16bytes_bias(oc_step, tl_bias, bias_gaddr);
    } else {
      *tl_bias = nullptr;
    }
  }
}

void CviBackendContext::load_32byte_multiplier(
    int oc_step, bool do_bias, gaddr_t bias_gaddr,
    cvk_tl_t **tl_chl_quan_param // 32 byte multiplier
) const {

  if (do_bias) {
    // call tl_default_stride
    cvk_tl_shape_t coeff_shape_9byte = tl_shape_t4(1, oc_step, 1, 9);
    *tl_chl_quan_param =
        lmem_alloc_tensor(coeff_shape_9byte, CVK_FMT_U8, /*eu_align=*/0);
  } else {
    cvk_tl_shape_t coeff_shape_5byte = tl_shape_t4(1, oc_step, 1, 5);
    *tl_chl_quan_param =
        lmem_alloc_tensor(coeff_shape_5byte, CVK_FMT_U8, /*eu_align=*/0);
  }

  tdma_load(*tl_chl_quan_param, bias_gaddr);
}

void CviBackendContext::load_16bytes_bias(int oc, cvk_tl_t **tl_bias,
                                          gaddr_t bias_gaddr) const {

  cvk_tl_shape_t tl_bias_shape;
  tl_bias_shape.n = 2;
  tl_bias_shape.c = oc;
  tl_bias_shape.h = 1;
  tl_bias_shape.w = 1;
  *tl_bias = lmem_alloc_tensor(tl_bias_shape, CVK_FMT_I8, 0);
  tdma_load(*tl_bias, bias_gaddr);
}

// apply quantize int 8 mode
void CviBackendContext::apply_qi8(cvk_tl_t *ifmap, uint32_t layer_id,
                                  int do_relu, int right_shift_width,
                                  int threshold_x_quantized) const {
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = ifmap;
  p.a = ifmap;
  p.b_const.val = threshold_x_quantized;
  p.b_const.is_signed = false;
  p.b_is_const = 1;
  p.rshift_bits = right_shift_width;
  p.layer_id = layer_id;
  p.relu_enable = do_relu;
  tiu_mul(&p);
}

int CviBackendContext::tensor_size_lmem(int n, int c, int h, int w,
                                        cvk_fmt_t fmt) const {
  return n * ALIGN(c, NPU_NUM) * get_csize_local(h, w, fmt);
}

void CviBackendContext::assert_support_fmt(cvk_fmt_t fmt) const {
  assert((fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8 || fmt == CVK_FMT_BF16) &&
         "others not supported");
}

cvk_tl_stride_t CviBackendContext::tl_fp32_stride(cvk_tl_t *tl,
                                                  int eu_align) const {
  int fmt_sz = 4; // 4 means fp32 takes 4 bytes
  cvk_tl_stride_t s;

  s.w = fmt_sz;
  s.h = tl->shape.w * fmt_sz;
  s.c = tl->shape.h * tl->shape.w * fmt_sz;

  if (eu_align) {
    s.c = align_up(s.c, EU_NUM);
  }

  s.n = s.c * ceiling_func(tl->shape.c, NPU_NUM);
  return s;
}

void CviBackendContext::fill_fp32_lmem_0(uint32_t layer_id, int batch,
                                         int channel, int height,
                                         int width) const {
  int blob_num = 3; // 2 for output fp32, 1 for load bf16
  int input_n = batch;
  int input_c = channel;
  int input_h = height;
  int input_w = width;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  // +2 means we prevent wrap to top, reserver it
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 2;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;

  // lmem fmt store as bf16
  tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt, &tiling_info);

  int i = 0;
  int n = tiling_info[i].first.n;
  int c = tiling_info[i].first.c;
  int h = tiling_info[i].first.h;
  int w = tiling_info[i].first.w;

  cvk_tl_t *bottom;

  // force clean all
  cvk_tl_shape_t input_shape =
      tl_shape_t4(n, c, h * 2, w * 2); // fp32 takes 4 times than int8
  bottom = lmem_alloc_tensor(input_shape, CVK_FMT_I8, /*eu_align=*/0);
  cvk_tiu_xor_int8_param_t param = {0};
  param.res = bottom;
  param.a = bottom;
  param.b = bottom;
  param.layer_id = layer_id;
  tiu_xor_int8(&param);

  lmem_free_tensor(bottom);
}

void CviBackendContext::lmem_shrink_fp32_bf16(cvk_tl_t *lmem_bf16,
                                              cvk_tl_t *lmem_fp32, int bf16_n,
                                              int bf16_c, int bf16_h,
                                              int bf16_w,
                                              uint32_t layer_id) const {

  assert((uint32_t)bf16_w * 2 == lmem_fp32->shape.w &&
         lmem_fp32->shape.h == (uint32_t)bf16_h &&
         lmem_fp32->shape.c == (uint32_t)bf16_c &&
         lmem_fp32->shape.n == (uint32_t)bf16_n &&
         "the fp32's width should be twice than bf16's");

  // move high 16bit as bf16 format
  *lmem_bf16 = *lmem_fp32;
  lmem_bf16->shape = tl_shape_t4(bf16_n, bf16_c, bf16_h, bf16_w);
  lmem_bf16->stride =
      tl_default_stride(lmem_bf16->shape, lmem_bf16->fmt, /*eu_align=*/0);

  // fake shape for cmodel constrain that shape SHOULD be equal
  lmem_fp32->shape = tl_shape_t4(bf16_n, bf16_c, bf16_h, bf16_w);
  lmem_fp32->stride = tl_fp32_stride(lmem_fp32);

  laddr_t lmem_fp32_addr = lmem_fp32->start_address;
  lmem_fp32->start_address = lmem_fp32_addr + 2; // start with high 16 bits

  cvk_tiu_copy_param_t param = {0};
  param.src = lmem_fp32;
  param.dst = lmem_bf16;
  param.layer_id = layer_id;
  tiu_copy(&param);
}

void *cvi_backend_get_cvk_ctx(const CviBackendContext &ctx) {
  return ctx.get_cvk_ctx();
}