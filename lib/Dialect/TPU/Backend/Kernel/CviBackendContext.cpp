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
  tdmaBaseSelects[SHARED_MEMORY] = 0;
  tdmaBaseSelects[WEIGHT_MEMORY] = 1;
  tdmaBaseSelects[PRIVATE_MEMORY] = 2;
  tdmaBaseSelects[IO_MEMORY_0] = 3;
  tdmaBaseSelects[IO_MEMORY_1] = 4;
  tdmaBaseSelects[IO_MEMORY_2] = 5;
  tdmaBaseSelects[IO_MEMORY_3] = 6;
  tdmaBaseSelects[IO_MEMORY_4] = 7;

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
  // we store memory region value in bits (40 ~ 42) of gaddr;
  uint32_t memoryRegion = ((((uint64_t)gaddr) >> 40) & 0x07);
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
    cvk_cmpr_tg_t cmpr_dst = {0};
    cmpr_dst.bias0 = (ts_data.fmt == CVK_FMT_BF16) ? 127 : 0;
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
  uint32_t total_lmem_needs = blob_num * lmem_tensor_to_size(n, c, h, w);

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

    while (blob_num * lmem_tensor_to_size(1, c, h_units_per_slice, w) + reserved >
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

    while (blob_num * lmem_tensor_to_size(n_units_per_slice, c, h, w) + reserved >
           (uint32_t)LOCAL_MEM_SIZE) {
      *n_slices += 1;
      n_units_per_slice = (n + *n_slices - 1) / *n_slices;
    }
  }
}

void CviBackendContext::tiling_all(std::vector<tiling_info_t> &tiling_result,
                                   int64_t total, cvk_fmt_t fmt, int blob_num,
                                   uint32_t lmem_size, bool do_parallel) const {
  tiling_info_t tile;
  memset(&tile, 0, sizeof(tile));
  int max_slice = (do_parallel ? TILING_SLICE_NUM : 1);
  tile.n = 1;
  tile.c = NPU_NUM;
  tile.w = EU_NUM;
  tile.h = std::max(1, ceiling_func(total / (NPU_NUM * EU_NUM), max_slice));
  tile.h = std::min(tile.h, MAX_HEIGHT);
  bool lmem_ok = false;
  while (total > 0) {
    int64_t count = tile.n * tile.c * tile.h * tile.w;
    if (lmem_ok == false) {
      uint32_t lsize = blob_num * lmem_tensor_to_size(tile.n, tile.c, tile.h,
                                                      tile.w, fmt, 1);
      lmem_ok = (lsize <= lmem_size);
    }
    if (count > total || lmem_ok == false) {
      if (tile.h > 1) {
        tile.h--;
      } else if (tile.w > 1) {
        tile.w--;
      } else if (tile.c > 1) {
        tile.c--;
      } else {
        assert(0 && "lmem is not enough");
      }
    } else {
      LLVM_DEBUG(llvm::errs() << llvm::format(
                     "Tiles all, tile:(%d,%d,%d,%d), offset:%lu\n", tile.n,
                     tile.c, tile.h, tile.w, tile.offset););
      tiling_result.emplace_back(tile);
      total -= count;
      tile.offset += count * bytesize_of_fmt(fmt);
    }
  }
  assert(total == 0 && "tiling error");
  return;
}

static int slice_dim(int & max_dim, int dim, int max_num, int max_slice, int unit = 1) {
  if (dim <= unit || max_slice <= 1) {
    max_dim = dim;
    return max_slice;
  }
  int slice = ceiling_func(dim, unit);
  max_dim = ceiling_func(slice, max_slice) * unit;
  max_dim = std::min(max_dim, ALIGN(max_num, unit));
  slice = ceiling_func(dim, max_dim);
  return ceiling_func(max_slice, slice);
}

void CviBackendContext::tiling_nchw(std::vector<tiling_info_t> &tiling_result,
                                    int n, int c, int h, int w, cvk_fmt_t fmt,
                                    int blob_num, uint32_t lmem_size,
                                    tiling_mode_t mode,
                                    bool do_parallel) const {
  int max_w = std::min(w, MAX_WIDTH);
  int max_h = std::min(h, MAX_HEIGHT);
  int max_c = std::min(c, MAX_CHANNEL);
  int max_n = std::min(n, MAX_CHANNEL);
  int min_c = 1;
  int max_slice = (do_parallel ? TILING_SLICE_NUM : 1);
  if (mode == TilingDimNHW) { // keep c
    assert(max_c == c && "keep c, but c too large");
    min_c = max_c;
  } else if (do_parallel) {
    // slice c
    // e.g c = 97 and npu_num = 32, then c will slice to 32,32,32,1
    max_slice = slice_dim(max_c, c, MAX_CHANNEL, max_slice, NPU_NUM);
  }
  if (max_slice > 1) {
    // slice n
    // e.g n = 3, then will slice to 1,1,1
    max_slice = slice_dim(max_n, n, MAX_CHANNEL, max_slice);
    if (h >= EU_NUM || w >= EU_NUM) { // slice h and w
      int &a = (h > w ? h : w);
      int &b = (h > w ? w : h);
      int &max_a = (h > w ? max_h : max_w);
      int &max_b = (h > w ? max_w : max_h);
      max_slice = slice_dim(max_a, a, MAX_WIDTH, max_slice, EU_NUM);
      max_slice = slice_dim(max_b, b, MAX_WIDTH, max_slice);
    }
  }

  int step_w, step_h, step_c, step_n;
  uint32_t lmem_required;
  for (step_w = max_w; step_w >= 1; --step_w) {
    for (step_h = max_h; step_h >= 1; --step_h) {
      for (step_n = max_n; step_n >= 1; --step_n) {
        for (step_c = max_c; step_c >= min_c;) {
          cvk_tl_shape_t max_shape =
              tl_shape_t4(step_n, step_c, step_h, step_w);
          lmem_required = blob_num * lmem_tensor_to_size(max_shape, fmt, 1);
          if (lmem_required <= lmem_size) {
            goto after_loop;
          }
          if (step_c % NPU_NUM) {
            step_c -= step_c % NPU_NUM;
          } else {
            step_c -= NPU_NUM;
          }
        }
      }
    }
  }
after_loop:
  if (lmem_required > lmem_size) {
    llvm::errs() << llvm::format(
        "Tilling[%d] failed, src shape:(%d,%d,%d,%d), fmt:%d\n", n, c, h, w,
        fmt);
    assert(0);
  }

  tiling_info_t tile;
  cvk_tg_stride_t src_stride = tg_default_stride(c, h, w, fmt);
  for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
    tile.n = std::min(n - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
      tile.c = std::min(c - tile.pos_c, step_c);
      for (tile.pos_h = 0; tile.pos_h < h; tile.pos_h += step_h) {
        tile.h = std::min(h - tile.pos_h, step_h);
        for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
          tile.w = std::min(w - tile.pos_w, step_w);
          tile.offset = tile.pos_w * src_stride.w + tile.pos_h * src_stride.h +
                        tile.pos_c * src_stride.c + tile.pos_n * src_stride.n;
          tiling_result.emplace_back(tile);
          LLVM_DEBUG(llvm::errs() << llvm::format(
                         "Tiles[%d], tile:(%d,%d,%d,%d), pos:(%d,%d,%d,%d), "
                         "offset:%lu\n",
                         mode, tile.n, tile.c, tile.h, tile.w, tile.pos_n,
                         tile.pos_c, tile.pos_h, tile.pos_w, tile.offset););
        }
      }
    }
  }
}

void CviBackendContext::tiling_packing(
    std::vector<tiling_info_t> &tiling_result, cvk_tg_shape_t shape,
    cvk_fmt_t fmt, int blob_num, uint32_t reserved_lmem,
    tiling_mode_t mode, bool do_parallel) const {
  int n = static_cast<int>(shape.n);
  int c = static_cast<int>(shape.c);
  int h = static_cast<int>(shape.h);
  int w = static_cast<int>(shape.w);
  tiling_packing(tiling_result, n, c, h, w, fmt, blob_num, reserved_lmem, mode, do_parallel);
}

void CviBackendContext::tiling_packing(
    std::vector<tiling_info_t> &tiling_result, int n, int c, int h, int w,
    cvk_fmt_t fmt, int blob_num, uint32_t reserved_lmem,
    tiling_mode_t mode, bool do_parallel) const {
  uint32_t lmem_size = (uint32_t)LOCAL_MEM_SIZE - reserved_lmem;
  assert((uint32_t)LOCAL_MEM_SIZE > reserved_lmem && "reserved_lmem too large");

  if (mode == TilingDimAll) {
    tiling_all(tiling_result, n * c * h * w, fmt, blob_num, lmem_size, do_parallel);
  } else {
    tiling_nchw(tiling_result, n, c, h, w, fmt, blob_num, lmem_size, mode, do_parallel);
  }
}

void CviBackendContext::assert_support_fmt(cvk_fmt_t fmt) const {
  assert((fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8 || fmt == CVK_FMT_BF16) &&
         "others not supported");
}

void *cvi_backend_get_cvk_ctx(const CviBackendContext &ctx) {
  return ctx.get_cvk_ctx();
}