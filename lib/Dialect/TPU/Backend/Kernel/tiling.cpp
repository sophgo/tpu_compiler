/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tiling.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <iostream>
#include "CviBackendContext.h"

#define DEBUG_TYPE "tiling"

#if 1
// define bm_kernel.h
int bitsize_of_fmt(uint32_t fmt) {
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

#endif

// \return rest
void tiling_packing(const CviBackendContext &ctx, int require_shape, int coeff_lane_shape,
                    int blob_num, cvk_fmt_t fmt,
                    std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > *tiling_info) {

  assert(fmt == CVK_FMT_BF16 || fmt == CVK_FMT_I8);

  int data_type_size = bitsize_of_fmt(fmt) / 8;  // byte
  int coeff_lane_size = coeff_lane_shape * data_type_size;
  gaddr_t gaddr_offset = 0;

  // available size per lane = LOCAL_MEM_SIZE - coeff_lane_size
  int height = require_shape / (NPU_NUM * EU_NUM);
  if (require_shape + coeff_lane_shape * NPU_NUM >= (NPU_NUM * EU_NUM) && height) {
    do {
      // Find height
      height = require_shape / (NPU_NUM * EU_NUM);
      do {
        cvk_tl_shape_t tmp_shape = ctx.shape_t4(1, 1, height, EU_NUM);
        int required_size = blob_num * ctx.lmem_tensor_to_size(tmp_shape, fmt, /*eu_align=*/1);

        if (required_size <= LOCAL_MEM_SIZE - coeff_lane_size) break;
      } while (--height);

      int step_shape = height * NPU_NUM * EU_NUM;

      LLVM_DEBUG(llvm::errs() << llvm::format(
          "    step_shape %d, require_shape %d, gaddr_offset 0x%lx\n",
          step_shape, require_shape, gaddr_offset););

      tiling_info->push_back(
          std::make_pair(ctx.shape_t4(1, NPU_NUM, height, EU_NUM), gaddr_offset));

      // Update step
      require_shape -= step_shape;
      gaddr_offset += step_shape * data_type_size;
    } while (require_shape >= ((NPU_NUM * EU_NUM)));
  }

  // Use one lane to handle remaining
  if (require_shape) {
    int step_shape = require_shape;

    tiling_info->push_back(std::make_pair(ctx.shape_t4(1, 1, 1, step_shape), gaddr_offset));

    LLVM_DEBUG(llvm::errs() << llvm::format("    (r)step_shape %d, require_shape %d, gaddr_offset 0x%lx\n",
                                 step_shape, require_shape, gaddr_offset));
    require_shape -= step_shape;
    gaddr_offset += step_shape * data_type_size;
  }

  assert(tiling_info->size());
}

void _split_nh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
               uint32_t reserved, int *n_slices, int *h_slices) {
  *h_slices = 1;
  *n_slices = 1;
  uint32_t total_lmem_needs = blob_num * __get_lmem_usage(ctx, n, c, h, w);

  LLVM_DEBUG(llvm::errs() << llvm::format(
       "<%d,%d,%d,%d>, reserved:%u, total:%u\n",
       n, c, h, w, reserved, total_lmem_needs););

  if (total_lmem_needs + reserved <= (uint32_t)LOCAL_MEM_SIZE) {
    return;
  }

  // split h if lmem usage per image is larger than LOCAL_MEM_SIZE
  if (n == 1 || total_lmem_needs > (uint32_t)(LOCAL_MEM_SIZE * n)) {
    *n_slices = n;
    total_lmem_needs = total_lmem_needs / n;

    *h_slices = (total_lmem_needs + LOCAL_MEM_SIZE - 1) / LOCAL_MEM_SIZE;
    int h_units_per_slice = (h + *h_slices - 1) / *h_slices;
    LLVM_DEBUG(llvm::errs()
        << "h_units_per_slice is " << h_units_per_slice
        << ", h_slices is " << *h_slices<< "\n";);

    while (blob_num * __get_lmem_usage(ctx, 1, c, h_units_per_slice, w) + reserved >
               (uint32_t)LOCAL_MEM_SIZE ||
           h_units_per_slice > (4095 - 32)) {
      *h_slices += 1;
      h_units_per_slice = (h + *h_slices - 1) / *h_slices;

      LLVM_DEBUG(llvm::errs()
          << "h_units_per_slice is " << h_units_per_slice
          << ", h_slices is " << *h_slices << "\n";);
    }
  } else {  // split n if local memory can store more than on image
    *n_slices = (total_lmem_needs + LOCAL_MEM_SIZE - 1) / LOCAL_MEM_SIZE;
    int n_units_per_slice = (n + *n_slices - 1) / *n_slices;

    while (blob_num * __get_lmem_usage(ctx, n_units_per_slice, c, h, w) + reserved >
           (uint32_t)LOCAL_MEM_SIZE) {
      *n_slices += 1;
      n_units_per_slice = (n + *n_slices - 1) / *n_slices;
    }
  }
}

void _split_cnh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
                uint32_t reserved, int *c_slices, int *n_slices, int *h_slices) {
  *c_slices = 1;
  *n_slices = 1;
  *h_slices = 1;
  uint32_t total_lmem_needs = blob_num * __get_lmem_usage(ctx, n, c, h, w) + reserved;

  if (total_lmem_needs > (uint32_t)LOCAL_MEM_SIZE) {
    if (c > NPU_NUM) {
      int c_units_per_npu = (c + NPU_NUM - 1) / NPU_NUM;

      if (total_lmem_needs / c_units_per_npu <= LOCAL_MEM_SIZE - reserved) {
        int _c = (c + *c_slices - 1) / *c_slices;

        while (blob_num * __get_lmem_usage(ctx, n, _c, h, w) + reserved > (uint32_t)LOCAL_MEM_SIZE) {
          *c_slices += 1;
          _c = (c + *c_slices - 1) / *c_slices;
        }
        return;
      }
    }
    _split_nh(ctx, n, c, h, w, blob_num, reserved, n_slices, h_slices);
  }
}

int __split(const CviBackendContext &ctx, int blob_num, int count) {
  int slice_num = 1;
  int W_param = EU_NUM;
  int C_param = (count + EU_NUM - 1) / EU_NUM;
  int aligned_csize = get_csize_local(ctx, 1, W_param);
  int c_per_npu = (C_param + NPU_NUM - 1) / NPU_NUM;
  int local_mem_usage = c_per_npu * aligned_csize * blob_num;
  const int local_mem_size = LOCAL_MEM_SIZE;
  int proportion_mem_usage = (local_mem_usage + local_mem_size - 1) / local_mem_size;

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

int shape_size(int n, int c, int h, int w, cvk_fmt_t fmt) {
  return n * c * h * w * (bitsize_of_fmt(fmt) / 8);
}
