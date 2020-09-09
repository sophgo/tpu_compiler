/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_tiling.cpp
 * Description:
 */


#if 0

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#if 1
// define bm_kernel.h
static inline int bitsize_of_fmt(uint32_t fmt)
{
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
void tiling_packing(const CviBackendContext &ctx,
    int require_shape, int coeff_lane_shape, int blob_num, cvk_fmt_t fmt,
    std::vector<std::pair<cvk_tl_shape_t, gaddr_t> >* tiling_info) {

  assert(fmt == CVK_FMT_BF16 || fmt == CVK_FMT_I8);

  int data_type_size = bitsize_of_fmt(fmt) / 8; // byte
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

        if (required_size <= LOCAL_MEM_SIZE - coeff_lane_size)
          break;
      } while (--height);

      int step_shape = height * NPU_NUM * EU_NUM;

      LLVM_DEBUG(llvm::errs() << llvm::format("    step_shape %d, require_shape %d, gaddr_offset 0x%lx\n",
                                            step_shape, require_shape, gaddr_offset););

      tiling_info->push_back(std::make_pair(ctx.shape_t4(1, NPU_NUM, height, EU_NUM), gaddr_offset));

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
                                          step_shape, require_shape, gaddr_offset););
    require_shape -= step_shape;
    gaddr_offset += step_shape * data_type_size;
  }

  assert(tiling_info->size());
}

#endif
