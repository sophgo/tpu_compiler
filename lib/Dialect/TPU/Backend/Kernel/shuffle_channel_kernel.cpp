/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: shuffle_channel_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "cvi_backend_shuffle_channel_kernel"

void cvi_backend_tg_shuffle_channel_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int batch, int channel, int frame_size, int group,  cvk_fmt_t fmt) {
  int unit_size = ((fmt == CVK_FMT_BF16) ? sizeof(uint16_t) : sizeof(uint8_t));
  frame_size *= unit_size;
  uint32_t feature_size = (uint32_t)(channel * frame_size);
  uint32_t col = (uint32_t)(channel / group);
  cvk_tg_shape_t shape = {(uint32_t)batch, (uint32_t)group, (uint32_t)col, (uint32_t)frame_size};
  cvk_tg_stride_t s_stride = {(uint32_t)feature_size, (uint32_t)(col * frame_size),
                                              (uint32_t)frame_size};
  cvk_tg_stride_t d_stride = {(uint32_t)feature_size, (uint32_t)frame_size,
                                              (uint32_t)(frame_size * group)};

  LLVM_DEBUG(llvm::dbgs()
            << "cvi_backend_tg_shuffle_channel_kernel\n"
            << "  layer_id " << layer_id << "\n");

  // For tdma
  ctx.set_layer_id(layer_id);

  shape.n = 1; // TODO: support n dim shift
  shape.w /= unit_size;
  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;
  tiling_packing(ctx, /*require_shape=*/0, /*coeff_lane_shape=*/0, /*blob_num=*/1,
      fmt, &tiling_info, TilingDimNH, &shape);

  // accumuate store
  int output_n = batch;
  for (int batch = 0; batch < output_n; batch++) {
    int top_h_shift = 0;
    int bottom_local_shift = 0;
    for (size_t i = 0; i < tiling_info.size(); i++) {
      int n = tiling_info[i].first.n;
      int c = tiling_info[i].first.c;
      int h = tiling_info[i].first.h;
      int w = tiling_info[i].first.w;
      //gaddr_t gaddr_offset = tiling_info[i].second;

      assert(w == frame_size / unit_size && "not support tiling w");

      // load
      cvk_tl_shape_t input_shape = ctx.shape_t4(n, c, h, w);
      cvk_tl_t *bottom = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);
      ctx.tdma_load_stride_bf16(bottom,
          input_gaddr + bottom_local_shift + batch * s_stride.n, s_stride);
      bottom_local_shift += bottom->stride.c;

      // store back
      ctx.tdma_store_stride_bf16(bottom,
          output_gaddr + top_h_shift * d_stride.h + batch * d_stride.n, d_stride,
          /*do_transpose=*/0);

      // shift gaddr
      top_h_shift += bottom->shape.h;

      // release
      ctx.lmem_free_tensor(bottom);
    }
  }
}
