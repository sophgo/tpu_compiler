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

void shuffle_channel_forward_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int batch, int channel, int frame_size, int group,  cvi_backend_fmt_t fmt) {
  frame_size *= ((fmt == CVI_FMT_BF16) ? sizeof(uint16_t) : sizeof(uint8_t));
  uint32_t feature_size = (uint32_t)(channel * frame_size);
  uint32_t col = (uint32_t)(channel / group);
  cvk_tg_shape_t shape = {(uint32_t)batch, (uint32_t)group, (uint32_t)col, (uint32_t)frame_size};
  cvk_tg_stride_t s_stride = {(uint32_t)feature_size, (uint32_t)(col * frame_size),
                                              (uint32_t)frame_size};
  cvk_tg_stride_t d_stride = {(uint32_t)feature_size, (uint32_t)frame_size,
                                              (uint32_t)(frame_size * group)};

  LLVM_DEBUG(llvm::dbgs()
            << "shuffle_channel_forward_kernel\n"
            << "  layer_id " << layer_id << "\n");

  // For tdma
  ctx.set_layer_id(layer_id);

  tdma_g2g_tensor_copy(ctx, input_gaddr, shape, s_stride, CVK_FMT_I8, output_gaddr,
                       shape, d_stride, CVK_FMT_I8);
}
