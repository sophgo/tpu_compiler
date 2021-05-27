/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgSwapChannelKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

void cvi_backend_tg_swap_channel_kernel(const CviBackendContext &ctx,
                                        uint32_t layer_id, gaddr_t input_gaddr,
                                        gaddr_t output_gaddr,
                                        int input_dim_size, int *input_dim,
                                        int *channel_order, cvk_fmt_t fmt) {
  assert(input_dim_size == 4 && input_dim[1] == 3 && "parameter error");
  cvk_tg_shape_t shape =
      ctx.tg_shape_t4(input_dim[0], 1, input_dim[2], input_dim[3]);
  cvk_tg_stride_t stride =
      ctx.tg_default_stride(input_dim[1], input_dim[2], input_dim[3], fmt);
  uint64_t frame_size = input_dim[2] * input_dim[3] * ctx.bytesize_of_fmt(fmt);
  for (uint32_t i = 0; i < 3; i++) {
    assert((uint32_t)channel_order[i] < 3 && "channel_order is illegal");
    gaddr_t s_gaddr = input_gaddr + frame_size * channel_order[i];
    gaddr_t d_gaddr = output_gaddr + frame_size * i;
    ctx.tdma_g2g_tensor_copy(s_gaddr, shape, stride, fmt, d_gaddr, shape,
                             stride, fmt);
  }
}
