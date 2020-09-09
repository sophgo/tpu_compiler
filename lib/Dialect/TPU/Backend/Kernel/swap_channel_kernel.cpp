/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: swap_channel_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

void cvi_backend_tg_swap_channel_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                 uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
                                 uint32_t depends_len, gaddr_t input_gaddr,
                                 gaddr_t output_gaddr, int input_dim_size,
                                 int *input_dim, int * channel_order, cvk_fmt_t fmt) {
  assert(input_dim_size == 4 && input_dim[1] == 3 && "paramter error");
  uint32_t fmt_size = ((fmt == CVK_FMT_BF16) ? sizeof(uint16_t) : sizeof(uint8_t));
  uint32_t n = input_dim[0];
  uint32_t c = input_dim[1];
  uint32_t h = input_dim[2];
  uint32_t w = input_dim[3] * fmt_size;
  cvk_tg_shape_t shape = {n, 1, h, w};
  cvk_tg_stride_t stride = {c * h * w, h * w, w};
  for (uint32_t i = 0; i < c; i++) {
    assert((uint32_t)channel_order[i] < c && "channel_order is illegal");
    gaddr_t s_gaddr = input_gaddr + h * w * channel_order[i];
    gaddr_t d_gaddr = output_gaddr + h * w * i;
    tdma_g2g_tensor_copy(ctx, s_gaddr, shape, stride, CVK_FMT_I8, d_gaddr, shape,
                         stride, CVK_FMT_I8);
  }
}
