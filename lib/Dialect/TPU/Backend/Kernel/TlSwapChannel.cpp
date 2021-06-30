/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_concat.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "tl_swapchannel"

void cvi_backend_tl_swap_channel(const CviBackendContext &ctx,
                                 uint32_t layer_id, laddr_t la_input,
                                 laddr_t la_output, int n, int c, int h, int w,
                                 int *order, cvk_fmt_t fmt) {
  ctx.parallel_disable();
  cvk_tl_t tl_input = {0};
  tl_input.fmt = fmt;
  tl_input.shape = ctx.tl_shape_t4(n, 1, h, w);
  tl_input.stride = ctx.tl_default_stride(ctx.tl_shape_t4(n, 3, h, w), fmt, 1);
  cvk_tl_t tl_output = tl_input;
  cvk_tdma_l2l_tensor_copy_param_t p = {0};
  p.dst = &tl_output;
  p.src = &tl_input;
  p.layer_id = layer_id;
  for (int i = 0; i < 3; i++) {
    tl_input.start_address = la_input + order[i] * LOCAL_MEM_SIZE;
    tl_output.start_address = la_output + i * LOCAL_MEM_SIZE;
    ctx.tdma_l2l_tensor_copy(&p);
  }
}
