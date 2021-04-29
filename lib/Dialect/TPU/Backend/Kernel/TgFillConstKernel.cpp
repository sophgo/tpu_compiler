/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFillConstKernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "cvi_backend_fill_const_kernel"

void cvi_backend_tg_fill_const_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ofmap, int output_n,
    int output_c, int output_h, int output_w, float const_val,
    cvk_fmt_t fmt) {

  ctx.set_layer_id(layer_id);
  cvk_tdma_l2g_tensor_fill_constant_param_t p = {0};
  cvk_tg_t dst;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  dst.start_address = ga_ofmap;
  dst.fmt = fmt;
  dst.shape = ctx.tg_shape_t4(output_n, output_c, output_h, output_w);
  dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);
  if (fmt == CVK_FMT_BF16) {
    p.constant = ctx.convert_fp32_to_bf16(const_val);
  } else if (fmt == CVK_FMT_I8) {
    assert(const_val >= -128 && const_val <= 127);
    int8_t val = (int8_t)const_val;
    p.constant = *((uint8_t *)&val);
  } else {
    assert(0);
  }
  p.dst = &dst;
  ctx.tdma_l2g_tensor_fill_constant(&p);
}
