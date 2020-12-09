/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgScaleKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bm1880v2_scale"

void cvi_backend_tg_scale_kernel(const CviBackendContext &ctx,
                                 uint32_t layer_id, gaddr_t input_gaddr,
                                 gaddr_t scale_gaddr, gaddr_t bias_gaddr,
                                 gaddr_t output_gaddr, int input_n, int input_c,
                                 int input_h, int input_w, int scale_dim,
                                 int inner_dim, bool is_scale_const,
                                 int const_scale, int do_relu, bool do_bias,
                                 cvk_fmt_t fmt) {
  if (is_scale_const) {
    assert(0 && "TODO: Scale Const");
  }
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tg_scale_kernel:\n"
                 "    layer_id %d\n"
                 "    bottom = %lx, scale_gaddr = 0x%lx, bias_gaddr = 0x%lx, "
                 "top = %lx\n"
                 "    nchw = (%d, %d, %d, %d)\n"
                 "    do_bias = %d"
                 "    scale_dim = %d, inner_dim = %d \n",
                 layer_id, input_gaddr, scale_gaddr, bias_gaddr, output_gaddr,
                 input_n, input_c, input_h, input_w, do_bias, scale_dim,
                 inner_dim););

  assert(input_n * input_c * input_h * input_w == scale_dim * inner_dim);
  // input_c = scale_dim;
  if (inner_dim != input_h * input_w) {
    input_h = inner_dim;
    input_w = 1;
  }

  cvk_tg_shape_t ts_bottom_shape =
      ctx.tg_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tg_stride_t ts_bottom_stride =
      ctx.tg_default_stride(ts_bottom_shape, fmt);

  uint32_t lmem_usage = 0;
  // load scale
  cvk_tl_shape_t tl_scale_shape = ctx.tl_shape_t4(1, input_c, 1, 1);
  cvk_tl_t *tl_scale = ctx.lmem_alloc_tensor(tl_scale_shape, fmt, 1);
  ctx.tdma_load(tl_scale, scale_gaddr);
  lmem_usage += ctx.lmem_tensor_to_size(tl_scale_shape, fmt, 1);

  cvk_tl_t *tl_bias = nullptr;
  cvk_tl_t tl_bias_fixed;
  cvk_tl_shape_t bias_shape;
  if (fmt == CVK_FMT_BF16) {
    if (do_bias) {
      bias_shape = ctx.tl_shape_t4(2, input_c, 1, 1);
      tl_bias = ctx.lmem_alloc_tensor(bias_shape, fmt, 1);
      ctx.tdma_load(tl_bias, bias_gaddr);
      lmem_usage += ctx.lmem_tensor_to_size(bias_shape, fmt, 1);
    }
  } else {
    int bias_size = ctx.chan_quan_param_size(do_bias);
    bias_shape = ctx.tl_shape_t4(1, input_c, 1, bias_size);
    tl_bias = ctx.lmem_alloc_tensor(bias_shape, CVK_FMT_U8, 0);
    ctx.tdma_load(tl_bias, bias_gaddr);
    lmem_usage += ctx.lmem_tensor_to_size(bias_shape, fmt, 1);
    tl_bias_fixed = *tl_bias;
    tl_bias_fixed.shape = ctx.tl_shape_t4(1, input_c, 1, 1);
    tl_bias_fixed.stride = ctx.tl_default_stride(tl_bias_fixed.shape, fmt, 0);
  }

  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, ts_bottom_shape, fmt, 1, lmem_usage,
                     CviBackendContext::TilingNHW);
  cvk_tl_shape_t max_shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(max_shape, fmt, 1);
  if (tl_input == nullptr) {
    llvm::errs() << llvm::format(
        "lmem_alloc_tensor failed, src shape:(%d,%d,%d,%d)\n", max_shape.n,
        max_shape.c, max_shape.h, max_shape.w);
    assert(0);
  }
  for (auto &tile : tiles) {
    cvk_tl_t tl_bslice = *tl_input;
    tl_bslice.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    tl_bslice.stride = ctx.tl_default_stride(tl_bslice.shape, fmt, 1);

    ctx.tdma_load_stride(&tl_bslice, input_gaddr + tile.offset,
                         ts_bottom_stride);

    if (fmt == CVK_FMT_BF16) {
      cvk_tiu_depthwise_pt_convolution_param_t param = {0};
      param.ofmap = &tl_bslice;
      param.ifmap = &tl_bslice;
      param.weight = tl_scale;
      param.bias = tl_bias;
      param.ins_h = 0;
      param.ins_last_h = 0;
      param.ins_w = 0;
      param.ins_last_w = 0;
      param.pad_top = 0;
      param.pad_bottom = 0;
      param.pad_left = 0;
      param.pad_right = 0;
      param.stride_h = 1;
      param.stride_w = 1;
      param.dilation_h = 1;
      param.dilation_w = 1;
      param.relu_enable = do_relu;
      param.layer_id = layer_id;
      param.ins_val = 0;                            // symmetric quantization
      param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
      ctx.tiu_pt_depthwise_convolution(&param);
    } else {
      cvk_tiu_depthwise_convolution_param_t param = {nullptr};
      param.ofmap = &tl_bslice;
      param.ifmap = &tl_bslice;
      param.weight = tl_scale;
      param.chl_quan_param = &tl_bias_fixed;
      param.ins_h = 0;
      param.ins_last_h = 0;
      param.ins_w = 0;
      param.ins_last_w = 0;
      param.pad_top = 0;
      param.pad_bottom = 0;
      param.pad_left = 0;
      param.pad_right = 0;
      param.stride_h = 1;
      param.stride_w = 1;
      param.dilation_h = 1;
      param.dilation_w = 1;
      param.has_bias = do_bias ? 1 : 0;
      param.relu_enable = do_relu;
      param.layer_id = layer_id;
      param.ins_val = 0;                            // symmetric quantization
      param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
      ctx.tiu_depthwise_convolution(&param);
    }

    ctx.tdma_store_stride(&tl_bslice, output_gaddr + tile.offset,
                          ts_bottom_stride);
  }
  ctx.lmem_free_tensor(tl_input);

  if (tl_bias != nullptr) {
    ctx.lmem_free_tensor(tl_bias);
  }
  ctx.lmem_free_tensor(tl_scale);
}
