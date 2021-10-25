/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgConvFcKernel.cpp
 * Description:
 */

#include "TgBf16ConvFcKernel.hpp"
#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

// use conv to do fc
// M = 1

void TgConvFcKernel::init(uint32_t layer_id, gaddr_t ga_input,
                          gaddr_t ga_filter, gaddr_t ga_output, int M, int K,
                          int N, bool do_quant, gaddr_t ga_scale,
                          gaddr_t ga_zeropoint) {
  this->layer_id = layer_id;
  this->K = K;
  this->M = M;
  this->N = N;
  this->ga_input = ga_input;
  this->ga_filter = ga_filter;
  this->ga_output = ga_output;
  this->do_quant = do_quant;
  this->ga_scale = ga_scale;
  this->ga_zeropoint = ga_zeropoint;
  this->fmt = CVK_FMT_BF16;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  ctx.set_layer_id(layer_id);
}

void TgConvFcKernel::selectTilePolicy() {
  int maxH = std::min(std::max(1, N / NPU_NUM), MAX_HEIGHT);
  int stepH = maxH;
  auto input_shape = ctx.tl_shape_t4(1, NPU_NUM, 1, K);
  auto input_size = ctx.lmem_tensor_to_size(input_shape, fmt, 1);
  auto lmem_used = input_size;
  if (do_quant) { // need quant_scale + quant_zeropoint
    lmem_used += 2 * input_size;
  }
  while (stepH > 0) {
    auto filter_shape = ctx.tl_shape_t4(1, NPU_NUM, stepH, K);
    auto output_shape = ctx.tl_shape_t4(1, NPU_NUM, stepH, 1);
    auto filter_size = ctx.lmem_tensor_to_size(filter_shape, fmt, 1);
    auto output_size = ctx.lmem_tensor_to_size(output_shape, fmt, 1);
    auto lmem_need = lmem_used + 2 * filter_size + 2 * output_size;
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    stepH--;
  }

  if (stepH == 0) {
    llvm_unreachable("tiling failed");
  }
  int pos_n = 0;
  CviBackendContext::tiling_info_t tile = {0};
  while (pos_n + NPU_NUM * stepH <= N) {
    tile.pos_c = pos_n;
    tile.w = K;
    tile.h = stepH;
    tile.c = NPU_NUM;
    tile.n = 1;
    tile.offset = pos_n * K * fmt_size;
    tiles.emplace_back(tile);
    pos_n += NPU_NUM * stepH;
  }
  int left_n = N - pos_n;
  int left_h = left_n / NPU_NUM;
  if (left_h > 0) {
    tile.pos_c = pos_n;
    tile.w = K;
    tile.h = left_h;
    tile.c = NPU_NUM;
    tile.n = 1;
    tile.offset = pos_n * K * fmt_size;
    tiles.emplace_back(tile);
    pos_n += NPU_NUM * left_h;
  }
  left_n = N - pos_n;
  if (left_n > 0) {
    tile.pos_c = pos_n;
    tile.w = K;
    tile.h = 1;
    tile.c = left_n;
    tile.n = 1;
    tile.offset = pos_n * K * fmt_size;
    tiles.emplace_back(tile);
    pos_n += left_n;
  }
}

void TgConvFcKernel::allocLmem() {
  auto &tile = tiles[0];
  auto input_shape = ctx.tl_shape_t4(1, NPU_NUM, 1, K);
  auto filter_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto output_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, 1);

  // filter
  tl_mem[0] = ctx.lmem_alloc_tensor(filter_shape, fmt, 1);
  tl_mem[1] = ctx.lmem_alloc_tensor(filter_shape, fmt, 1);
  // output
  tl_mem[2] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
  tl_mem[3] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
  // input
  tl_mem[4] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  if (do_quant) {
    tl_mem[5] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
    tl_mem[6] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  }
  // load input here
  auto input_gstride = ctx.tg_default_stride(NPU_NUM, 1, K, fmt);
  input_gstride.c = 0;
  ctx.tdma_load_stride(tl_mem[4], ga_input, input_gstride);
  if (do_quant) {
    ctx.tdma_load_stride(tl_mem[5], ga_scale, input_gstride);
    ctx.tdma_load_stride(tl_mem[6], ga_zeropoint, input_gstride);
  }
}

void TgConvFcKernel::deallocLmem() {
  int blob_num = do_quant ? 7 : 5;
  for (int i = blob_num - 1; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
}

void TgConvFcKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[step_idx % 2 + 2];
  tl_kernel = *tl_mem[4];
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_ofmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, 1);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, 1);
  tl_kernel.shape = ctx.tl_shape_t4(1, tile.c, 1, K);
  tl_kernel.stride = ctx.tl_default_stride(tl_kernel.shape, fmt, 1);
  if (do_quant) {
    tl_scale = *tl_mem[5];
    tl_zeropoint = *tl_mem[6];
    auto shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto stride = ctx.tl_default_stride(shape, fmt, 1);
    stride.h = 0; // broadcast h
    tl_scale.shape = shape;
    tl_scale.stride = stride;
    tl_zeropoint.shape = shape;
    tl_zeropoint.stride = stride;
  }
}

void TgConvFcKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  if (do_quant) {
    cvk_tg_t src = {0};
    src.start_address = ga_filter + tile.offset / 2;
    src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
    src.int8_rnd_mode = 0;
    src.fmt = CVK_FMT_I8;
    src.shape = ctx.tg_shape_t4(tile.n, tile.c, tile.h, tile.w);
    src.stride = ctx.tg_default_stride(src.shape, src.fmt);

    cvk_tdma_g2l_tensor_copy_param_t p = {0};
    p.src = &src;
    p.dst = &tl_ifmap;
    p.layer_id = layer_id;
    ctx.tdma_g2l_tensor_copy(&p);
  } else {
    ctx.tdma_load(&tl_ifmap, ga_filter + tile.offset);
  }
}

void TgConvFcKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store(&tl_ofmap, ga_output + tile.offset / K);
}

void TgConvFcKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  if (do_quant) {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_ifmap;
    p.a = &tl_ifmap;
    p.b_is_const = 0;
    p.b = &tl_scale;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    cvk_tiu_add_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_ifmap;
    p1.a_high = nullptr;
    p1.a_low = &tl_ifmap;
    p1.b_is_const = false;
    p1.b.high = nullptr;
    p1.b.low = &tl_zeropoint;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_add(&p1);
  }
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ofmap = &tl_ofmap;
  p2.ifmap = &tl_ifmap;
  p2.weight = &tl_kernel;
  p2.bias = nullptr;
  p2.ins_h = 0;
  p2.ins_last_h = 0;
  p2.ins_w = 0;
  p2.ins_last_w = 0;
  p2.pad_top = 0;
  p2.pad_bottom = 0;
  p2.pad_left = 0;
  p2.pad_right = 0;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                            // symmetric quantization
  p2.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
  ctx.tiu_pt_depthwise_convolution(&p2);
}

void TgConvFcKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    ctx.parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_bf16_convfc_kernel(const CviBackendContext &ctx,
                                       uint32_t layer_id, gaddr_t ga_input,
                                       gaddr_t ga_filter, gaddr_t ga_output,
                                       int M, int K, int N, bool do_quant,
                                       gaddr_t ga_scale, gaddr_t ga_zeropoint) {
  TgConvFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_filter, ga_output, M, K, N, do_quant,
              ga_scale, ga_zeropoint);
  kernel.selectTilePolicy();
  kernel.schedule();
}
