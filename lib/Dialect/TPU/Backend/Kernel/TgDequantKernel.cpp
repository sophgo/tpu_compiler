/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgDequantKernel.cpp
 * Description:
 */

#include "TgDequantKernel.hpp"
#include "backend/backend_tl_api.h"
#include <numeric>
#define DEBUG_TYPE "dequant_kernel"

void TgDequantKernel::reshape() {
  int shape[] = {n, c, h, w};
  assert(axis >= 0 && axis < 4);
  int outer_size =
      std::accumulate(shape, shape + axis, 1, std::multiplies<int>());
  int inner_size =
      std::accumulate(shape + axis + 1, shape + 4, 1, std::multiplies<int>());
  axis_dim = shape[axis];
  if (inner_size == 1) {
    n = 1;
    c = outer_size;
    h = 1;
    w = axis_dim;
    mode = AXIS_W;
    return;
  }
  n = outer_size;
  c = axis_dim;
  h = 1;
  w = inner_size;
  mode = AXIS_C;
  return;
}

void TgDequantKernel::init(uint32_t layer_id, gaddr_t ga_input,
                           gaddr_t ga_scale, gaddr_t ga_zeropoint,
                           gaddr_t ga_output, int axis, int n, int c, int h,
                           int w) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_scale = ga_scale;
  this->ga_zeropoint = ga_zeropoint;
  this->axis = axis;
  this->fmt = CVK_FMT_BF16;
  reshape();
  ctx.set_layer_id(layer_id);
}

void TgDequantKernel::tiling_axis_w() {
  int c_step = std::min(c, MAX_CHANNEL);
  auto scale_shape = ctx.tl_shape_t4(1, NPU_NUM, 1, axis_dim);
  auto scale_size = ctx.lmem_tensor_to_size(scale_shape, fmt, 1);
  auto zp_size = scale_size;
  while (c_step > 0) {
    auto i_s = ctx.tl_shape_t4(1, c_step, 1, axis_dim);
    auto input_size = ctx.lmem_tensor_to_size(i_s, fmt, 1);
    uint32_t lmem_need = scale_size + zp_size + 2 * input_size;
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (c_step % NPU_NUM) {
      c_step -= c_step % NPU_NUM;
    } else {
      c_step -= NPU_NUM;
    }
  }
  if (c_step == 0) {
    llvm_unreachable("tiling by axis w failed");
  }
  CviBackendContext::tiling_info_t tile = {0};
  tile.n = 1;
  tile.pos_n = 0;
  tile.h = 1;
  tile.pos_h = 0;
  tile.w = axis_dim;
  tile.pos_w = 0;
  for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += c_step) {
    tile.c = std::min(c_step, c - tile.pos_c);
    tile.offset = tile.pos_c * axis_dim;
    tiles.push_back(tile);
  }
}

void TgDequantKernel::selectTilePolicy() {
  switch (mode) {
  case AXIS_W:
    tiling_axis_w();
    return;
  default:
    llvm_unreachable("Support later");
    return;
  }
}

void TgDequantKernel::allocLmem() {
  auto &tile = tiles[0];
  auto scale_shape = ctx.tl_shape_t4(1, NPU_NUM, 1, axis_dim);
  auto zp_shape = ctx.tl_shape_t4(1, NPU_NUM, 1, axis_dim);
  auto input_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_mem[0] = ctx.lmem_alloc_tensor(scale_shape, fmt, 1);
  tl_mem[1] = ctx.lmem_alloc_tensor(zp_shape, fmt, 1);
  tl_mem[2] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  tl_mem[3] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  auto gstride = ctx.tg_default_stride(NPU_NUM, 1, axis_dim, fmt);
  gstride.c = 0;
  ctx.tdma_load_stride(tl_mem[0], ga_scale, gstride);
  ctx.tdma_load_stride(tl_mem[1], ga_zeropoint, gstride);
}

void TgDequantKernel::deallocLmem() {
  for (int i = 3; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
}

void TgDequantKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_scale = *tl_mem[0];
  tl_zeropoint = *tl_mem[1];
  tl_ifmap = *tl_mem[step_idx % 2 + 2];
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_scale.shape.c = tile.c;
  tl_scale.stride.c = 0;
  tl_zeropoint.shape.c = tile.c;
  tl_zeropoint.stride.c = 0;
}

void TgDequantKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);

  cvk_tg_t src = {0};
  src.start_address = ga_input + tile.offset;
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
}

void TgDequantKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store(&tl_ifmap, ga_output + tile.offset * ctx.bytesize_of_fmt(fmt));
}

void TgDequantKernel::compute(int32_t step_idx) {
  refresh(step_idx);
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

void TgDequantKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();
    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    if (i < total_steps) {
      load(i);
    }
    ctx.parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_dequant_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_scale, gaddr_t ga_zeropoint,
                                   gaddr_t ga_output, int axis, int n, int c,
                                   int h, int w) {
  TgDequantKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_scale, ga_zeropoint, ga_output, axis, n, c,
              h, w);
  kernel.selectTilePolicy();
  kernel.schedule();
}
