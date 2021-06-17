#include "TgTileKernel.hpp"
#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

void TgTileKernel::g2g_tile_W() {
  auto src_shape = ctx.tg_shape_t4(1, c, 1, w);
  auto src_stride = ctx.tg_default_stride(src_shape, fmt);
  src_shape.h = factor;
  src_stride.h = 0;
  auto dst_shape = src_shape;
  auto dst_stride = ctx.tg_default_stride(dst_shape, fmt);
  ctx.tdma_g2g_tensor_copy(ga_input, src_shape, src_stride, fmt, ga_output,
                           dst_shape, dst_stride, fmt);
}

void TgTileKernel::g2g_tile_N() {
  auto shape = ctx.tg_shape_t4(factor, n * c, h, w);
  auto src_stride = ctx.tg_default_stride(shape, fmt);
  auto dst_stride = src_stride;
  src_stride.n = 0;
  ctx.tdma_g2g_tensor_copy(ga_input, shape, src_stride, fmt, ga_output, shape,
                           dst_stride, fmt);
}

void TgTileKernel::init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
                        int n, int c, int h, int w, int axis, int factor,
                        cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->factor = factor;
  this->fmt = fmt;
  this->fmt_bytes = ctx.bytesize_of_fmt(fmt);
  this->mode = TILE_W;
  switch (axis) {
  case 0:
    this->n = n;
    this->c = c;
    this->h = h;
    this->w = w;
    mode = TILE_N;
    break;
  case 1:
    this->n = 1;
    this->c = n;
    this->h = 1;
    this->w = c * h * w;
    break;
  case 2:
    this->n = 1;
    this->c = n * c;
    this->h = 1;
    this->w = h * w;
    break;
  case 3:
    this->n = 1;
    this->c = n * c * h;
    this->h = 1;
    this->w = w;
    break;
  default:
    llvm_unreachable("axis is not correct");
    break;
  }
  ctx.set_layer_id(layer_id);
}

void TgTileKernel::selectTilePolicy() {
  if (mode == TILE_N) {
    return;
  }
  // tile c
  step_c = std::min(c, MAX_CHANNEL);
  while (step_c > 0) {
    auto src_shape = ctx.tl_shape_t4(1, step_c, 1, w);
    auto dst_shape = ctx.tl_shape_t4(1, step_c, factor, w);
    uint32_t src_size = ctx.lmem_tensor_to_size(src_shape, fmt, 1);
    uint32_t dst_size = ctx.lmem_tensor_to_size(dst_shape, fmt, 1);
    uint32_t lmem_need = 2 * src_size + 2 * dst_size;
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (step_c % NPU_NUM == 0) {
      step_c -= NPU_NUM;
    } else {
      step_c -= step_c % NPU_NUM;
    }
  }
  if (step_c == 0) {
    return;
  }
  tiling_t tile = {0};
  for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
    tile.c = std::min(step_c, c - tile.pos_c);
    tiles.emplace_back(tile);
  }
}

void TgTileKernel::allocLmem() {
  auto src_shape = ctx.tl_shape_t4(1, step_c, 1, w);
  auto dst_shape = ctx.tl_shape_t4(1, step_c, factor, w);
  tl_mem[0] = ctx.lmem_alloc_tensor(src_shape, fmt, 1);
  tl_mem[1] = ctx.lmem_alloc_tensor(src_shape, fmt, 1);
  tl_mem[2] = ctx.lmem_alloc_tensor(dst_shape, fmt, 1);
  tl_mem[3] = ctx.lmem_alloc_tensor(dst_shape, fmt, 1);
}

void TgTileKernel::deallocLmem() {
  for (int i = 3; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
}

void TgTileKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[2 + step_idx % 2];
  tl_ifmap.shape = ctx.tl_shape_t4(1, tile.c, 1, w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_ofmap.shape = ctx.tl_shape_t4(1, tile.c, factor, w);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, 1);
}

void TgTileKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_load(&tl_ifmap, ga_input + tile.pos_c * w * fmt_bytes);
}

void TgTileKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store(&tl_ofmap, ga_output + tile.pos_c * w * factor * fmt_bytes);
}

void TgTileKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  tl_ifmap.shape.h = factor;
  tl_ifmap.stride.h = 0;
  cvk_tiu_copy_param_t p = {0};
  p.src = &tl_ifmap;
  p.dst = &tl_ofmap;
  p.layer_id = layer_id;
  ctx.tiu_copy(&p);
}

void TgTileKernel::schedule() {
  // no tiling, just g2g
  if (mode == TILE_N) {
    g2g_tile_N();
    return;
  }
  if (tiles.empty()) {
    g2g_tile_W();
    return;
  }
  // tiling
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

void cvi_backend_tg_tile_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t input_gaddr, gaddr_t output_gaddr,
                                int n, int c, int h, int w, int axis,
                                int factor, cvk_fmt_t fmt) {
  TgTileKernel kernel(ctx);
  kernel.init(layer_id, input_gaddr, output_gaddr, n, c, h, w, axis, factor,
              fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
