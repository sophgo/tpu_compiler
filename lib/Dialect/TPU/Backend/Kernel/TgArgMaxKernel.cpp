/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 *
 */

#include "TgArgMaxKernel.hpp"
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "TgArgMaxKernel"

void TgArgMaxKernel::init(uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int32_t _n, int32_t _c, int32_t _h, int32_t _w,
    int32_t w_tile_size, cvk_fmt_t fmt) {

  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = 1;   // fuse n and c
  this->c = _n * _c;
  this->h = 1;
  this->w = _h * _w;
  this->oh = (this->w + w_tile_size - 1) / w_tile_size;
  this->ow = 1;
  this->w_tile_size = w_tile_size;
  this->fmt = fmt;
}

void TgArgMaxKernel::allocLmem(cvk_tl_shape_t &input_shape,
                                    cvk_tl_shape_t &output_shape) {
  tl_input[0] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  tl_output[0] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
  tl_output[1] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
  assert(tl_input[0] && tl_input[1]);
  assert(tl_output[0] && tl_output[1]);
}

void TgArgMaxKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_output[1]);
  ctx.lmem_free_tensor(tl_output[0]);
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
}

void TgArgMaxKernel::selectTilePolicy() {
  doTileForNormalCase();
}

void TgArgMaxKernel::doTileForNormalCase() {
  int32_t step_c, step_oh;

  // determin the shape of tile.
  for (step_oh = std::min(oh, 1); step_oh > 0; step_oh--) {
    for (step_c = std::min(c, MAX_CHANNEL); step_c > 0; step_c -= NPU_NUM) {
      if (step_c != c) {
        step_c = align_up(step_c, NPU_NUM);
      }
      cvk_tl_shape_t input_shape = ctx.tl_shape_t4(1, step_c, step_oh, w_tile_size);
      cvk_tl_shape_t output_shape = ctx.tl_shape_t4(1, step_c, step_oh, 1);
      auto total_lmem = 2 * (ctx.lmem_tensor_to_size(input_shape, fmt, 1) +
                              ctx.lmem_tensor_to_size(output_shape, fmt, 1));
      if (total_lmem <= (uint32_t)LOCAL_MEM_SIZE) {
        allocLmem(input_shape, output_shape);
        goto do_tile;
      }
    }
  }
  assert(0 && "failed to split");

do_tile:
  LLVM_DEBUG(llvm::errs() << "shape:" << n << "," << c << "," << h << "," << w
              << " oh:" << oh << ", ow:" << ow
              << " step_c:" << step_c << " step_oh:" << step_oh << "\n");

  ArgMaxTile tile;
  for (int c_pos = 0; c_pos < c; c_pos += step_c) {
    auto cur_c = std::min(c - c_pos, step_c);
    for (int iw_pos = 0, oh_pos = 0;
         oh_pos < oh;
         oh_pos += step_oh, iw_pos += w_tile_size) {
      int32_t cur_oh = std::min(oh - oh_pos, step_oh);
      int32_t cur_iw = std::min(w - iw_pos, w_tile_size);
      int32_t cur_ih = cur_oh;
      tile.n = 1;
      tile.c = cur_c;
      tile.h = cur_ih;
      tile.w = cur_iw;
      tile.oh = cur_oh;
      tile.ow = 1;
      tile.c_pos = c_pos;
      tile.iw_pos = iw_pos;
      tile.oh_pos = oh_pos;
      tile.input_offset = (c_pos * w + iw_pos) * ctx.bytesize_of_fmt(fmt);
      tile.output_offset = (c_pos * oh + oh_pos) * ctx.bytesize_of_fmt(fmt);
      tiles.push_back(tile);
    }
  }
}

void TgArgMaxKernel::schedule() {
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1, flip);
    }
    if (i < total_steps) {
      load(i, flip);
    }
    if (i - 2 >= 0) {
      store(i - 2, flip);
    }
    flip = 1 - flip;
    ctx.parallel_disable();
  }
  deallocLmem();
}

void TgArgMaxKernel::load(int32_t step_idx, int32_t flip) {
  cvk_tl_t operand;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  operand.start_address = tl_input[1 - flip]->start_address;
  operand.shape = shape;
  operand.stride = ctx.tl_default_stride(shape, fmt, 1);
  operand.fmt = fmt;
  int unit_sz = ctx.bytesize_of_fmt(fmt);
  cvk_tg_stride_t stride = {
    (uint32_t)(c * w * unit_sz),
    (uint32_t)(w * unit_sz),
    (uint32_t)(w_tile_size * unit_sz), 1
  };
  ctx.tdma_load_stride(&operand, ga_input + tile.input_offset, stride);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "load[%d], flip[%d], addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
      step_idx, 1 - flip, operand.start_address, shape.n, shape.c, shape.h, shape.w,
      stride.n, stride.c, stride.h, stride.w,
      tile.input_offset));
}

void TgArgMaxKernel::store(int32_t step_idx, int32_t flip) {
  cvk_tl_t result;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.oh, tile.ow);
  result.start_address = tl_output[1 - flip]->start_address;
  result.shape = shape;
  result.stride = ctx.tl_default_stride(shape, fmt, 1);
  result.fmt = fmt;
  int unit_sz = ctx.bytesize_of_fmt(fmt);
  cvk_tg_stride_t stride = {
    (uint32_t)(c * oh * ow * unit_sz),
    (uint32_t)(oh * ow * unit_sz),
    (uint32_t)(ow * unit_sz), 1
  };
  ctx.tdma_store_stride(&result, ga_output + tile.output_offset, stride);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "store[%d], flip[%d], addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
      step_idx, 1 - flip, result.start_address, shape.n, shape.c, shape.h, shape.w,
      stride.n, stride.c, stride.h, stride.w,
      tile.output_offset));
}

void TgArgMaxKernel::compute(int32_t step_idx, int32_t flip) {
  auto tile = tiles[step_idx];
  cvk_tl_shape_t input_shape;
  cvk_tl_shape_t output_shape;
  cvk_tl_t input;
  cvk_tl_t output;

  input_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  input.start_address = tl_input[flip]->start_address;
  input.shape = input_shape;
  input.fmt = fmt;
  input.stride = ctx.tl_default_stride(input_shape, fmt, 1);

  output_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.oh, tile.ow);
  output.start_address = tl_output[flip]->start_address;
  output.shape = output_shape;
  output.fmt = fmt;
  output.stride = ctx.tl_default_stride(output_shape, fmt, 1);

  cvk_tiu_max_pooling_param_t param = {0};
  param.ofmap = &output;
  param.ifmap = &input;
  param.kh = 1;
  param.kw = tile.w;
  param.pad_top = 0;
  param.pad_bottom = 0;
  param.pad_left = 0;
  param.pad_right = 0;
  param.stride_h = 1;
  param.stride_w = 1;
  param.layer_id = layer_id;
  param.ins_val = -128;
  param.ins_fp = 0xff7f;
  ctx.tiu_max_pooling(&param);
}

void cvi_backend_tg_argmax_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w, int w_tile_size, cvk_fmt_t fmt) {

  TgArgMaxKernel kernel(ctx);
  kernel.init(
      layer_id, ga_input, ga_output, n, c, h, w, w_tile_size, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
