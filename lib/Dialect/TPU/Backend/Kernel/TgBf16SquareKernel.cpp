/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 *
 */

#include "TgBf16SquareKernel.hpp"

#define DEBUG_TYPE "TgBf16SquareKernel"

void TgBf16SquareKernel::init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
                              int32_t n, int32_t c, int32_t h, int32_t w) {

  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = 1; // fuse n and c
  this->c = n * c;
  this->h = h;
  this->w = w;
}

void TgBf16SquareKernel::allocLmem(cvk_tl_shape_t &input_shape,
                                   cvk_tl_shape_t &output_shape) {
  tl_input[0] = ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);
  tl_output[0] = ctx.lmem_alloc_tensor(output_shape, CVK_FMT_BF16, 1);
  tl_output[1] = ctx.lmem_alloc_tensor(output_shape, CVK_FMT_BF16, 1);
  assert(tl_input[0] && tl_input[1]);
  assert(tl_output[0] && tl_output[1]);
}

void TgBf16SquareKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_output[1]);
  ctx.lmem_free_tensor(tl_output[0]);
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
}

void TgBf16SquareKernel::selectTilePolicy() {
  doTileForNormalCase();
}

void TgBf16SquareKernel::doTileForNormalCase() {
  int32_t block_num = 4;
  uint32_t remain = n * c * h * w;
  uint32_t offset = 0;
  int32_t element_size = 2;
  int32_t max_h = LOCAL_MEM_SIZE / (EU_NUM * block_num * element_size);
  max_h = std::min(max_h, MAX_HEIGHT);
  assert(max_h);
  int32_t cur_h = max_h;

  cvk_tl_shape_t shape = ctx.tl_shape_t4(1, NPU_NUM, cur_h, EU_NUM);
  allocLmem(shape, shape);

  SquareTile tile;
  int32_t loop = remain / (NPU_NUM * EU_NUM * max_h);
  remain %= NPU_NUM * EU_NUM * max_h;
  for (int32_t i = 0; i < loop; i++) {
    tile.n = 1;
    tile.c = NPU_NUM;
    tile.h = cur_h;
    tile.w = EU_NUM;
    tile.input_offset = offset;
    tile.output_offset = offset;
    tiles.push_back(tile);
    offset += 1 * NPU_NUM * cur_h * EU_NUM * element_size;
  }
  if (remain) {
    int32_t cur_c = remain / (EU_NUM * cur_h);
    if (cur_c) {
      tile.n = 1;
      tile.c = cur_c;
      tile.h = cur_h;
      tile.w = EU_NUM;
      tile.input_offset = offset;
      tile.output_offset = offset;
      tiles.push_back(tile);
      offset += 1 * cur_c * cur_h * EU_NUM * element_size;
    }

    remain %= EU_NUM * cur_h;
    if (remain) {
      tile.n = 1;
      tile.c = 1;
      // memory size of neuon is align to 16 bytes, so
      // it's safe to compute more data than needed.
      tile.h = ceiling_func(remain, EU_NUM);
      tile.w = EU_NUM;
      tile.input_offset = offset;
      tile.output_offset = offset;
      tiles.push_back(tile);
    }
  }
}

void TgBf16SquareKernel::schedule() {
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

    LLVM_DEBUG(llvm::errs() << "########\n");
  }
  deallocLmem();
}

void TgBf16SquareKernel::load(int32_t step_idx, int32_t flip) {
  cvk_tl_t operand;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  operand.start_address = tl_input[1 - flip]->start_address;
  operand.shape = shape;
  operand.stride = ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);
  operand.fmt = CVK_FMT_BF16;
  cvk_tg_stride_t stride =
      ctx.tg_default_stride({shape.n, shape.c, shape.h, shape.w}, CVK_FMT_BF16);
  ctx.tdma_load_stride(&operand, ga_input + tile.input_offset, stride);

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "load[%d], flip[%d], addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
          step_idx, 1 - flip, operand.start_address, shape.n, shape.c, shape.h, shape.w,
          stride.n, stride.c, stride.h, stride.w, tile.input_offset));
}

void TgBf16SquareKernel::store(int32_t step_idx, int32_t flip) {
  cvk_tl_t result;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  result.start_address = tl_output[1 - flip]->start_address;
  result.shape = shape;
  result.stride = ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);
  result.fmt = CVK_FMT_BF16;
  cvk_tg_stride_t stride =
      ctx.tg_default_stride({shape.n, shape.c, shape.h, shape.w}, CVK_FMT_BF16);
  ctx.tdma_store_stride(&result, ga_output + tile.output_offset, stride);

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "store[%d], flip[%d], addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
          step_idx, 1 - flip, result.start_address, shape.n, shape.c, shape.h, shape.w,
          stride.n, stride.c, stride.h, stride.w, tile.output_offset));
}

void TgBf16SquareKernel::compute(int32_t step_idx, int32_t flip) {
  auto tile = tiles[step_idx];
  cvk_tl_shape_t input_shape;
  cvk_tl_shape_t output_shape;
  cvk_tl_t input;
  cvk_tl_t output;

  input_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  input.start_address = tl_input[flip]->start_address;
  input.shape = input_shape;
  input.fmt = CVK_FMT_BF16;
  input.stride = ctx.tl_default_stride(input_shape, CVK_FMT_BF16, 1);

  output_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  output.start_address = tl_output[flip]->start_address;
  output.shape = output_shape;
  output.fmt = CVK_FMT_BF16;
  output.stride = ctx.tl_default_stride(output_shape, CVK_FMT_BF16, 1);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], addr:%d, %d,input<%d,%d,%d,%d>, "
                 "output<%d,%d,%d,%d>\n",
                 step_idx, flip, flip, input.start_address, output.start_address,
                 input.shape.n, input.shape.c, input.shape.h, input.shape.w,
                 output.shape.n, output.shape.c, output.shape.h, output.shape.w));

  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &output;
  p.a = &input;
  p.b = &input;
  p.b_is_const = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = false;
  ctx.tiu_mul(&p);
}

void cvi_backend_tg_bf16_square_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                       gaddr_t ga_input, gaddr_t ga_output, int n, int c,
                                       int h, int w, bool do_relu) {

  assert(!do_relu);
  TgBf16SquareKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w);

  kernel.selectTilePolicy();
  kernel.schedule();
}