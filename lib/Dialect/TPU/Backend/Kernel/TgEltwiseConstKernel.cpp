/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */

#include "TgEltwiseConstKernel.hpp"
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "TgEltwiseConstKernel"

void TgEltwiseConstKernel::init(uint32_t layer_id, gaddr_t ga_input,
                                gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w,
                                bool do_relu, float const_val, int32_t coeff,
                                int32_t rshift, std::vector<int8_t> &multiplier) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->do_relu = do_relu;
  this->rshift = rshift;
  this->multipliers = multiplier;
  this->const_val = const_val;
  this->fmt = CVK_FMT_I8;
  this->coeff = coeff;
  this->elementSize = this->fmt == CVK_FMT_I8 ? 1 : 2;
}

void TgEltwiseConstKernel::init(uint32_t layer_id, gaddr_t ga_input,
                                gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w,
                                bool do_relu, float const_val) {
                                
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->do_relu = do_relu;
  this->const_val = const_val;
  this->fmt = CVK_FMT_BF16;
  this->elementSize = this->fmt == CVK_FMT_I8 ? 1 : 2;
}

void TgEltwiseConstKernel::allocLmem() {
  tl_input[0] = ctx.lmem_alloc_tensor(load_shape, fmt, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(load_shape, fmt, 1);
  assert(tl_input[0] && tl_input[1]);
}

void TgEltwiseConstKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
}

void TgEltwiseConstKernel::selectTilePolicy() {
  int remain = n * c * h * w;
  int max_h = LOCAL_MEM_SIZE / (EU_NUM * block_num * elementSize);
  max_h = std::min(max_h, MAX_HEIGHT);
  assert(max_h);
  int32_t cur_h = max_h;

  load_shape = ctx.tl_shape_t4(1, NPU_NUM, cur_h, EU_NUM);

  // n, c, h, w -> 1, n*c, h*w, 1
  EltwiseConstTile tile;
  int32_t loop = remain / (NPU_NUM * EU_NUM * max_h);
  int offset = 0;
  remain %= NPU_NUM * EU_NUM * max_h;
  for (int32_t i = 0; i < loop; i++) {
    tile.n = 1;
    tile.c = NPU_NUM;
    tile.h = cur_h;
    tile.w = EU_NUM;
    tile.input_offset = offset;
    tiles.push_back(tile);
    offset += 1 * NPU_NUM * cur_h * EU_NUM * elementSize;
  }
  if (remain) {
    int32_t cur_c = remain / (EU_NUM * cur_h);
    if (cur_c) {
      tile.n = 1;
      tile.c = cur_c;
      tile.h = cur_h;
      tile.w = EU_NUM;
      tile.input_offset = offset;
      tiles.push_back(tile);
      offset += 1 * cur_c * cur_h * EU_NUM * elementSize;
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
      tiles.push_back(tile);
    }
  }
}

void TgEltwiseConstKernel::load(int32_t step_idx) {
   auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
   cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
   cvk_tl_t operand;
   operand.start_address = tl_input[input_idx]->start_address;
   operand.shape = shape;
   operand.stride = ctx.tl_default_stride(shape, fmt, 1);
   operand.fmt = fmt;
   ctx.tdma_load(&operand, ga_input + tile.input_offset);
   LLVM_DEBUG(
       llvm::errs() << llvm::format(
           "load[%d], shape<%d,%d,%d,%d>, global:%d -> local: %u\n",
           step_idx, tile.n, tile.c, tile.h, tile.w,
           tile.input_offset, operand.start_address));
}

void TgEltwiseConstKernel::store(int32_t step_idx) {
  auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_t result;
  result.start_address = tl_input[input_idx]->start_address;
  result.shape = shape;
  result.stride = ctx.tl_default_stride(shape, fmt, 1);
  result.fmt = fmt;
  ctx.tdma_store(&result, ga_output + tile.input_offset);
  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "store[%d], shape<%d,%d,%d,%d>, local:%u -> global: %d\n",
          step_idx, result.shape.n, result.shape.c,
          result.shape.h, result.shape.w, result.start_address,
          tile.input_offset));
}

void TgInt8EltwiseConstAddKernel::allocLmem() {
  tl_input[0] = ctx.lmem_alloc_tensor(load_shape, fmt, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(load_shape, fmt, 1);
  tl_output_h = ctx.lmem_alloc_tensor(load_shape, fmt, 1);
  assert(tl_input[0] && tl_input[1] && tl_output_h);
}

void TgInt8EltwiseConstAddKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_output_h);
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
}

void TgEltwiseConstKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; ++i) {
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


// y = R / thY * (thX / R * X + 127)
// TODO: thX / R * x is float, maybe loss accuracy
void TgInt8EltwiseConstAddKernel::compute(int32_t step_idx) {
  auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_t input, output_high;
  input.start_address = tl_input[input_idx]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  input.stride = ctx.tl_default_stride(shape, fmt, 1);

  output_high.start_address = tl_output_h->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, fmt, 1);
  output_high.fmt = fmt;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output_h<%d,%d,%d,%d:%d,%d,%d,%d> "
                 "in %u -> out %u\n",
                 step_idx, input.shape.n, input.shape.c, input.shape.h,
                 input.shape.w, input.stride.n, input.stride.c, input.stride.h,
                 input.stride.w, output_high.shape.n, output_high.shape.c,
                 output_high.shape.h, output_high.shape.w,
                 output_high.stride.n, output_high.stride.c,
                 output_high.stride.h, output_high.stride.w,
                 input.start_address, input.start_address));

  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = &output_high;
  p1.res_low = &input;
  p1.a = &input;
  p1.b_const.val = multipliers[0];
  p1.b_const.is_signed = true;
  p1.b_is_const = true;
  p1.rshift_bits = rshift;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_mul(&p1);

  cvk_tiu_add_param_t p2 = {0};
  p2.res_high = &output_high;
  p2.res_low = &input;
  p2.a_high = &output_high;
  p2.a_low = &input;
  p2.b_is_const = true;
  p2.b_const.val = static_cast<int8_t>(const_val);
  p2.b_const.is_signed = 1;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_add(&p2);

  cvk_tiu_mul_param_t p3 = {0};
  p3.res_high = &output_high;
  p3.res_low = &input;
  p3.a = &input;
  p3.b_const.val = multipliers[1];
  p3.b_const.is_signed = true;
  p3.b_is_const = true;
  p3.rshift_bits = rshift;
  p3.layer_id = layer_id;
  p3.relu_enable = do_relu;
  ctx.tiu_mul(&p3);
}

// mul
// y = n * thX / thY * x
void TgInt8EltwiseConstMulKernel::compute(int32_t step_idx) {
  auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_t input;
  input.start_address = tl_input[input_idx]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  input.stride = ctx.tl_default_stride(shape, fmt, 1);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>\n",
                 step_idx, input.shape.n, input.shape.c, input.shape.h,
                 input.shape.w, input.stride.n, input.stride.c, input.stride.h,
                 input.stride.w));

  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &input;
  p1.a = &input;
  p1.b_const.val = multipliers[0] * coeff;
  p1.b_const.is_signed = true;
  p1.b_is_const = true;
  p1.rshift_bits = rshift;
  p1.layer_id = layer_id;
  p1.relu_enable = do_relu;
  ctx.tiu_mul(&p1);
}

void TgBf16EltwiseConstAddKernel::compute(int32_t step_idx) {
  auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_t input;
  input.start_address = tl_input[input_idx]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  input.stride = ctx.tl_default_stride(shape, fmt, 1);

  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &input;
  p1.a_high = nullptr;
  p1.a_low = &input;
  p1.b_is_const = true;
  p1.b_const.val = ctx.convert_fp32_to_bf16(const_val);
  p1.b_const.is_signed = true;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = do_relu;
  ctx.tiu_add(&p1);
}

void TgBf16EltwiseConstMulKernel::compute(int32_t step_idx) {
  auto tile = tiles[step_idx];
   int input_idx = step_idx % 2;
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_t input;
  input.start_address = tl_input[input_idx]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  input.stride = ctx.tl_default_stride(shape, fmt, 1);

  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &input;
  p1.a = &input;
  p1.b_const.val = ctx.convert_fp32_to_bf16(const_val);
  p1.b_const.is_signed = true;
  p1.b_is_const = true;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = do_relu;
  ctx.tiu_mul(&p1);
}

void cvi_backend_tg_fixed_eltwise_const_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w, bool do_relu,
    float const_val, int32_t coeff, int32_t rshift, std::vector<int8_t> &multiplier) {
  TgInt8EltwiseConstAddKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, do_relu, const_val, coeff, rshift,
              multiplier);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_const_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w, bool do_relu,
    float const_val, int32_t coeff, int32_t rshift, std::vector<int8_t> &multiplier) {
  TgInt8EltwiseConstMulKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, do_relu, const_val, coeff, rshift,
              multiplier);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_const_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w, bool do_relu,
    float const_val) {
  TgBf16EltwiseConstAddKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, do_relu, const_val);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_const_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w, bool do_relu,
    float const_val) {
  TgBf16EltwiseConstMulKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, do_relu, const_val);
  kernel.selectTilePolicy();
  kernel.schedule();
}