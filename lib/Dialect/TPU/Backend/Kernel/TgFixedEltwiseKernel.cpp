/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: eltwise_bmkernel.cpp
 * Description:
 */

#include "TgFixedEltwiseKernel.hpp"

#define DEBUG_TYPE "TgInt8EltwiseKernel"

void TgInt8EltwiseKernel::init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs) {
  this->layer_id = layer_id;
  this->ga_inputs = ga_inputs;
  this->ga_output = ga_output;
  this->operand_num = operand_num;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->do_relu = do_relu;
  this->do_early_stride = do_early_stride;
  this->stride_h = do_early_stride ? stride_h : 1;
  this->stride_w = do_early_stride ? stride_w : 1;
  this->rshift = rshift;
  this->multipliers = multipliers;
  this->coeffs = coeffs;
}

void TgInt8EltwiseKernel::allocLmem(
    cvk_tl_shape_t &input_shape,
    cvk_tl_shape_t &output_shape) {
  tl_input[0] = ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);
  tl_output[0] = ctx.lmem_alloc_tensor(output_shape, CVK_FMT_I8, 1);
  tl_output[1] = ctx.lmem_alloc_tensor(output_shape, CVK_FMT_I8, 1);
  tl_output_h = ctx.lmem_alloc_tensor(output_shape, CVK_FMT_I8, 1);
  assert(tl_input[0] && tl_input[1]);
  assert(tl_output[0] && tl_output[1]);
  assert(tl_output_h);
}

void TgInt8EltwiseKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_output_h);
  ctx.lmem_free_tensor(tl_output[1]);
  ctx.lmem_free_tensor(tl_output[0]);
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
}

void TgInt8EltwiseKernel::selectTilePolicy() {
  if (do_early_stride) {
    doTileForStrideCase();
  } else {
    doTileForNormalCase();
  }
}

void TgInt8EltwiseKernel::doTileForNormalCase() {
  int32_t block_num = 5;
  uint32_t remain = n * c * h * w;
  uint32_t offset = 0;
  int32_t max_h = LOCAL_MEM_SIZE / (EU_NUM * block_num);
  max_h = std::min(max_h, MAX_HEIGHT);
  assert(max_h);
  int32_t cur_h = max_h;

  cvk_tl_shape_t shape = ctx.shape_t4(1, NPU_NUM, cur_h, EU_NUM);
  allocLmem(shape, shape);

  EltwiseTile tile;
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
    offset += 1 * NPU_NUM * cur_h * EU_NUM;
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
      offset += 1 * cur_c * cur_h * EU_NUM;
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

void TgInt8EltwiseKernel::doTileForStrideCase() {
  int n_step = 1;
  int c_step = std::min(c, NPU_NUM);
  int h_step = h / stride_h;
  cvk_tl_shape_t input_shape = ctx.shape_t4(n_step, c_step, h_step, w);
  cvk_tl_shape_t output_shape = ctx.shape_t4(n_step, c_step, h_step, w / stride_w);
  uint32_t lmem_required = ctx.lmem_tensor_to_size(input_shape, CVK_FMT_I8, 1) * 2 +
                           ctx.lmem_tensor_to_size(output_shape, CVK_FMT_I8, 1) * 3;
  if (lmem_required > (uint32_t)LOCAL_MEM_SIZE) {
    for (; h_step > 0; --h_step) {
      input_shape = ctx.shape_t4(n_step, c_step, h_step, w);
      output_shape = ctx.shape_t4(n_step, c_step, h_step, w / stride_w);
      lmem_required = ctx.lmem_tensor_to_size(input_shape, CVK_FMT_I8, 1) * 2 +
                      ctx.lmem_tensor_to_size(output_shape, CVK_FMT_I8, 1) * 3;
      if (lmem_required <= (uint32_t)LOCAL_MEM_SIZE) {
        break;
      }
    }
  }
  assert(lmem_required <= (uint32_t)LOCAL_MEM_SIZE);
  allocLmem(input_shape, output_shape);

  EltwiseTile tile;
  for (int n_pos = 0; n_pos < n; n_pos += n_step) {
    int cur_n = std::min(n - n_pos, n_step);
    for (int c_pos = 0; c_pos < c; c_pos += c_step) {
      int cur_c = std::min(c - c_pos, c_step);
      for (int h_pos = 0; h_pos < (h / stride_h); h_pos += h_step) {
        int cur_h = std::min((h / stride_h) - h_pos, h_step);
        tile.n = cur_n;
        tile.c = cur_c;
        tile.h = cur_h;
        tile.w = w;
        tile.input_offset = n_pos * c * h * w + c_pos * h * w + (h_pos * stride_h) * w;
        tile.output_offset = n_pos * c * (h / stride_h) * (w / stride_w) +
                             c_pos * (h / stride_h) * (w / stride_w) +
                             h_pos * (w / stride_w);
        tiles.push_back(tile);
      }
    }
  }
}

void TgInt8EltwiseKernel::schedule() {
  int32_t total_steps = tiles.size() * operand_num;
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if ((i - 2) >= 0 &&
        (i - 2) % operand_num == operand_num - 1) {
      store(i - 2);
    }

    ctx.parallel_disable();

    LLVM_DEBUG(llvm::errs() << "########\n");
  }
  deallocLmem();
}

void TgInt8EltwiseKernel::load(int32_t step_idx) {
  cvk_tl_t operand;
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];
  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w);
  operand.start_address = tl_input[input_flip]->start_address;
  operand.shape = shape;
  operand.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  operand.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tg_stride_t stride = {(uint32_t)(c * h * w), (uint32_t)(h * w), (uint32_t)(stride_h * w), 1};
    ctx.tdma_load_stride(&operand,
        ga_inputs[opd_idx] + tile.input_offset, stride);

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "load[%d], flip[%d], shape<%d,%d,%d,%d>,"
        " stride:<%d,%d,%d,%d>, offset:%d\n",
        step_idx, input_flip, tile.n, tile.c, tile.h, tile.w,
        stride.n, stride.c, stride.h, stride.w,
        tile.input_offset));
  } else {
    ctx.tdma_load(&operand, ga_inputs[opd_idx] + tile.input_offset);

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "load[%d], flip[%d], shape<%d,%d,%d,%d>, offset:%d\n",
        step_idx, input_flip, tile.n, tile.c,
        tile.h, tile.w, tile.input_offset));
  }
  input_flip = 1 - input_flip;
}

void TgInt8EltwiseKernel::store(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto tile = tiles[tile_idx];
  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);
  cvk_tl_t result;
  result.start_address = tl_output[1 - output_flip]->start_address;
  result.shape = shape;
  result.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  result.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tg_stride_t stride = {
      (uint32_t)(c * (h / stride_h) * (w / stride_w)),
      (uint32_t)((h / stride_h) * (w / stride_w)),
      (uint32_t)(w / stride_w), 1
    };
    ctx.tdma_store_stride(&result, ga_output + tile.output_offset, stride);

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "store[%d], flip[%d], shape<%d,%d,%d,%d>,"
        " stride<%d,%d,%d,%d>, offset:%d\n",
        step_idx, 1 - output_flip,
        result.shape.n, result.shape.c,
        result.shape.h, result.shape.w,
        stride.n, stride.c, stride.h, stride.w,
        tile.output_offset));
  } else {
    ctx.tdma_store(&result, ga_output + tile.output_offset);

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "store[%d], flip[%d], shape<%d,%d,%d,%d>, offset:%d\n",
        step_idx, 1 - output_flip,
        result.shape.n, result.shape.c,
        result.shape.h, result.shape.w,
        tile.output_offset));
  }
}

void TgInt8EltwiseAddKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
      "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
      step_idx, 1 - input_flip, output_flip,
      input.shape.n, input.shape.c, input.shape.h, input.shape.w,
      input.stride.n, input.stride.c, input.stride.h, input.stride.w,
      output.shape.n, output.shape.c, output.shape.h, output.shape.w,
      output.stride.n, output.stride.c, output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // calculate first input.
    cvk_tiu_mul_param_t p = {0};
    p.res_high = &output_high;
    p.res_low = &output;
    p.a = &input;
    p.b_const.val = multipliers[opd_idx] * coeffs[opd_idx];
    p.b_const.is_signed = true;
    p.b_is_const = true;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  } else if (opd_idx != operand_num - 1) {
    // calculate inputs in middle.
    cvk_tiu_mac_param_t p = {0};
    p.res_high = &output_high;
    p.res_low = &output;
    p.a = &input;
    p.res_is_int8 = false;
    p.b_const.val = multipliers[opd_idx] * coeffs[opd_idx];
    p.b_is_const = 1;
    p.b_const.is_signed = true;
    p.lshift_bits = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mac(&p);
  } else { // calculate last input.
    cvk_tiu_mac_param_t p = {0};
    p.res_high = &output_high;
    p.res_low = &output;
    p.a = &input;
    p.res_is_int8 = true;
    p.b_const.val = multipliers[opd_idx] * coeffs[opd_idx];
    p.b_is_const = 1;
    p.b_const.is_signed = true;
    p.lshift_bits = 0;
    p.rshift_bits = rshift;
    p.layer_id = layer_id;
    p.relu_enable = do_relu;
    ctx.tiu_mac(&p);
    output_flip = 1 - output_flip;
  }
}

void TgInt8EltwiseMaxKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
      "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
      step_idx, 1 - input_flip, output_flip,
      input.shape.n, input.shape.c, input.shape.h, input.shape.w,
      input.stride.n, input.stride.c, input.stride.h, input.stride.w,
      output.shape.n, output.shape.c, output.shape.h, output.shape.w,
      output.stride.n, output.stride.c, output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // calculate first input.
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &output;
    p.a = &input;
    p.b_const.val = multipliers[opd_idx] * coeffs[opd_idx];
    p.b_const.is_signed = true;
    p.b_is_const = true;
    p.rshift_bits = rshift;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  } else { // calculate last input.
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &output_high;
    p1.a = &input;
    p1.b_const.val = multipliers[1] * coeffs[1];
    p1.b_const.is_signed = true;
    p1.b_is_const = true;
    p1.rshift_bits = rshift;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    cvk_tiu_max_param_t p2 = {0};
    p2.max = &output;
    p2.a = &output_high;
    p2.b = &output;
    p2.b_is_const = false;
    p2.layer_id = layer_id;
    ctx.tiu_max(&p2);

    if (do_relu) {
      cvk_tiu_max_param_t p2 = {0};
      p2.max = &output;
      p2.a = &output;
      p2.b_is_const = true;
      p2.b_const.val = (0);
      p2.b_const.is_signed = 1;
      p2.layer_id = layer_id;
      ctx.tiu_max(&p2);
    }
    output_flip = 1 - output_flip;
  }
}

void TgInt8EltwiseMinKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
      "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
      step_idx, 1 - input_flip, output_flip,
      input.shape.n, input.shape.c, input.shape.h, input.shape.w,
      input.stride.n, input.stride.c, input.stride.h, input.stride.w,
      output.shape.n, output.shape.c, output.shape.h, output.shape.w,
      output.stride.n, output.stride.c, output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // calculate first input.
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &output;
    p.a = &input;
    p.b_const.val = multipliers[opd_idx] * coeffs[opd_idx];
    p.b_const.is_signed = true;
    p.b_is_const = true;
    p.rshift_bits = rshift;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  } else { // calculate last input.
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &output_high;
    p1.a = &input;
    p1.b_const.val = multipliers[1] * coeffs[1];
    p1.b_const.is_signed = true;
    p1.b_is_const = true;
    p1.rshift_bits = rshift;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    cvk_tiu_min_param_t p2 = {0};
    p2.min = &output;
    p2.a = &output_high;
    p2.b = &output;
    p2.b_is_const = false;
    p2.layer_id = layer_id;
    ctx.tiu_min(&p2);

    if (do_relu) {
      cvk_tiu_max_param_t p2 = {0};
      p2.max = &output;
      p2.a = &output;
      p2.b_is_const = true;
      p2.b_const.val = (0);
      p2.b_const.is_signed = 1;
      p2.layer_id = layer_id;
      ctx.tiu_max(&p2);
    }
    output_flip = 1 - output_flip;
  }
}

void TgInt8EltwiseMulKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
      "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
      step_idx, 1 - input_flip, output_flip,
      input.shape.n, input.shape.c, input.shape.h, input.shape.w,
      input.stride.n, input.stride.c, input.stride.h, input.stride.w,
      output.shape.n, output.shape.c, output.shape.h, output.shape.w,
      output.stride.n, output.stride.c, output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // calculate first input.
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &output;
    p.a = &input;
    p.b_const.val = 1;
    p.b_const.is_signed = true;
    p.b_is_const = true;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  } else {
    cvk_tiu_mul_qm_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &output;
    p1.a = &input;
    p1.b_is_const = 0;
    p1.b = &output;
    p1.rshift_bits = rshift;
    p1.relu_enable = do_relu;
    p1.layer_id = layer_id;
    p1.multiplier = multipliers[0];
    ctx.tiu_mul_qm(&p1);
    output_flip = 1 - output_flip;
  }
}

void cvi_backend_tg_fixed_eltwise_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs) {

  TgInt8EltwiseAddKernel kernel(ctx);
  kernel.init(
    layer_id, ga_inputs, ga_output,
    operand_num, n, c, h, w, do_relu, do_early_stride,
    stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_max_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMaxKernel kernel(ctx);
  kernel.init(
    layer_id, ga_inputs, ga_output,
    operand_num, n, c, h, w, do_relu, do_early_stride,
    stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_min_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMinKernel kernel(ctx);
  kernel.init(
    layer_id, ga_inputs, ga_output,
    operand_num, n, c, h, w, do_relu, do_early_stride,
    stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu, bool do_early_stride,
    int32_t stride_h, int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMulKernel kernel(ctx);
  kernel.init(
    layer_id, ga_inputs, ga_output,
    operand_num, n, c, h, w, do_relu, do_early_stride,
    stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}
