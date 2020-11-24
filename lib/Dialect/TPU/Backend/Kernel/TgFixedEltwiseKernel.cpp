/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 * refined 2020-10-12
 */

#include "TgFixedEltwiseKernel.hpp"

#define DEBUG_TYPE "TgEltwiseKernel"

void TgEltwiseKernel::init(uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
                           int32_t operand_num, int32_t n, int32_t c, int32_t h,
                           int32_t w, bool do_relu, bool do_early_stride,
                           int32_t stride_h, int32_t stride_w, int32_t rshift,
                           const int32_t *multipliers, const int32_t *coeffs,
                           int32_t *inputs_offset, int32_t output_offset) {
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
  this->coeffs_float = nullptr;
  this->fmt = CVK_FMT_I8;
  this->elementSize = this->fmt == CVK_FMT_I8 ? 1 : 2;

  this->inputs_offset = inputs_offset;
  this->output_offset = output_offset;
  if (this->output_offset) {
    this->is_asymmetric = true;
    assert(do_relu == false); // asymmetric no need fused relu
  }
}

void TgEltwiseKernel::init(uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
                           int32_t operand_num, int32_t n, int32_t c, int32_t h,
                           int32_t w, bool do_relu, bool do_early_stride,
                           int32_t stride_h, int32_t stride_w, const float *coeffs) {
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
  this->coeffs_float = coeffs;
  this->coeffs = nullptr;
  this->fmt = CVK_FMT_BF16;
  this->elementSize = this->fmt == CVK_FMT_I8 ? 1 : 2;
}

void TgEltwiseKernel::allocLmem(cvk_tl_shape_t &input_shape,
                                cvk_tl_shape_t &output_shape) {
  tl_input[0] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
  if (dynamic_cast<TgBf16EltwiseMinMaxKernel*>(this) && fmt == CVK_FMT_BF16) {
    tl_output[0] = tl_input[0];
    tl_output[1] = tl_input[1];
    tl_output_h[0] = nullptr;
  }
  else {
    tl_output[0] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    tl_output[1] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    tl_output_h[0] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    assert(tl_output_h[0]);
  }
  assert(tl_input[0] && tl_input[1]);
  assert(tl_output[0] && tl_output[1]);

  if (is_asymmetric) {
    tl_output_h[1] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    tl_working[0] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    tl_working[1] = ctx.lmem_alloc_tensor(output_shape, fmt, 1);
    assert(tl_output_h[1] && tl_working[0] && tl_working[1]);
  }
}

void TgEltwiseKernel::deallocLmem() {
  if (is_asymmetric) {
    ctx.lmem_free_tensor(tl_working[1]);
    ctx.lmem_free_tensor(tl_working[0]);
    ctx.lmem_free_tensor(tl_output_h[1]);
  }
  if (dynamic_cast<TgBf16EltwiseMinMaxKernel*>(this) && fmt == CVK_FMT_BF16) {
    // no allocate output
    ctx.lmem_free_tensor(tl_input[1]);
    ctx.lmem_free_tensor(tl_input[0]);
  }
  else {
    ctx.lmem_free_tensor(tl_output_h[0]);
    ctx.lmem_free_tensor(tl_output[1]);
    ctx.lmem_free_tensor(tl_output[0]);
    ctx.lmem_free_tensor(tl_input[1]);
    ctx.lmem_free_tensor(tl_input[0]);
  }

}

void TgEltwiseKernel::selectTilePolicy() {
  if (do_early_stride) {
    doTileForStrideCase();
  } else {
    doTileForNormalCase();
  }
}

void TgEltwiseKernel::doTileForNormalCase() {
  int32_t block_num = 5;

  if (dynamic_cast<TgBf16EltwiseMinMaxKernel*>(this) && fmt == CVK_FMT_BF16) {
    block_num = 2; // 2 for ping pong buffer and reuse activation
  }

  if (this->is_asymmetric) {
    // In asymmetric
    // we need more space
    // one output_h, one working_high, one working_low
    block_num += 3;
  }

  uint32_t remain = n * c * h * w;
  uint32_t offset = 0;
  int32_t max_h = LOCAL_MEM_SIZE / (EU_NUM * block_num * elementSize);
  max_h = std::min(max_h, MAX_HEIGHT);
  assert(max_h);
  int32_t cur_h = max_h;

  cvk_tl_shape_t shape = ctx.tl_shape_t4(1, NPU_NUM, cur_h, EU_NUM);
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
      tile.output_offset = offset;
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
      tile.output_offset = offset;
      tiles.push_back(tile);
    }
  }
}

void TgEltwiseKernel::doTileForStrideCase() {
  int n_step = 1;
  int c_step = std::min(c, NPU_NUM);
  int h_step = h / stride_h;
  cvk_tl_shape_t input_shape = ctx.tl_shape_t4(n_step, c_step, h_step, w);
  cvk_tl_shape_t output_shape = ctx.tl_shape_t4(n_step, c_step, h_step, w / stride_w);
  uint32_t lmem_required = ctx.lmem_tensor_to_size(input_shape, fmt, 1) * 2 +
                           ctx.lmem_tensor_to_size(output_shape, fmt, 1) * 3;
  if (lmem_required > (uint32_t)LOCAL_MEM_SIZE) {
    for (; h_step > 0; --h_step) {
      input_shape = ctx.tl_shape_t4(n_step, c_step, h_step, w);
      output_shape = ctx.tl_shape_t4(n_step, c_step, h_step, w / stride_w);
      lmem_required = ctx.lmem_tensor_to_size(input_shape, fmt, 1) * 2 +
                      ctx.lmem_tensor_to_size(output_shape, fmt, 1) * 3;
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
        tile.input_offset *= elementSize;
        tile.output_offset = n_pos * c * (h / stride_h) * (w / stride_w) +
                             c_pos * (h / stride_h) * (w / stride_w) +
                             h_pos * (w / stride_w);
        tile.output_offset *= elementSize;
        tiles.push_back(tile);
      }
    }
  }
}

void TgEltwiseKernel::schedule() {
  int32_t total_steps = tiles.size() * operand_num;
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if ((i - 2) >= 0 && (i - 2) % operand_num == operand_num - 1) {
      store(i - 2);
    }
    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }

    ctx.parallel_disable();

    LLVM_DEBUG(llvm::errs() << "########\n");
  }
  deallocLmem();
}

void TgEltwiseKernel::load(int32_t step_idx) {
  cvk_tl_t operand;
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  operand.start_address = tl_input[input_flip]->start_address;
  operand.shape = shape;
  operand.stride = ctx.tl_default_stride(shape, fmt, 1);
  operand.fmt = fmt;
  if (do_early_stride) {
    cvk_tg_stride_t stride = {(uint32_t)(c * h * w * elementSize),
                              (uint32_t)(h * w * elementSize),
                              (uint32_t)(stride_h * w * elementSize), elementSize};
    ctx.tdma_load_stride(&operand, ga_inputs[opd_idx] + tile.input_offset, stride);

    LLVM_DEBUG(llvm::errs() << llvm::format("load[%d], flip[%d], shape<%d,%d,%d,%d>,"
                                            " stride:<%d,%d,%d,%d>, offset:%d\n",
                                            step_idx, input_flip, tile.n, tile.c, tile.h,
                                            tile.w, stride.n, stride.c, stride.h,
                                            stride.w, tile.input_offset));
  } else {
    ctx.tdma_load(&operand, ga_inputs[opd_idx] + tile.input_offset);

    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "load[%d], flip[%d], shape<%d,%d,%d,%d>, global:%d -> local: %u\n", step_idx,
                   input_flip, tile.n, tile.c, tile.h, tile.w, tile.input_offset, operand.start_address));
  }
  input_flip = 1 - input_flip;
}

void TgEltwiseKernel::store(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto tile = tiles[tile_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);
  cvk_tl_t result;

  result.start_address = tl_output[1 - output_flip]->start_address;
  result.shape = shape;
  result.stride = ctx.tl_default_stride(shape, fmt, 1);
  result.fmt = fmt;
  if (do_early_stride) {
    cvk_tg_stride_t stride = {
        (uint32_t)(c * (h / stride_h) * (w / stride_w) * elementSize),
        (uint32_t)((h / stride_h) * (w / stride_w) * elementSize),
        (uint32_t)(w / stride_w * elementSize), elementSize};
    ctx.tdma_store_stride(&result, ga_output + tile.output_offset, stride);

    LLVM_DEBUG(llvm::errs() << llvm::format("store[%d], flip[%d], shape<%d,%d,%d,%d>,"
                                            " stride<%d,%d,%d,%d>, offset:%d\n",
                                            step_idx, 1 - output_flip, result.shape.n,
                                            result.shape.c, result.shape.h,
                                            result.shape.w, stride.n, stride.c, stride.h,
                                            stride.w, tile.output_offset));
  } else {
    ctx.tdma_store(&result, ga_output + tile.output_offset);
    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "store[%d], flip[%d], shape<%d,%d,%d,%d>, local:%u -> global: %d\n", step_idx,
                   1 - output_flip, result.shape.n, result.shape.c, result.shape.h,
                   result.shape.w, result.start_address, tile.output_offset));
  }
}

void TgInt8EltwiseAddKernel::symmetric_compute(const int opd_idx,
                                               cvk_tl_t &input,
                                               cvk_tl_t &output,
                                               cvk_tl_t &output_high) {
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

// In Asymmetric Add,
// Qy = Sx/Sy(Qx1-Zx1) + Sx/Sy(Qx2-Zx2) + Zy
// we make Sx/Sy = 2^rshift * multiplier(8bit)

// output = [((Qx1+offset1) * multiplier1 + ((Qx2 + offset1) * multiplier2)...]
// >> rshift) + output_offset

void TgInt8EltwiseAddKernel::asymmetric_compute(const int opd_idx,
                                                cvk_tl_t &input,
                                                cvk_tl_t &output,
                                                cvk_tl_t &output_high,
                                                bool last_step) {

  reset(tl_working[0]);
  reset(tl_working[1]);
  // make input from int8 to int16, use mac
  // put tmp result to tl_working space
  cvk_tiu_mac_param_t p = {0};
  p.res_high = tl_working[1];
  p.res_low = tl_working[0];
  p.a = &input;
  p.res_is_int8 = false;
  p.b_const.val = 1;
  p.b_is_const = 1;
  p.b_const.is_signed = true;
  p.lshift_bits = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mac(&p);

  // add input offset, and replace tl_working
  // (Qx-Zx) 16bit
  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = tl_working[1];
  p1.res_low = tl_working[0];
  p1.a_high = tl_working[1];
  p1.a_low = tl_working[0];
  p1.b_is_const = 1;
  p1.b_const.val = inputs_offset[opd_idx];
  p1.b_const.is_signed = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_add(&p1);

  // This step:
  //    Sx/Sy(Qx1-Zx1)
  // we already make Sx/Sy = 2^rshift * multiplier(8bit)
  // rshift will be done at final, here we do
  // (Qx-Zx) * multiplier, beacause of (Qx-Zx) is 16bit
  // we mul high 8bit first, get Qx'_high
  cvk_tiu_mul_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = tl_working[1];
  p2.a = tl_working[1];
  p2.b_is_const = 1;
  p2.b_const.is_signed = 1;
  p2.b_const.val = multipliers[opd_idx];
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_mul(&p2);

  // refresh input
  reset(&input);

  // then we do mac with low 8bit
  // and add Qx'_high with put in res_high( high 8bit)
  // (Qx_low) * multiplier + (Qx'_high << 8) = (Qx-Zx) * multiplier
  cvk_fmt_t input_fmt = tl_working[0]->fmt;
  tl_working[0]->fmt = CVK_FMT_U8;
  p.res_high = tl_working[1];
  p.res_low = &input;
  p.a = tl_working[0];
  p.res_is_int8 = false;
  p.b_const.val = multipliers[opd_idx];
  p.b_is_const = 1;
  p.b_const.is_signed = true;
  p.lshift_bits = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mac(&p);

  tl_working[0]->fmt = input_fmt;

  // final we get ((Qx-Zx) * multiplier)(16bit)
  // add to output space
  p1.res_high = &output_high;
  p1.res_low = &output;
  p1.a_high = tl_working[1];
  p1.a_low = &input;
  p1.b.high = &output_high;
  p1.b.low = &output;
  p1.b_is_const = 0;
  p1.rshift_bits = 0;
  if (last_step) {
    // we make Sx/Sy = 2^rshift * multiplier(8bit)
    // and rshift do at final step
    p1.rshift_bits = rshift;
  }
  ctx.tiu_add(&p1);

  if (last_step) {
    // sub offset, fill 1 in tensor
    // dirty input
    {
      // reset to 0
      reset(&input);
      // clear to 1 for a*1 + c
      cvk_tiu_add_param_t p9 = {0};
      p9.res_high = tl_working[0];
      p9.res_low = &input;
      p9.a_high = tl_working[0];
      p9.a_low = &input;
      p9.b_const.val = 1;
      p9.b_is_const = 1;
      p9.rshift_bits = 0;
      p9.relu_enable = 0;
      ctx.tiu_add(&p9);
    }

    // leverage 'p.res_is_int8'
    p.res_high = &output_high;
    p.res_low = &output;
    p.a = &input;
    p.b_is_const = 1;
    p.b_const.val = output_offset;
    p.layer_id = layer_id;
    p.res_is_int8 = true;
    p.lshift_bits = 0;
    p.rshift_bits = 0;
    p.relu_enable = do_relu;
    ctx.tiu_mac(&p);
    output_flip = 1 - output_flip;
  }
}

void TgInt8EltwiseAddKernel::reset(cvk_tl_t *tl){
  // reset tl to 0
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = tl;
  p.a = tl;
  p.b_const.val = 0;
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);
}

void TgInt8EltwiseAddKernel::compute(int32_t step_idx) {
  int tile_idx = step_idx / operand_num;
  int opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  if (is_asymmetric) {
    output_high.start_address = tl_output_h[output_flip]->start_address;
    tl_working[0]->shape = tl_working[1]->shape = output_high.shape;
    tl_working[0]->stride = tl_working[1]->stride = output_high.stride;
    tl_working[0]->fmt = tl_working[1]->fmt = output_high.fmt;
  }

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d> "
                 "in %u -> out %u\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w,
                 input.start_address, output.start_address));

  if (is_asymmetric) {
    bool is_last = !(opd_idx == 0 || opd_idx != operand_num - 1);
    if (opd_idx == 0) {
      reset(&output_high);
      reset(&output);
    }
    asymmetric_compute(opd_idx, input, output, output_high, is_last);
  } else {
    symmetric_compute(opd_idx, input, output, output_high);
  }
}

void TgInt8EltwiseMaxKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

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

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

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

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
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
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

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

void TgBf16EltwiseAddKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, fmt, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, fmt, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, fmt, 1);
  output.fmt = fmt;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, fmt, 1);
  output_high.fmt = fmt;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // move first input.
    cvk_tiu_copy_param_t p3 = {0};
    p3.src = &input;
    p3.dst = &output;
    p3.layer_id = layer_id;
    ctx.tiu_copy(&p3);
  } else {
    // calculate inputs in middle.
    cvk_tiu_add_param_t p3 = {0};
    p3.res_high = nullptr;
    p3.res_low = &output;
    p3.a_high = nullptr;
    p3.a_low = &input;
    p3.b_is_const = false;
    p3.b.high = nullptr;
    p3.b.low = &output;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    p3.relu_enable = (opd_idx != operand_num - 1) ? 0 : do_relu;
    ctx.tiu_add(&p3);
    output_flip = (opd_idx != operand_num - 1) ? output_flip : 1 - output_flip;
  }
}

void TgBf16EltwiseMaxKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));
  if (operand_num == 1) {
    cvk_tiu_max_param_t p = {0};
    p.max = &output;
    p.a = &input;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(coeffs_float[0]);
    p.b_const.is_signed = 1;
    p.layer_id = layer_id;

    ctx.tiu_max(&p);

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
  } else {
    if (opd_idx == 0) {
      // move first input.
      cvk_tiu_copy_param_t p3 = {0};
      p3.src = &input;
      p3.dst = &output;
      p3.layer_id = layer_id;
      ctx.tiu_copy(&p3);
    } else { // calculate last input.
      cvk_tiu_max_param_t p = {0};
      p.max = &output;
      p.a = &input;
      p.b_is_const = 0;
      p.b = &output;
      p.layer_id = layer_id;

      ctx.tiu_max(&p);

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
}

void TgBf16EltwiseMinKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output, output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = CVK_FMT_I8;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, CVK_FMT_I8, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output.fmt = CVK_FMT_I8;

  output_high.start_address = tl_output_h[0]->start_address;
  output_high.shape = shape;
  output_high.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);
  output_high.fmt = CVK_FMT_I8;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

  if (operand_num == 1) {
    cvk_tiu_min_param_t p = {0};
    p.min = &output;
    p.a = &input;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(coeffs_float[0]);
    p.b_const.is_signed = 1;
    p.layer_id = layer_id;

    ctx.tiu_min(&p);

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
  } else {
    if (opd_idx == 0) {
      // move first input.
      cvk_tiu_copy_param_t p3 = {0};
      p3.src = &input;
      p3.dst = &output;
      p3.layer_id = layer_id;
      ctx.tiu_copy(&p3);
    } else { // calculate last input.
      cvk_tiu_min_param_t p = {0};
      p.min = &output;
      p.a = &input;
      p.b_is_const = 0;
      p.b = &output;
      p.layer_id = layer_id;

      ctx.tiu_min(&p);

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
}

void TgBf16EltwiseMulKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto opd_idx = step_idx % operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, fmt, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, fmt, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, fmt, 1);
  output.fmt = fmt;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d>\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w));

  if (opd_idx == 0) {
    // move first input.
    cvk_tiu_copy_param_t p3 = {0};
    p3.src = &input;
    p3.dst = &output;
    p3.layer_id = layer_id;
    ctx.tiu_copy(&p3);
  } else {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &output;
    p.a = &input;
    p.b = &output;
    p.b_is_const = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = do_relu;
    ctx.tiu_mul(&p);
    output_flip = 1 - output_flip;
  }
}

void TgBf16EltwiseMinMaxKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output;
  //cvk_tl_t output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, fmt, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, fmt, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, fmt, 1);
  output.fmt = fmt;

  //output_high.start_address = tl_output_h[0]->start_address;
  //output_high.shape = shape;
  //output_high.stride = ctx.tl_default_stride(shape, fmt, 1);
  //output_high.fmt = fmt;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d> "
                 "in:%u -> out:%u\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w,
                 input.start_address, output.start_address));

  cvk_tiu_min_param_t p7 = {0};
  p7.min = &output;
  p7.a = &input;
  p7.b_is_const = 1;
  p7.b_const.val = ctx.convert_fp32_to_bf16(coeffs_float[0]);
  p7.b_const.is_signed = 1;
  p7.layer_id = layer_id;

  ctx.tiu_min(&p7);

  // ELTWISE_MIN
  cvk_tiu_max_param_t p = {0};
  p.max = &output;
  p.a = &output;
  p.b_is_const = 1;
  p.b_const.val = ctx.convert_fp32_to_bf16(coeffs_float[1]);
  p.b_const.is_signed = 1;
  p.layer_id = layer_id;

  ctx.tiu_max(&p);

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

void TgEltwiseAbsKernel::compute(int32_t step_idx) {
  auto tile_idx = step_idx / operand_num;
  auto tile = tiles[tile_idx];

  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w / stride_w);

  cvk_tl_t input, output;
  //cvk_tl_t output_high;
  input.start_address = tl_input[1 - input_flip]->start_address;
  input.shape = shape;
  input.fmt = fmt;
  if (do_early_stride) {
    cvk_tl_shape_t tdma_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    input.stride = ctx.tl_default_stride(tdma_shape, fmt, 1);
    input.stride.w = stride_w * elementSize;
  } else {
    input.stride = ctx.tl_default_stride(shape, fmt, 1);
  }

  output.start_address = tl_output[output_flip]->start_address;
  output.shape = shape;
  output.stride = ctx.tl_default_stride(shape, fmt, 1);
  output.fmt = fmt;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], input<%d,%d,%d,%d:"
                 "%d,%d,%d,%d>, output<%d,%d,%d,%d:%d,%d,%d,%d> "
                 "in:%u -> out:%u\n",
                 step_idx, 1 - input_flip, output_flip, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, input.stride.n, input.stride.c,
                 input.stride.h, input.stride.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, output.stride.n, output.stride.c,
                 output.stride.h, output.stride.w,
                 input.start_address, output.start_address));

  int16_t mul_const = -1;
  if (fmt == CVK_FMT_BF16) {
    mul_const = ctx.convert_fp32_to_bf16(-1.0);
  }
  // abs = max(-1 * x, 1)
  cvk_tiu_mul_param_t p = {0};
  p.res_high = NULL;
  p.res_low = &output;
  p.a = &input;
  p.b_const.val = mul_const;
  p.b_const.is_signed = true;
  p.b_is_const = true;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  cvk_tiu_max_param_t p2 = {0};
  p2.max = &output;
  p2.a = &output;
  p2.b_is_const = 0;
  p2.b = &input;
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

void cvi_backend_tg_fixed_eltwise_add_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs,
    int32_t *inputs_offset, int32_t output_offset) {

  TgInt8EltwiseAddKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, rshift, multipliers, coeffs,
              inputs_offset, output_offset);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_max_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMaxKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_min_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMinKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_eltwise_mul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs) {
  assert(operand_num == 2);
  TgInt8EltwiseMulKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, rshift, multipliers, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_add_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_inputs[],
                                            gaddr_t ga_output, int32_t operand_num,
                                            int32_t n, int32_t c, int32_t h, int32_t w,
                                            bool do_relu, bool do_early_stride,
                                            int32_t stride_h, int32_t stride_w,
                                            const float coeffs[]) {
  TgBf16EltwiseAddKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_mul_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_inputs[],
                                            gaddr_t ga_output, int32_t operand_num,
                                            int32_t n, int32_t c, int32_t h, int32_t w,
                                            bool do_relu, bool do_early_stride,
                                            int32_t stride_h, int32_t stride_w,
                                            const float coeffs[]) {
  TgBf16EltwiseMulKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_max_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_inputs[],
                                            gaddr_t ga_output, int32_t operand_num,
                                            int32_t n, int32_t c, int32_t h, int32_t w,
                                            bool do_relu, bool do_early_stride,
                                            int32_t stride_h, int32_t stride_w,
                                            const float coeffs[]) {
  assert((operand_num == 1) || (operand_num == 2));
  TgBf16EltwiseMaxKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_min_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_inputs[],
                                            gaddr_t ga_output, int32_t operand_num,
                                            int32_t n, int32_t c, int32_t h, int32_t w,
                                            bool do_relu, bool do_early_stride,
                                            int32_t stride_h, int32_t stride_w,
                                            const float coeffs[]) {
  assert((operand_num == 1) || (operand_num == 2));
  TgBf16EltwiseMinKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_eltwise_min_max_kernel(const CviBackendContext &ctx,
                                                uint32_t layer_id, gaddr_t ga_inputs[],
                                                gaddr_t ga_output, int32_t operand_num,
                                                int32_t n, int32_t c, int32_t h,
                                                int32_t w, bool do_relu,
                                                bool do_early_stride, int32_t stride_h,
                                                int32_t stride_w, const float coeffs[]) {
  TgBf16EltwiseMinMaxKernel kernel(ctx);
  kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
              do_early_stride, stride_h, stride_w, coeffs);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_eltwise_abs_kernel(const CviBackendContext &ctx,
                                       uint32_t layer_id, gaddr_t ga_inputs[],
                                       gaddr_t ga_output, int32_t operand_num,
                                       int32_t n, int32_t c, int32_t h,
                                       int32_t w, bool do_relu,
                                       bool do_early_stride, int32_t stride_h,
                                       int32_t stride_w, int32_t rshift,
                                       const int32_t *multipliers,
                                       const int32_t *coeffs, cvk_fmt_t fmt) {
  TgEltwiseAbsKernel kernel(ctx);

  if (fmt == CVK_FMT_BF16) {
    kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
        do_early_stride, stride_h, stride_w, NULL);
  }
  else {
    kernel.init(layer_id, ga_inputs, ga_output, operand_num, n, c, h, w, do_relu,
        do_early_stride, stride_h, stride_w, rshift, multipliers, coeffs);
  }

  kernel.selectTilePolicy();
  kernel.schedule();
}
