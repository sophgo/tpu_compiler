/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_gru.cpp
 * Description:
 */

#include "TgBf16GruKernel.hpp"
#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_gru_kernel"

#define ASSERT(x) assert(x)

void TgGruKernel::matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix) {
  cvk_tl_shape_t shape = {matrix.shape.n, matrix.shape.c, 1, matrix.shape.w};
  ctx.lmem_init_tensor(tensor, shape, fmt, 1);
  tensor->start_address = matrix.start_address;
}

void TgGruKernel::matrix_for_tiu(cvk_ml_t *matrix) {
  if (matrix->shape.w < ctx.tiu_eu_num(fmt) && matrix->shape.c > 1) {
    matrix->shape.w = ctx.tiu_eu_num(fmt);
    matrix->stride = ctx.ml_default_stride(matrix->shape, fmt, 1);
  }
}

void TgGruKernel::zeros(const cvk_ml_t &matrix) {
  cvk_tl_t tl_mem;
  matrix_to_tensor(&tl_mem, matrix);
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_mem;
  p.a = &tl_mem; // rt
  p.b_is_const = 1;
  p.b_const.val = ctx.convert_fp32_to_bf16(0.0f);
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);
}

void TgGruKernel::matrix_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                             const cvk_ml_t &ml_right, const cvk_ml_t &ml_bias,
                             uint8_t ps32_mode) {
  cvk_ml_t ml_res_ = ml_res;
  cvk_ml_t ml_left_ = ml_left;
  cvk_ml_t ml_right_ = ml_right;
  cvk_ml_t ml_bias_ = ml_bias;
  matrix_for_tiu(&ml_res_);
  matrix_for_tiu(&ml_left_);
  matrix_for_tiu(&ml_right_);
  cvk_ml_t *p_bias = nullptr;
  if (ps32_mode == 0 || ps32_mode == 1) {
    matrix_for_tiu(&ml_bias_);
    p_bias = &ml_bias_;
  }
  cvk_tiu_matrix_multiplication_param_t p = {0};
  p.res = &ml_res_;
  p.left = &ml_left_;
  p.right = &ml_right_;
  p.bias = p_bias;
  p.ps32_mode = ps32_mode;
  p.layer_id = layer_id;
  ctx.tiu_matrix_multiplication(&p);
}

void TgGruKernel::eltwise_mul(const cvk_ml_t &ml_res,
                              const cvk_ml_t &ml_right) {
  eltwise_mul(ml_res, ml_res, ml_right);
}

void TgGruKernel::eltwise_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                              const cvk_ml_t &ml_right) {
  cvk_tl_t tl_res, tl_left, tl_right;
  matrix_to_tensor(&tl_res, ml_res);
  matrix_to_tensor(&tl_left, ml_left);
  matrix_to_tensor(&tl_right, ml_right);
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_res;
  p.a = &tl_left;
  p.b_is_const = 0;
  p.b = &tl_right;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);
}

void TgGruKernel::eltwise_add(const cvk_ml_t &ml_res,
                              const cvk_ml_t &ml_right) {
  cvk_tl_t tl_res, tl_right;
  matrix_to_tensor(&tl_res, ml_res);
  matrix_to_tensor(&tl_right, ml_right);
  cvk_tiu_add_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_res;
  p.a_high = nullptr;
  p.a_low = &tl_res;
  p.b_is_const = false;
  p.b.high = nullptr;
  p.b.low = &tl_right;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_add(&p);
}

void TgGruKernel::eltwise_sub(const cvk_ml_t &ml_res,
                              const cvk_ml_t &ml_right) {
  eltwise_sub(ml_res, ml_res, ml_right);
}

void TgGruKernel::eltwise_sub(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                              const cvk_ml_t &ml_right) {
  cvk_tl_t tl_res, tl_left, tl_right;
  matrix_to_tensor(&tl_res, ml_res);
  matrix_to_tensor(&tl_left, ml_left);
  matrix_to_tensor(&tl_right, ml_right);
  cvk_tiu_sub_param_t p = {0};
  p.res_high = 0;
  p.res_low = &tl_res;
  p.a_high = 0;
  p.a_low = &tl_left;
  p.b_high = 0;
  p.b_low = &tl_right;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  ctx.tiu_sub(&p);
}

void TgGruKernel::sigmoid(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
                          const cvk_ml_t &ml_buff) {
  cvi_backend_bf16_tl_lut_slope_method(
      ctx, layer_id, ml_in.start_address, ml_out.start_address,
      ml_buff.start_address, addr_sigmoid, addr_sigmoid_slope, -8, 8, false,
      ml_in.shape.n, ml_in.shape.c, 1, ml_in.shape.w);
}
void TgGruKernel::tanh(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
                       const cvk_ml_t &ml_buff) {
  cvi_backend_bf16_tl_lut_slope_method(
      ctx, layer_id, ml_in.start_address, ml_out.start_address,
      ml_buff.start_address, addr_tanh, addr_tanh_slope, -8, 8, false,
      ml_in.shape.n, ml_in.shape.c, 1, ml_in.shape.w);
}

void TgGruKernel::assign_matrix(cvk_ml_t *ml_mem, const cvk_ml_shape_t &shape) {
  ctx.lmem_init_matrix(ml_mem, shape, fmt, 1);
  ml_mem->start_address = lmem_used;
  lmem_used += ctx.lmem_matrix_to_size(shape, fmt, 1);
  assert(lmem_used <= (uint32_t)LOCAL_MEM_SIZE);
}

void TgGruKernel::fill_matrix(cvk_ml_t *ml_mem, uint32_t row, uint32_t col,
                              uint32_t addr) {
  auto shape = ctx.ml_default_shape(row, col, fmt);
  ctx.lmem_init_matrix(ml_mem, shape, fmt, 1);
  ml_mem->start_address = addr;
}

void TgGruKernel::assign_addr(cvk_tl_t *tl_mem, uint32_t size) {
  tl_mem->start_address = lmem_used;
  lmem_used += size;
  assert(lmem_used <= (uint32_t)LOCAL_MEM_SIZE);
}

void TgGruKernel::init_table() {
  cvk_tl_t tl_table;
  table_shape = ctx.lut_table_shape(fmt);
  table_size = ctx.lmem_tensor_to_size(table_shape, fmt, 1);
  ctx.lmem_init_tensor(&tl_table, table_shape, fmt, 1);
  assign_addr(&tl_table, table_size);
  addr_sigmoid = tl_table.start_address;
  ctx.tdma_load(&tl_table, ga_sigmoid_lut);

  assign_addr(&tl_table, table_size);
  addr_sigmoid_slope = tl_table.start_address;
  ctx.tdma_load(&tl_table, ga_sigmoid_slope_lut);

  assign_addr(&tl_table, table_size);
  addr_tanh = tl_table.start_address;
  ctx.tdma_load(&tl_table, ga_tanh_lut);

  assign_addr(&tl_table, table_size);
  addr_tanh_slope = tl_table.start_address;
  ctx.tdma_load(&tl_table, ga_tanh_slope_lut);
}

bool TgGruKernel::need_tiling() {
  auto x_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto x_size = ctx.lmem_matrix_to_size(x_shape, fmt, 1);
  auto r_size = ctx.lmem_matrix_to_size(r_shape, fmt, 1);
  auto b_size = ctx.lmem_matrix_to_size(b_shape, fmt, 1);
  uint64_t total_size = lmem_used + 3 * r_size + 3 * b_size + 6 * x_size;
  if (total_size > (uint32_t)LOCAL_MEM_SIZE) {
    return true;
  }
  return false;
}

void TgGruKernel::tiling() {
  auto lmem_addr = lmem_used;
  for (step_size = hidden_size; step_size > 0; step_size--) {
    lmem_used = lmem_addr;
    auto x_shape = ctx.ml_default_shape(batch_size, step_size, fmt);
    auto r_shape = ctx.ml_default_shape(step_size, step_size, fmt);
    auto b_shape = ctx.ml_default_shape(4 / fmt_size, step_size, fmt);
    auto x_size = ctx.lmem_matrix_to_size(x_shape, fmt, 1);
    auto x_ps32_size = ctx.lmem_ps32_matrix_to_size(x_shape, fmt, 1);
    auto r_size = ctx.lmem_matrix_to_size(r_shape, fmt, 1);
    auto b_size = ctx.lmem_matrix_to_size(b_shape, fmt, 1);
    addr_recurrence = lmem_used;
    lmem_used += r_size;
    addr_bias = lmem_used;
    lmem_used += b_size;
    addr_work0 = lmem_used;
    lmem_used += std::max(2 * x_size, x_ps32_size);
    addr_work1 = lmem_used;
    lmem_used += x_size;
    addr_xz = lmem_used;
    lmem_used += x_size;
    addr_xh = lmem_used;
    lmem_used += x_size;
    // hidden state size
    uint32_t state_size = 0;
    for (int pos = 0; pos < hidden_size; pos += step_size) {
      int h = std::min(step_size, hidden_size - pos);
      auto h_shape = ctx.ml_default_shape(batch_size, h, fmt);
      state_size += ctx.lmem_matrix_to_size(h_shape, fmt, 1);
    }
    if (step_size != hidden_size) { // need backup hiddens
      state_size *= 2;
    }
    if (lmem_used + state_size <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
  }
  if (step_size == 0) {
    llvm::errs() << llvm::format(
        "Tilling GRU failed,seq:%d,batch:%d,hidden:%d\n", seq_length,
        batch_size, hidden_size);
    assert(0);
  }
  step_num = ceiling_func(hidden_size, step_size);
  tiling_t tile = {0, 0};
  for (int i = 0; i < step_num; i++) {
    tile.pos_h = i * step_size;
    tile.h = std::min(step_size, hidden_size - tile.pos_h);
    tiles.emplace_back(tile);
    cvk_ml_t ml_hidden;
    auto h_shape = ctx.ml_default_shape(batch_size, tile.h, fmt);
    assign_matrix(&ml_hidden, h_shape);
    ml_hiddens[0].emplace_back(ml_hidden);
    if (step_num > 1) {
      assign_matrix(&ml_hidden, h_shape);
      ml_hiddens[1].emplace_back(ml_hidden);
    }
  }
}

uint8_t TgGruKernel::ps32_mode(int step_idx) {
  assert(step_idx < step_num);
  if (step_num == 1) {
    return 0;
  }
  if (step_idx == 0) {
    return 2;
  }
  if (step_idx == step_num - 1) {
    return 1;
  }
  return 3;
}

void TgGruKernel::init_h0() {
  for (int step = 0; step < step_num; step++) {
    auto &ml_hidden = ml_hiddens[0][step];
    auto &tile = tiles[step];
    if (with_initial_h) {
      ctx.tdma_load_stride(&ml_hidden, ga_h0 + tile.pos_h * fmt_size,
                           h_gstride);
    } else {
      zeros(ml_hidden);
    }
  }
}

void TgGruKernel::matrix_recurrence(const cvk_ml_t &ml_res, int flip,
                                    const tiling_t &tile, gaddr_t ga_weight,
                                    gaddr_t ga_bias) {
  cvk_ml_t ml_weight, ml_bias;
  fill_matrix(&ml_bias, 4 / fmt_size, tile.h, addr_bias);
  ctx.tdma_load_stride(&ml_bias, ga_bias + tile.pos_h * fmt_size, h_gstride);
  for (int i = 0; i < step_num; i++) {
    int offset = tiles[i].pos_h * hidden_bytes + tile.pos_h * fmt_size;
    fill_matrix(&ml_weight, tiles[i].h, tile.h, addr_recurrence);
    ctx.tdma_load_stride(&ml_weight, ga_weight + offset, h_gstride);
    matrix_mul(ml_res, ml_hiddens[flip][i], ml_weight, ml_bias, ps32_mode(i));
  }
}

void TgGruKernel::compute(int idx, bool forward) {
  cvk_ml_t ml_work0, ml_work1, ml_xz, ml_xr, ml_xh;
  int seq_idx = forward ? idx : (seq_length - 1 - idx);
  int x_offset = seq_idx * batch_size * input_bytes; // load input
  int s_offset = seq_idx * num_dir * x_bytes;        // store output
  int flip = 0, next = 0;
  if (step_num > 1) {
    flip = idx % 2;
    next = 1 - flip;
  }
  for (int step = 0; step < step_num; step++) {
    auto &tile = tiles[step];
    auto &ml_hidden = ml_hiddens[flip][step];
    auto &ml_result = ml_hiddens[next][step];
    int goffset = tile.pos_h * fmt_size;
    fill_matrix(&ml_xz, batch_size, tile.h, addr_xz);
    fill_matrix(&ml_xr, batch_size, tile.h, addr_xh); // use addr_xh
    fill_matrix(&ml_xh, batch_size, tile.h, addr_xh);
    fill_matrix(&ml_work0, batch_size, tile.h, addr_work0);
    fill_matrix(&ml_work1, batch_size, tile.h, addr_work1);
    // gate z
    matrix_recurrence(ml_work0, flip, tile, ga_rz, ga_rbz);
    ctx.tdma_load_stride(&ml_work1, ga_xz + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_xz, ml_work1, ml_work0);
    // gate r
    matrix_recurrence(ml_work0, flip, tile, ga_rr, ga_rbr);
    ctx.tdma_load_stride(&ml_work1, ga_xr + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_xr, ml_work1, ml_work0);
    // gate h
    matrix_recurrence(ml_work0, flip, tile, ga_rh, ga_rbh);
    eltwise_mul(ml_work0, ml_xr);
    ctx.tdma_load_stride(&ml_work1, ga_xh + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    tanh(ml_xh, ml_work1, ml_work0);
    // H = (1-z)*h + z * H_t
    eltwise_mul(ml_work1, ml_hidden, ml_xz);
    eltwise_add(ml_work1, ml_xh);
    eltwise_mul(ml_xz, ml_xh);
    eltwise_sub(ml_result, ml_work1, ml_xz);
    ctx.tdma_store_stride(&ml_result, ga_store + s_offset + goffset, h_gstride);
  }
}

void TgGruKernel::compute_without_tiling(bool forward) {
  auto lmem_used_backup = lmem_used;
  init_gaddr(forward);
  auto x_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto x2_shape = ctx.ml_default_shape(batch_size * 2, hidden_size, fmt);
  // assign lmem
  cvk_ml_t ml_rz, ml_rr, ml_rh, ml_rbz, ml_rbr, ml_rbh;
  cvk_ml_t ml_xz, ml_xh, ml_hidden, ml_work0, ml_work1;
  assign_matrix(&ml_rz, r_shape);
  assign_matrix(&ml_rr, r_shape);
  assign_matrix(&ml_rh, r_shape);
  assign_matrix(&ml_rbz, b_shape);
  assign_matrix(&ml_rbr, b_shape);
  assign_matrix(&ml_rbh, b_shape);
  assign_matrix(&ml_hidden, x_shape);
  assign_matrix(&ml_xz, x_shape);
  assign_matrix(&ml_xh, x_shape);
  assign_matrix(&ml_work0, x2_shape);
  assign_matrix(&ml_work1, x_shape);

  // load recurrence and bias
  ctx.tdma_load(&ml_rz, ga_rz);
  ctx.tdma_load(&ml_rr, ga_rr);
  ctx.tdma_load(&ml_rh, ga_rh);
  ctx.tdma_load(&ml_rbz, ga_rbz);
  ctx.tdma_load(&ml_rbr, ga_rbr);
  ctx.tdma_load(&ml_rbh, ga_rbh);

  // load initial_h if exist or clear to zeros
  if (with_initial_h) {
    ctx.tdma_load(&ml_hidden, ga_h0);
  } else {
    zeros(ml_hidden);
  }

  for (int i = 0; i < seq_length; i++) {
    int seq_idx = forward ? i : (seq_length - i - 1);
    int x_offset = seq_idx * batch_size * input_bytes;
    int s_offset = seq_idx * num_dir * x_bytes;
    // gate_z => ml_xz
    ctx.tdma_load_stride(&ml_xz, ga_xz + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_rz, ml_rbz);
    eltwise_add(ml_work1, ml_xz);
    sigmoid(ml_xz, ml_work1, ml_work0);
    // gate_r => ml_xh (tmp)
    ctx.tdma_load_stride(&ml_xh, ga_xr + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_rr, ml_rbr);
    eltwise_add(ml_work1, ml_xh);
    sigmoid(ml_xh, ml_work1, ml_work0);
    // gate_h => ml_xh
    matrix_mul(ml_work1, ml_hidden, ml_rh, ml_rbh);
    eltwise_mul(ml_work1, ml_xh);
    ctx.tdma_load_stride(&ml_xh, ga_xh + x_offset, x_gstride);
    eltwise_add(ml_work1, ml_xh);
    tanh(ml_xh, ml_work1, ml_work0);
    // H = (1-gate_z)*gate_h + gate_z * hidden
    eltwise_mul(ml_hidden, ml_xz);
    eltwise_add(ml_hidden, ml_xh);
    eltwise_mul(ml_xz, ml_xh);
    eltwise_sub(ml_hidden, ml_xz);
    ctx.tdma_store_stride(&ml_hidden, ga_store + s_offset, h_gstride);
  }
  lmem_used = lmem_used_backup;
}

void TgGruKernel::init_gaddr(bool forward) {
  if (forward) {
    ga_xz = ga_input;
    ga_rz = ga_recurrence;
    ga_rbz = ga_bias;
    ga_store = ga_output;
    ga_h0 = ga_init_h;
  } else {
    ga_xz = ga_input + hidden_bytes * 3;
    ga_rz = ga_recurrence + recurrence_bytes * 3;
    ga_rbz = ga_bias + hidden_bytes * 3;
    ga_store = ga_output + x_bytes;
    ga_h0 = ga_init_h + x_bytes;
  }
  ga_xr = ga_xz + hidden_bytes;
  ga_xh = ga_xr + hidden_bytes;
  ga_rr = ga_rz + recurrence_bytes;
  ga_rh = ga_rr + recurrence_bytes;
  ga_rbr = ga_rbz + hidden_bytes;
  ga_rbh = ga_rbr + hidden_bytes;
}

void TgGruKernel::init(uint32_t layer_id, gaddr_t ga_input,
                       gaddr_t ga_recurrence, gaddr_t ga_bias,
                       gaddr_t ga_init_h, gaddr_t ga_sigmoid_lut,
                       gaddr_t ga_sigmoid_slope_lut, gaddr_t ga_tanh_lut,
                       gaddr_t ga_tanh_slope_lut, gaddr_t ga_output,
                       int seq_length, int num_dir, int batch_size,
                       int hidden_size, bool do_bias, bool with_initial_h,
                       bool linear_before_reset, bool bidirectional) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_recurrence = ga_recurrence;
  this->ga_bias = ga_bias;
  this->ga_init_h = ga_init_h;
  this->ga_sigmoid_lut = ga_sigmoid_lut;
  this->ga_sigmoid_slope_lut = ga_sigmoid_slope_lut;
  this->ga_tanh_lut = ga_tanh_lut;
  this->ga_tanh_slope_lut = ga_tanh_slope_lut;
  this->ga_output = ga_output;
  this->seq_length = seq_length;
  this->batch_size = batch_size;
  this->num_dir = num_dir;
  this->hidden_size = hidden_size;
  this->do_bias = do_bias;
  this->with_initial_h = with_initial_h;
  this->linear_before_reset = linear_before_reset;
  this->bidirectional = bidirectional;
  this->fmt = CVK_FMT_BF16;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  this->lmem_used = 0;
  this->hidden_bytes = hidden_size * fmt_size;
  this->input_bytes = num_dir * 3 * hidden_bytes;
  this->recurrence_bytes = hidden_size * hidden_bytes;
  this->x_bytes = batch_size * hidden_bytes;
  this->x_gstride.row = input_bytes;
  this->h_gstride.row = hidden_bytes;
  assert(linear_before_reset == true); // support later
  assert(do_bias == true);             // support later
  init_table();
}

void TgGruKernel::compute_with_tiling(bool forward) {
  init_gaddr(forward);
  init_h0();
  for (int i = 0; i < seq_length; i++) {
    compute(i, forward);
  }
}

void TgGruKernel::schedule() {
  if (false == need_tiling()) {
    compute_without_tiling(true);
    if (bidirectional) {
      compute_without_tiling(false);
    }
  } else {
    tiling();
    compute_with_tiling(true);
    if (bidirectional) {
      compute_with_tiling(false);
    }
  }
}

void cvi_backend_tg_bf16_gru_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_recurrence, gaddr_t ga_bias, gaddr_t ga_initial_h,
    gaddr_t ga_sigmoid_lut, gaddr_t ga_sigmoid_slope_lut, gaddr_t ga_tanh_lut,
    gaddr_t ga_tanh_slope_lut, gaddr_t ga_output, int seq_len, int num_dir,
    int batch_size, int hidden_size, bool do_bias, bool with_initial_h,
    bool is_linear_before_reset, bool is_bidirectional) {
  TgGruKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_recurrence, ga_bias, ga_initial_h,
              ga_sigmoid_lut, ga_sigmoid_slope_lut, ga_tanh_lut,
              ga_tanh_slope_lut, ga_output, seq_len, num_dir, batch_size,
              hidden_size, do_bias, with_initial_h, is_linear_before_reset,
              is_bidirectional);
  kernel.schedule();
}
