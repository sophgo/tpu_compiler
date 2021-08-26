/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_lstm.cpp
 * Description:
 */

#include "TgBf16LstmKernel.hpp"
#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_lstm_kernel"

#define ASSERT(x) assert(x)

void TgLstmKernel::matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix) {
  cvk_tl_shape_t shape = {matrix.shape.n, matrix.shape.c, 1, matrix.shape.w};
  ctx.lmem_init_tensor(tensor, shape, fmt, 1);
  tensor->start_address = matrix.start_address;
}

void TgLstmKernel::matrix_for_tiu(cvk_ml_t *matrix) {
  if (matrix->shape.w < ctx.tiu_eu_num(fmt) && matrix->shape.c > 1) {
    matrix->shape.w = ctx.tiu_eu_num(fmt);
    matrix->stride = ctx.ml_default_stride(matrix->shape, fmt, 1);
  }
}

void TgLstmKernel::zeros(const cvk_ml_t &matrix) {
  cvk_tl_t tl_mem;
  matrix_to_tensor(&tl_mem, matrix);
  ctx.tiu_zeros(layer_id, &tl_mem);
}

void TgLstmKernel::matrix_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
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
  if (do_bias && (ps32_mode == 0 || ps32_mode == 1)) {
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

void TgLstmKernel::eltwise_mul(const cvk_ml_t &ml_res,
                               const cvk_ml_t &ml_right) {
  eltwise_mul(ml_res, ml_res, ml_right);
}

void TgLstmKernel::eltwise_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
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

void TgLstmKernel::eltwise_add(const cvk_ml_t &ml_res,
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

void TgLstmKernel::eltwise_sub(const cvk_ml_t &ml_res,
                               const cvk_ml_t &ml_right) {
  eltwise_sub(ml_res, ml_res, ml_right);
}

void TgLstmKernel::eltwise_sub(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
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

void TgLstmKernel::sigmoid(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
                           const cvk_ml_t &ml_buff) {
  cvi_backend_bf16_tl_lut_slope_method(
      ctx, layer_id, ml_in.start_address, ml_out.start_address,
      ml_buff.start_address, addr_sigmoid, addr_sigmoid_slope,
      -1 * SIGMOID_BF16_LUT_RANGE, SIGMOID_BF16_LUT_RANGE, ml_in.shape.n,
      ml_in.shape.c, 1, ml_in.shape.w);
}
void TgLstmKernel::tanh(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
                        const cvk_ml_t &ml_buff) {
  cvi_backend_bf16_tl_lut_slope_method(
      ctx, layer_id, ml_in.start_address, ml_out.start_address,
      ml_buff.start_address, addr_tanh, addr_tanh_slope,
      -1 * TANH_BF16_LUT_RANGE, TANH_BF16_LUT_RANGE, ml_in.shape.n,
      ml_in.shape.c, 1, ml_in.shape.w);
}

void TgLstmKernel::assign_matrix(cvk_ml_t *ml_mem,
                                 const cvk_ml_shape_t &shape) {
  ctx.lmem_init_matrix(ml_mem, shape, fmt, 1);
  ml_mem->start_address = lmem_used;
  lmem_used += ctx.lmem_matrix_to_size(shape, fmt, 1);
  assert(lmem_used <= (uint32_t)LOCAL_MEM_SIZE);
}

void TgLstmKernel::fill_matrix(cvk_ml_t *ml_mem, uint32_t row, uint32_t col,
                               uint32_t addr) {
  auto shape = ctx.ml_default_shape(row, col, fmt);
  ctx.lmem_init_matrix(ml_mem, shape, fmt, 1);
  ml_mem->start_address = addr;
}

void TgLstmKernel::assign_addr(cvk_tl_t *tl_mem, uint32_t size) {
  tl_mem->start_address = lmem_used;
  lmem_used += size;
  assert(lmem_used <= (uint32_t)LOCAL_MEM_SIZE);
}

void TgLstmKernel::init_table() {
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

// new_input = old_input dot weight => [seq_length, batch_size, num_dir * 4 *
// hidden_size]
bool TgLstmKernel::need_tiling() {
  auto x_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto x_size = ctx.lmem_matrix_to_size(x_shape, fmt, 1);
  auto r_size = ctx.lmem_matrix_to_size(r_shape, fmt, 1);
  uint32_t b_size = 0;
  if (do_bias) {
    b_size = ctx.lmem_matrix_to_size(b_shape, fmt, 1);
  }
  uint64_t total_size = lmem_used + 4 * r_size + 4 * b_size + 8 * x_size;
  if (total_size > (uint32_t)LOCAL_MEM_SIZE) {
    return true;
  }
  return false;
}

uint8_t TgLstmKernel::ps32_mode(int step_idx) {
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

void TgLstmKernel::init_h0c0() {
  for (int step = 0; step < step_num; step++) {
    auto &ml_hidden = ml_hiddens[0][step];
    auto &ml_cell = ml_cells[step];
    auto &tile = tiles[step];
    if (with_initial_h) {
      ctx.tdma_load_stride(&ml_hidden, ga_h0 + tile.pos_h * fmt_size,
                           h_gstride);
    } else {
      zeros(ml_hidden);
    }
    if (with_initial_c) {
      ctx.tdma_load_stride(&ml_cell, ga_c0 + tile.pos_h * fmt_size, h_gstride);
    } else {
      zeros(ml_cell);
    }
  }
}

void TgLstmKernel::matrix_recurrence(const cvk_ml_t &ml_res, int flip,
                                     const tiling_t &tile, gaddr_t ga_weight,
                                     gaddr_t ga_bias) {
  cvk_ml_t ml_weight, ml_bias;
  if (do_bias) {
    fill_matrix(&ml_bias, 4 / fmt_size, tile.h, addr_bias);
    ctx.tdma_load_stride(&ml_bias, ga_bias + tile.pos_h * fmt_size, h_gstride);
  }
  for (int i = 0; i < step_num; i++) {
    int offset = tiles[i].pos_h * hidden_bytes + tile.pos_h * fmt_size;
    fill_matrix(&ml_weight, tiles[i].h, tile.h, addr_recurrence);
    ctx.tdma_load_stride(&ml_weight, ga_weight + offset, h_gstride);
    matrix_mul(ml_res, ml_hiddens[flip][i], ml_weight, ml_bias, ps32_mode(i));
  }
}

void TgLstmKernel::tiling() {
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
    if (do_bias) {
      addr_bias = lmem_used;
      lmem_used += b_size;
    }
    if (with_cont) {
      addr_cont = lmem_used;
      lmem_used += x_size;
    }
    addr_work0 = lmem_used;
    lmem_used += std::max(2 * x_size, x_ps32_size);
    addr_work1 = lmem_used;
    lmem_used += x_size;
    addr_work2 = lmem_used;
    lmem_used += x_size;
    // hidden state size
    uint32_t hstate_size = 0;
    for (int pos = 0; pos < hidden_size; pos += step_size) {
      int h = std::min(step_size, hidden_size - pos);
      auto h_shape = ctx.ml_default_shape(batch_size, h, fmt);
      hstate_size += ctx.lmem_matrix_to_size(h_shape, fmt, 1);
    }
    uint32_t cstate_size = hstate_size;
    if (step_size != hidden_size) { // need backup hiddens
      hstate_size *= 2;
    }
    if (lmem_used + hstate_size + cstate_size <= (uint32_t)LOCAL_MEM_SIZE) {
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
    cvk_ml_t ml_cell;
    assign_matrix(&ml_cell, h_shape);
    ml_cells.emplace_back(ml_cell);
  }
}

void TgLstmKernel::compute(int idx, bool forward) {
  cvk_ml_t ml_work0, ml_work1, ml_xi, ml_xf, ml_xo, ml_cont;
  int seq_idx = forward ? idx : (seq_length - 1 - idx);
  int x_offset = seq_idx * batch_size * input_bytes; // load input
  int s_offset = seq_idx * num_dir * x_bytes;        // store output
  int cont_offset = seq_idx * batch_size * fmt_size;
  int flip = 0, next = 0;
  if (step_num > 1) {
    flip = idx % 2;
    next = 1 - flip;
  }
  for (int step = 0; step < step_num; step++) {
    auto &tile = tiles[step];
    auto &ml_cell = ml_cells[step];
    auto &ml_result = ml_hiddens[next][step];
    int goffset = tile.pos_h * fmt_size;
    fill_matrix(&ml_xi, batch_size, tile.h, addr_work2);
    fill_matrix(&ml_xf, batch_size, tile.h, addr_work2);
    fill_matrix(&ml_xo, batch_size, tile.h, addr_work2);
    if (with_cont) {
      fill_matrix(&ml_cont, batch_size, tile.h, addr_cont);
      load_cont(ml_cont, ga_cont + cont_offset);
    }
    fill_matrix(&ml_work0, batch_size, tile.h, addr_work0);
    fill_matrix(&ml_work1, batch_size, tile.h, addr_work1);
    // gate i
    matrix_recurrence(ml_work0, flip, tile, ga_ri, ga_rbi);
    if (with_cont) {
      eltwise_mul(ml_work0, ml_cont);
    }
    ctx.tdma_load_stride(&ml_work1, ga_xi + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_xi, ml_work1, ml_work0);
    // gate c
    matrix_recurrence(ml_work0, flip, tile, ga_rc, ga_rbc);
    if (with_cont) {
      eltwise_mul(ml_work0, ml_cont);
    }
    ctx.tdma_load_stride(&ml_work1, ga_xc + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    tanh(ml_result, ml_work1, ml_work0);
    // i * c => result
    eltwise_mul(ml_result, ml_xi);
    // gate f
    matrix_recurrence(ml_work0, flip, tile, ga_rf, ga_rbf);
    if (with_cont) {
      eltwise_mul(ml_work0, ml_cont);
    }
    ctx.tdma_load_stride(&ml_work1, ga_xf + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_xf, ml_work1, ml_work0);
    if (with_cont) {
      eltwise_mul(ml_xf, ml_cont);
    }
    // new Cell = cont * f * pre_cell + i * c
    eltwise_mul(ml_cell, ml_xf);
    eltwise_add(ml_cell, ml_result);
    // gate o
    matrix_recurrence(ml_work0, flip, tile, ga_ro, ga_rbo);
    if (with_cont) {
      eltwise_mul(ml_work0, ml_cont);
    }
    ctx.tdma_load_stride(&ml_work1, ga_xo + x_offset + goffset, x_gstride);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_xo, ml_work1, ml_work0);

    // new Hidden = o * tanh(new Cell)
    tanh(ml_result, ml_cell, ml_work0);
    eltwise_mul(ml_result, ml_xo);
    ctx.tdma_store_stride(&ml_result, ga_store + s_offset + goffset, h_gstride);
  }
}

void TgLstmKernel::load_cont(const cvk_ml_t &ml_cont, gaddr_t cont_addr) {
  cvk_tl_t tl_cont;
  matrix_to_tensor(&tl_cont, ml_cont);
  std::swap(tl_cont.shape.h, tl_cont.shape.w);
  tl_cont.stride = ctx.tl_default_stride(tl_cont.shape, fmt, 1);
  cvk_tg_stride_t cont_gstride = {.n = 1, .c = 0, .h = 0, .w = 1};
  ctx.tdma_load_stride(&tl_cont, cont_addr, cont_gstride);
  std::swap(tl_cont.shape.h, tl_cont.shape.w);
  tl_cont.stride = ctx.tl_default_stride(tl_cont.shape, fmt, 1);
}

void TgLstmKernel::compute_without_tiling(bool forward) {
  auto lmem_used_backup = lmem_used;
  init_gaddr(forward);
  auto x_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto x2_shape = ctx.ml_default_shape(batch_size * 2, hidden_size, fmt);
  // assign lmem
  cvk_ml_t ml_ri, ml_ro, ml_rf, ml_rc, ml_rbi, ml_rbo, ml_rbf, ml_rbc, ml_cont;
  cvk_ml_t ml_xi, ml_xo, ml_xf, ml_xc, ml_hidden, ml_cell, ml_work0, ml_work1;

  assign_matrix(&ml_ri, r_shape);
  assign_matrix(&ml_ro, r_shape);
  assign_matrix(&ml_rf, r_shape);
  assign_matrix(&ml_rc, r_shape);
  if (do_bias) {
    assign_matrix(&ml_rbi, b_shape);
    assign_matrix(&ml_rbo, b_shape);
    assign_matrix(&ml_rbf, b_shape);
    assign_matrix(&ml_rbc, b_shape);
  }
  assign_matrix(&ml_hidden, x_shape);
  assign_matrix(&ml_cell, x_shape);
  assign_matrix(&ml_xi, x_shape);
  assign_matrix(&ml_xc, x_shape);
  assign_matrix(&ml_work0, x2_shape);
  assign_matrix(&ml_work1, x_shape);
  if (with_cont) {
    assign_matrix(&ml_cont, x_shape);
  }
  ml_xf = ml_xi;
  ml_xo = ml_xi;

  // load recurrence and bias
  ctx.tdma_load(&ml_ri, ga_ri);
  ctx.tdma_load(&ml_rf, ga_rf);
  ctx.tdma_load(&ml_rc, ga_rc);
  ctx.tdma_load(&ml_ro, ga_ro);
  if (do_bias) {
    ctx.tdma_load(&ml_rbi, ga_rbi);
    ctx.tdma_load(&ml_rbf, ga_rbf);
    ctx.tdma_load(&ml_rbc, ga_rbc);
    ctx.tdma_load(&ml_rbo, ga_rbo);
  }

  // load initial_h if exist or clear to zeros
  if (with_initial_h) {
    ctx.tdma_load(&ml_hidden, ga_h0);
  } else {
    zeros(ml_hidden);
  }
  if (with_initial_c) {
    ctx.tdma_load(&ml_cell, ga_c0);
  } else {
    zeros(ml_cell);
  }

  for (int i = 0; i < seq_length; i++) {
    int seq_idx = forward ? i : (seq_length - i - 1);
    int x_offset = seq_idx * batch_size * input_bytes;
    if (with_cont) {
      gaddr_t cont_addr = ga_cont + seq_idx * batch_size * fmt_size;
      load_cont(ml_cont, cont_addr);
    }
    // f => ml_xf
    ctx.tdma_load_stride(&ml_xf, ga_xf + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_rf, ml_rbf);
    if (with_cont) {
      eltwise_mul(ml_work1, ml_cont);
    }
    eltwise_add(ml_work1, ml_xf);
    sigmoid(ml_xf, ml_work1, ml_work0);
    if (with_cont) {
      eltwise_mul(ml_xf, ml_cont);
    }
    eltwise_mul(ml_cell, ml_xf);

    // i => ml_xi
    ctx.tdma_load_stride(&ml_xi, ga_xi + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_ri, ml_rbi);
    if (with_cont) {
      eltwise_mul(ml_work1, ml_cont);
    }
    eltwise_add(ml_work1, ml_xi);
    sigmoid(ml_xi, ml_work1, ml_work0);
    // c => ml_xc
    ctx.tdma_load_stride(&ml_xc, ga_xc + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_rc, ml_rbc);
    if (with_cont) {
      eltwise_mul(ml_work1, ml_cont);
    }
    eltwise_add(ml_work1, ml_xc);
    tanh(ml_xc, ml_work1, ml_work0);
    // i * c => c
    eltwise_mul(ml_xc, ml_xi);
    // C = f * C_t + i * c
    eltwise_add(ml_cell, ml_xc);
    // o => ml_xo
    ctx.tdma_load_stride(&ml_xo, ga_xo + x_offset, x_gstride);
    matrix_mul(ml_work1, ml_hidden, ml_ro, ml_rbo);
    if (with_cont) {
      eltwise_mul(ml_work1, ml_cont);
    }
    eltwise_add(ml_work1, ml_xo);
    sigmoid(ml_xo, ml_work1, ml_work0);
    // hidden = o * tanh(cell)
    tanh(ml_hidden, ml_cell, ml_work0);
    eltwise_mul(ml_hidden, ml_xo);

    int s_offset = seq_idx * num_dir * x_bytes;
    ctx.tdma_store_stride(&ml_hidden, ga_store + s_offset, h_gstride);
  }
  lmem_used = lmem_used_backup;
}

void TgLstmKernel::init_gaddr(bool forward) {
  if (forward) {
    ga_xi = ga_input;
    ga_ri = ga_recurrence;
    ga_rbi = ga_bias;
    ga_store = ga_output;
    ga_h0 = ga_init_h;
    ga_c0 = ga_init_c;
  } else {
    ga_xi = ga_input + hidden_bytes * 4;
    ga_ri = ga_recurrence + recurrence_bytes * 4;
    ga_rbi = ga_bias + hidden_bytes * 4;
    ga_store = ga_output + x_bytes;
    ga_h0 = ga_init_h + x_bytes;
    ga_c0 = ga_init_c + x_bytes;
  }
  ga_xo = ga_xi + hidden_bytes;
  ga_xf = ga_xo + hidden_bytes;
  ga_xc = ga_xf + hidden_bytes;
  ga_ro = ga_ri + recurrence_bytes;
  ga_rf = ga_ro + recurrence_bytes;
  ga_rc = ga_rf + recurrence_bytes;
  ga_rbo = ga_rbi + hidden_bytes;
  ga_rbf = ga_rbo + hidden_bytes;
  ga_rbc = ga_rbf + hidden_bytes;
}

void TgLstmKernel::init(uint32_t layer_id, gaddr_t ga_input,
                        gaddr_t ga_recurrence, gaddr_t ga_bias,
                        gaddr_t ga_init_h, gaddr_t ga_init_c, gaddr_t ga_cont,
                        gaddr_t ga_sigmoid_lut, gaddr_t ga_sigmoid_slope_lut,
                        gaddr_t ga_tanh_lut, gaddr_t ga_tanh_slope_lut,
                        gaddr_t ga_output, int seq_length, int num_dir,
                        int batch_size, int hidden_size, bool do_bias,
                        bool with_initial_h, bool with_initial_c,
                        bool with_cont, bool bidirectional) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_recurrence = ga_recurrence;
  this->ga_bias = ga_bias;
  this->ga_init_h = ga_init_h;
  this->ga_init_c = ga_init_c;
  this->ga_cont = ga_cont;
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
  this->with_initial_c = with_initial_c;
  this->with_cont = with_cont;
  this->bidirectional = bidirectional;
  this->fmt = CVK_FMT_BF16;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  this->lmem_used = 0;
  this->hidden_bytes = hidden_size * fmt_size;
  this->input_bytes = num_dir * 4 * hidden_bytes;
  this->recurrence_bytes = hidden_size * hidden_bytes;
  this->x_bytes = batch_size * hidden_bytes;
  this->x_gstride.row = input_bytes;
  this->h_gstride.row = hidden_bytes;
  if (with_cont && bidirectional) {
    llvm_unreachable("cont not support bidirectional!");
  }
  init_table();
}

void TgLstmKernel::compute_with_tiling(bool forward) {
  init_gaddr(forward);
  init_h0c0();
  for (int i = 0; i < seq_length; i++) {
    compute(i, forward);
  }
}

void TgLstmKernel::schedule() {
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

void cvi_backend_tg_bf16_lstm_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_recurrence, gaddr_t ga_bias, gaddr_t ga_initial_h,
    gaddr_t ga_initial_c, gaddr_t ga_cont, gaddr_t ga_sigmoid_lut,
    gaddr_t ga_sigmoid_slope_lut, gaddr_t ga_tanh_lut,
    gaddr_t ga_tanh_slope_lut, gaddr_t ga_output, int seq_len, int num_dir,
    int batch_size, int hidden_size, bool do_bias, bool with_initial_h,
    bool with_initial_c, bool with_cont, bool is_bidirectional) {
  TgLstmKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_recurrence, ga_bias, ga_initial_h,
              ga_initial_c, ga_cont, ga_sigmoid_lut, ga_sigmoid_slope_lut,
              ga_tanh_lut, ga_tanh_slope_lut, ga_output, seq_len, num_dir,
              batch_size, hidden_size, do_bias, with_initial_h, with_initial_c,
              with_cont, is_bidirectional);
  kernel.schedule();
}
