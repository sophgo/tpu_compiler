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
  cvk_tl_t tl_res, tl_right;
  matrix_to_tensor(&tl_res, ml_res);
  matrix_to_tensor(&tl_right, ml_right);
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_res;
  p.a = &tl_res;
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
  cvk_tl_t tl_res, tl_right;
  matrix_to_tensor(&tl_res, ml_res);
  matrix_to_tensor(&tl_right, ml_right);
  cvk_tiu_sub_param_t p = {0};
  p.res_high = 0;
  p.res_low = &tl_res;
  p.a_high = 0;
  p.a_low = &tl_res;
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

void TgGruKernel::assign_matrix(cvk_ml_t *ml_mem, uint32_t row, uint32_t col,
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
  auto w_shape = ctx.ml_default_shape(input_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto gate_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  auto w_size = ctx.lmem_matrix_to_size(w_shape, fmt, 1);
  auto r_size = ctx.lmem_matrix_to_size(r_shape, fmt, 1);
  auto b_size = ctx.lmem_matrix_to_size(b_shape, fmt, 1);
  auto g_size = ctx.lmem_matrix_to_size(gate_shape, fmt, 1);
  uint64_t total_size =
      lmem_used + 3 * w_size + 3 * r_size + 6 * b_size + 6 * g_size;
  if (total_size > (uint32_t)LOCAL_MEM_SIZE) {
    return true;
  }
  return false;
}

void TgGruKernel::tiling() {
  auto lmem_addr = lmem_used;
  for (step_size = hidden_size; step_size > 0; step_size--) {
    lmem_used = lmem_addr;
    addr_weight = lmem_used;
    auto w_shape = ctx.ml_default_shape(input_size, step_size, fmt);
    auto r_shape = ctx.ml_default_shape(step_size, step_size, fmt);
    auto w_size = ctx.lmem_matrix_to_size(w_shape, fmt, 1);
    auto r_size = ctx.lmem_matrix_to_size(r_shape, fmt, 1);
    lmem_used += (w_size > r_size ? w_size : r_size);
    addr_bias = lmem_used;
    auto b_shape = ctx.ml_default_shape(4 / fmt_size, step_size, fmt);
    auto b_size = ctx.lmem_matrix_to_size(b_shape, fmt, 1);
    lmem_used += b_size;

    auto gate_shape = ctx.ml_default_shape(batch_size, step_size, fmt);
    auto gate_size = ctx.lmem_matrix_to_size(gate_shape, fmt, 1);
    addr_work0 = lmem_used;
    lmem_used += 2 * gate_size;
    addr_work1 = lmem_used;

    auto work1_size = (step_size == hidden_size
                           ? gate_size
                           : ctx.lmem_ps32_matrix_to_size(gate_shape, fmt, 1));
    lmem_used += work1_size;
    addr_gate_z = lmem_used;
    lmem_used += gate_size;
    addr_gate_h = lmem_used;
    lmem_used += gate_size;
    // hidden state size
    uint32_t state_size = 0;
    for (int pos = 0; pos < hidden_size; pos += step_size) {
      int h = std::min(step_size, hidden_size - pos);
      auto h_shape = ctx.ml_default_shape(batch_size, h, fmt);
      state_size += ctx.lmem_matrix_to_size(h_shape, fmt, 1);
    }
    if (lmem_used + state_size <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
  }
  if (step_size == 0) {
    llvm::errs() << llvm::format(
        "Tilling GRU failed,seq:%d,batch:%d,input:%d,hidden:%d\n", seq_length,
        batch_size, input_size, hidden_size);
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
    ml_hiddens.emplace_back(ml_hidden);
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
    auto &ml_hidden = ml_hiddens[step];
    auto &tile = tiles[step];
    if (with_initial_h) {
      ctx.tdma_load_stride(&ml_hidden, ga_h0 + tile.pos_h * fmt_size, gstride);
    } else {
      cvk_tl_t tl_hidden;
      matrix_to_tensor(&tl_hidden, ml_hidden);
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = &tl_hidden;
      p.a = &tl_hidden; // rt
      p.b_is_const = 1;
      p.b_const.val = ctx.convert_fp32_to_bf16(0.0);
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      ctx.tiu_mul(&p);
    }
  }
}

void TgGruKernel::compute(int seq_idx) {
  // load input
  ctx.tdma_load(&ml_input, ga_input + seq_idx * input_bytes);
  for (int step = 0; step < step_num; step++) {
    auto &tile = tiles[step];
    auto &ml_hidden = ml_hiddens[step];
    cvk_ml_t ml_weight, ml_bias, ml_work0, ml_work1, ml_gate_z, ml_gate_h,
        ml_gate_r;
    cvk_ml_t ml_rr, ml_rz, ml_rh;
    int goffset = tile.pos_h * fmt_size;
    // gate z
    assign_matrix(&ml_weight, input_size, tile.h, addr_weight);
    ctx.tdma_load_stride(&ml_weight, ga_wz + goffset, gstride);
    assign_matrix(&ml_bias, 4 / fmt_size, tile.h, addr_bias);
    ctx.tdma_load_stride(&ml_bias, ga_wbz + goffset, gstride);
    assign_matrix(&ml_work0, batch_size, tile.h, addr_work0);
    matrix_mul(ml_work0, ml_input, ml_weight, ml_bias);

    assign_matrix(&ml_work1, batch_size, tile.h, addr_work1);
    ctx.tdma_load_stride(&ml_bias, ga_rbz + goffset, gstride);
    for (int i = 0; i < step_num; i++) {
      assign_matrix(&ml_rz, tiles[i].h, tile.h, addr_weight);
      ctx.tdma_load_stride(
          &ml_rz, ga_rz + tiles[i].pos_h * hidden_bytes + goffset, gstride);
      matrix_mul(ml_work1, ml_hiddens[i], ml_rz, ml_bias, ps32_mode(i));
    }
    eltwise_add(ml_work1, ml_work0);
    assign_matrix(&ml_gate_z, batch_size, tile.h, addr_gate_z);
    sigmoid(ml_gate_z, ml_work1, ml_work0);
    // gate r
    assign_matrix(&ml_weight, input_size, tile.h, addr_weight);
    ctx.tdma_load_stride(&ml_weight, ga_wr + goffset, gstride);
    ctx.tdma_load_stride(&ml_bias, ga_wbr + goffset, gstride);
    matrix_mul(ml_work0, ml_input, ml_weight, ml_bias);

    ctx.tdma_load_stride(&ml_bias, ga_rbr + goffset, gstride);
    for (int i = 0; i < step_num; i++) {
      assign_matrix(&ml_rr, tiles[i].h, tile.h, addr_weight);
      ctx.tdma_load_stride(
          &ml_rr, ga_rr + tiles[i].pos_h * hidden_bytes + goffset, gstride);
      matrix_mul(ml_work1, ml_hiddens[i], ml_rr, ml_bias, ps32_mode(i));
    }
    eltwise_add(ml_work1, ml_work0);
    assign_matrix(&ml_gate_r, batch_size, tile.h, addr_gate_h);
    sigmoid(ml_gate_r, ml_work1, ml_work0);
    // gate h
    assign_matrix(&ml_weight, input_size, tile.h, addr_weight);
    ctx.tdma_load_stride(&ml_weight, ga_wh + goffset, gstride);
    ctx.tdma_load_stride(&ml_bias, ga_wbh + goffset, gstride);
    matrix_mul(ml_work0, ml_input, ml_weight, ml_bias);

    ctx.tdma_load_stride(&ml_bias, ga_rbh + goffset, gstride);
    for (int i = 0; i < step_num; i++) {
      assign_matrix(&ml_rh, tiles[i].h, tile.h, addr_weight);
      ctx.tdma_load_stride(
          &ml_rh, ga_rh + tiles[i].pos_h * hidden_bytes + goffset, gstride);
      matrix_mul(ml_work1, ml_hiddens[i], ml_rh, ml_bias, ps32_mode(i));
    }
    eltwise_mul(ml_work1, ml_gate_r);
    eltwise_add(ml_work1, ml_work0);
    assign_matrix(&ml_gate_h, batch_size, tile.h, addr_gate_h);
    tanh(ml_gate_h, ml_work1, ml_work0);
    eltwise_mul(ml_hidden, ml_gate_z);
    eltwise_add(ml_hidden, ml_gate_h);
    eltwise_mul(ml_gate_z, ml_gate_h);
    eltwise_sub(ml_hidden, ml_gate_z);
    ctx.tdma_store_stride(&ml_hidden,
                          ga_store + seq_idx * num_dir * gate_bytes + goffset,
                          gstride);
  }
}

void TgGruKernel::compute_without_tiling(bool forward) {
  auto lmem_used_backup = lmem_used;
  init_gaddr(forward);
  // load weight/recurrence/bias
  auto w_shape = ctx.ml_default_shape(input_size, hidden_size, fmt);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  auto gate_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  cvk_ml_t ml_wz, ml_wr, ml_wh, ml_rz, ml_rr, ml_rh, ml_wbz, ml_wbr, ml_wbh,
      ml_rbz, ml_rbr, ml_rbh;
  assign_matrix(&ml_wz, w_shape);
  assign_matrix(&ml_wr, w_shape);
  assign_matrix(&ml_wh, w_shape);
  assign_matrix(&ml_rz, r_shape);
  assign_matrix(&ml_rr, r_shape);
  assign_matrix(&ml_rh, r_shape);
  assign_matrix(&ml_wbz, b_shape);
  assign_matrix(&ml_wbr, b_shape);
  assign_matrix(&ml_wbh, b_shape);
  assign_matrix(&ml_rbz, b_shape);
  assign_matrix(&ml_rbr, b_shape);
  assign_matrix(&ml_rbh, b_shape);

  ctx.tdma_load(&ml_wz, ga_wz);
  ctx.tdma_load(&ml_wr, ga_wr);
  ctx.tdma_load(&ml_wh, ga_wh);
  ctx.tdma_load(&ml_rz, ga_rz);
  ctx.tdma_load(&ml_rr, ga_rr);
  ctx.tdma_load(&ml_rh, ga_rh);
  ctx.tdma_load(&ml_wbz, ga_wbz);
  ctx.tdma_load(&ml_wbr, ga_wbr);
  ctx.tdma_load(&ml_wbh, ga_wbh);
  ctx.tdma_load(&ml_rbz, ga_rbz);
  ctx.tdma_load(&ml_rbr, ga_rbr);
  ctx.tdma_load(&ml_rbh, ga_rbh);
  // load initial_h if exist or clear to zeros
  cvk_ml_t ml_hidden;
  cvk_tl_t tl_hidden;
  assign_matrix(&ml_hidden, gate_shape);
  matrix_to_tensor(&tl_hidden, ml_hidden);
  if (with_initial_h) {
    ctx.tdma_load(&ml_hidden, ga_h0);
  } else {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_hidden;
    p.a = &tl_hidden; // rt
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(0.0);
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  }
  auto working_shape = ctx.ml_default_shape(batch_size * 2, hidden_size, fmt);
  cvk_ml_t ml_working, ml_work0, ml_work1, ml_gate_z, ml_gate_h;
  assign_matrix(&ml_working, working_shape);
  assign_matrix(&ml_work0, batch_size, hidden_size, ml_working.start_address);
  assign_matrix(&ml_work1, gate_shape);
  assign_matrix(&ml_gate_z, gate_shape);
  assign_matrix(&ml_gate_h, gate_shape);
  for (int i = 0; i < seq_length; i++) {
    int seq_idx = i;
    if (forward == false) {
      seq_idx = seq_length - i - 1;
    }
    ctx.tdma_load(&ml_input, ga_input + seq_idx * input_bytes);
    // gate_z => ml_gate_z
    matrix_mul(ml_work0, ml_input, ml_wz, ml_wbz);
    matrix_mul(ml_work1, ml_hidden, ml_rz, ml_rbz);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_gate_z, ml_work1, ml_working);
    // gate_r => ml_gate_h (tmp)
    matrix_mul(ml_work0, ml_input, ml_wr, ml_wbr);
    matrix_mul(ml_work1, ml_hidden, ml_rr, ml_rbr);
    eltwise_add(ml_work1, ml_work0);
    sigmoid(ml_gate_h, ml_work1, ml_working);
    // gate_h => ml_gate_h
    matrix_mul(ml_work1, ml_hidden, ml_rh, ml_rbh);
    eltwise_mul(ml_gate_h, ml_work1);
    matrix_mul(ml_work1, ml_input, ml_wh, ml_wbh);
    eltwise_add(ml_work1, ml_gate_h);
    tanh(ml_gate_h, ml_work1, ml_working);
    // H = (1-gate_z)*gate_h + gate_z * hidden
    eltwise_mul(ml_hidden, ml_gate_z);
    eltwise_add(ml_hidden, ml_gate_h);
    eltwise_mul(ml_gate_z, ml_gate_h);
    eltwise_sub(ml_hidden, ml_gate_z);
    ctx.tdma_store(&ml_hidden, ga_store + seq_idx * num_dir * gate_bytes);
  }
  lmem_used = lmem_used_backup;
}

void TgGruKernel::init_gaddr(bool forward) {
  if (forward) {
    ga_wz = ga_weight;
    ga_rz = ga_recurrence;
    ga_wbz = ga_bias;
    ga_store = ga_output;
    ga_h0 = ga_init_h;
  } else {
    ga_wz = ga_weight + weight_bytes * 3;
    ga_rz = ga_recurrence + recurrence_bytes * 3;
    ga_wbz = ga_bias + hidden_bytes * 6;
    ga_store = ga_output + gate_bytes;
    ga_h0 = ga_init_h + gate_bytes;
  }

  ga_wr = ga_wz + weight_bytes;
  ga_wh = ga_wr + weight_bytes;
  ga_rr = ga_rz + recurrence_bytes;
  ga_rh = ga_rr + recurrence_bytes;
  ga_wbr = ga_wbz + hidden_bytes;
  ga_wbh = ga_wbr + hidden_bytes;
  ga_rbz = ga_wbh + hidden_bytes;
  ga_rbr = ga_rbz + hidden_bytes;
  ga_rbh = ga_rbr + hidden_bytes;
}

void TgGruKernel::init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight,
                       gaddr_t ga_recurrence, gaddr_t ga_bias,
                       gaddr_t ga_init_h, gaddr_t ga_sigmoid_lut,
                       gaddr_t ga_sigmoid_slope_lut, gaddr_t ga_tanh_lut,
                       gaddr_t ga_tanh_slope_lut, gaddr_t ga_output,
                       int seq_length, int batch_size, int input_size,
                       int hidden_size, bool do_bias, bool with_initial_h,
                       bool linear_before_reset, bool bidirectional) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_weight = ga_weight;
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
  this->input_size = input_size;
  this->hidden_size = hidden_size;
  this->do_bias = do_bias;
  this->with_initial_h = with_initial_h;
  this->linear_before_reset = linear_before_reset;
  this->bidirectional = bidirectional;
  this->fmt = CVK_FMT_BF16;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  this->lmem_used = 0;
  this->input_shape = ctx.ml_default_shape(batch_size, input_size, fmt);
  this->hidden_bytes = hidden_size * fmt_size;
  this->input_bytes = batch_size * input_size * fmt_size;
  this->weight_bytes = input_size * hidden_size * fmt_size;
  this->recurrence_bytes = hidden_size * hidden_size * fmt_size;
  this->gate_bytes = batch_size * hidden_size * fmt_size;
  this->gstride = {(uint32_t)(hidden_bytes)};
  this->num_dir = (bidirectional ? 2 : 1);
  assert(linear_before_reset == true); // support later
  init_table();
  assign_matrix(&ml_input, input_shape);
}

void TgGruKernel::compute_with_tiling(bool forward) {
  init_gaddr(forward);
  init_h0();
  for (int i = 0; i < seq_length; i++) {
    int seq_idx = i;
    if (forward == false) {
      seq_idx = seq_length - i - 1;
    }
    compute(seq_idx);
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
    gaddr_t ga_weight, gaddr_t ga_recurrence, gaddr_t ga_bias,
    gaddr_t ga_initial_h, gaddr_t ga_sigmoid_lut, gaddr_t ga_sigmoid_slope_lut,
    gaddr_t ga_tanh_lut, gaddr_t ga_tanh_slope_lut, gaddr_t ga_output,
    int seq_len, int batch_size, int input_size, int hidden_size, bool do_bias,
    bool with_initial_h, bool is_linear_before_reset, bool is_bidirectional) {
  TgGruKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_recurrence, ga_bias,
              ga_initial_h, ga_sigmoid_lut, ga_sigmoid_slope_lut, ga_tanh_lut,
              ga_tanh_slope_lut, ga_output, seq_len, batch_size, input_size,
              hidden_size, do_bias, with_initial_h, is_linear_before_reset,
              is_bidirectional);
  kernel.schedule();
}
