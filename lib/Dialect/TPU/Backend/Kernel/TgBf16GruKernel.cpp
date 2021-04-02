/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_gru.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_gru_kernel"

#define ASSERT(x) assert(x)

static void matrix_to_tensor(const CviBackendContext &ctx, cvk_tl_t *tensor,
                             const cvk_ml_t *matrix) {
  cvk_tl_shape_t shape = {matrix->shape.n, matrix->shape.c, 1,
                          ctx.tiu_eu_num(CVK_FMT_BF16)};
  ctx.lmem_init_tensor(tensor, shape, CVK_FMT_BF16, 1);
  tensor->start_address = matrix->start_address;
}

static void matrix_for_tiu(const CviBackendContext &ctx, cvk_ml_t &matrix) {
  if (matrix.shape.w < ctx.tiu_eu_num(CVK_FMT_BF16) && matrix.shape.c > 1) {
    matrix.shape.w = ctx.tiu_eu_num(CVK_FMT_BF16);
    matrix.stride = ctx.ml_default_stride(matrix.shape, CVK_FMT_BF16, 1);
  }
}

static void matrix_mul(const CviBackendContext &ctx, int layer_id,
                       cvk_ml_t *ml_res, cvk_ml_t *ml_left, cvk_ml_t *ml_right,
                       cvk_ml_t *ml_bias) {
  cvk_ml_t ml_res_ = *ml_res;
  cvk_ml_t ml_left_ = *ml_left;
  cvk_ml_t ml_right_ = *ml_right;
  cvk_ml_t ml_bias_ = *ml_bias;
  matrix_for_tiu(ctx, ml_res_);
  matrix_for_tiu(ctx, ml_left_);
  matrix_for_tiu(ctx, ml_right_);
  matrix_for_tiu(ctx, ml_bias_);
  cvk_tiu_matrix_multiplication_param_t p = {0};
  p.res = &ml_res_;
  p.left = &ml_left_;
  p.right = &ml_right_;
  p.bias = &ml_bias_;
  p.ps32_mode = 0;
  p.layer_id = layer_id;
  ctx.tiu_matrix_multiplication(&p);
}

static void eltwise_mul(const CviBackendContext &ctx, int layer_id,
                        cvk_ml_t *ml_res, cvk_ml_t *ml_left) {
  cvk_tl_t tl_res, tl_left;
  matrix_to_tensor(ctx, &tl_res, ml_res);
  matrix_to_tensor(ctx, &tl_left, ml_left);
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_res;
  p.a = &tl_res;
  p.b_is_const = 0;
  p.b = &tl_left;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);
}

static void eltwise_add(const CviBackendContext &ctx, int layer_id,
                        cvk_ml_t *ml_res, cvk_ml_t *ml_left) {
  cvk_tl_t tl_res, tl_left;
  matrix_to_tensor(ctx, &tl_res, ml_res);
  matrix_to_tensor(ctx, &tl_left, ml_left);
  cvk_tiu_add_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_res;
  p.a_high = nullptr;
  p.a_low = &tl_res;
  p.b_is_const = false;
  p.b.high = nullptr;
  p.b.low = &tl_left;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_add(&p);
}

static void eltwise_sub(const CviBackendContext &ctx, int layer_id,
                        cvk_ml_t *ml_res, cvk_ml_t *ml_left) {
  cvk_tl_t tl_res, tl_left;
  matrix_to_tensor(ctx, &tl_res, ml_res);
  matrix_to_tensor(ctx, &tl_left, ml_left);
  cvk_tiu_sub_param_t p = {0};
  p.res_high = 0;
  p.res_low = &tl_res;
  p.a_high = 0;
  p.a_low = &tl_res;
  p.b_high = 0;
  p.b_low = &tl_left;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  ctx.tiu_sub(&p);
}

void cvi_backend_tg_bf16_gru_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_recurrence, gaddr_t ga_bias,
    gaddr_t ga_initial_h, gaddr_t ga_sigmoid_lut, gaddr_t ga_sigmoid_slope_lut,
    gaddr_t ga_tanh_lut, gaddr_t ga_tanh_slope_lut, gaddr_t ga_output,
    int seq_len, int batch_size, int input_size, int hidden_size, bool do_bias,
    bool with_initial_h, bool is_linear_before_reset, bool is_bidirectional) {
  cvk_fmt_t fmt = CVK_FMT_BF16;
  int fmt_size = ctx.bytesize_of_fmt(fmt);
  // load sigmoid table and tanh table
  auto table_shape = ctx.lut_table_shape(fmt);
  cvk_tl_t *tl_sigmoid_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_sigmoid_slope_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  ctx.tdma_load(tl_sigmoid_lut, ga_sigmoid_lut);
  ctx.tdma_load(tl_sigmoid_slope_lut, ga_sigmoid_slope_lut);
  cvk_tl_t *tl_tanh_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_tanh_slope_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  ctx.tdma_load(tl_tanh_lut, ga_tanh_lut);
  ctx.tdma_load(tl_tanh_slope_lut, ga_tanh_slope_lut);
  // load weight/recurrence/bias
  auto w_shape = ctx.ml_default_shape(input_size, hidden_size, fmt);
  cvk_ml_t *ml_wz = ctx.lmem_alloc_matrix(w_shape, fmt, 1);
  cvk_ml_t *ml_wr = ctx.lmem_alloc_matrix(w_shape, fmt, 1);
  cvk_ml_t *ml_wh = ctx.lmem_alloc_matrix(w_shape, fmt, 1);
  int w_size = input_size * hidden_size * fmt_size;
  ctx.tdma_load(ml_wz, ga_weight);
  ctx.tdma_load(ml_wr, ga_weight + w_size);
  ctx.tdma_load(ml_wh, ga_weight + 2 * w_size);
  auto r_shape = ctx.ml_default_shape(hidden_size, hidden_size, fmt);
  cvk_ml_t *ml_rz = ctx.lmem_alloc_matrix(r_shape, fmt, 1);
  cvk_ml_t *ml_rr = ctx.lmem_alloc_matrix(r_shape, fmt, 1);
  cvk_ml_t *ml_rh = ctx.lmem_alloc_matrix(r_shape, fmt, 1);
  int r_size = hidden_size * hidden_size * fmt_size;
  ctx.tdma_load(ml_rz, ga_recurrence);
  ctx.tdma_load(ml_rr, ga_recurrence + r_size);
  ctx.tdma_load(ml_rh, ga_recurrence + 2 * r_size);
  auto b_shape = ctx.ml_default_shape(4 / fmt_size, hidden_size, fmt);
  cvk_ml_t *ml_wbz = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  cvk_ml_t *ml_wbr = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  cvk_ml_t *ml_wbh = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  cvk_ml_t *ml_rbz = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  cvk_ml_t *ml_rbr = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  cvk_ml_t *ml_rbh = ctx.lmem_alloc_matrix(b_shape, fmt, 1);
  int b_size = hidden_size * fmt_size;
  ctx.tdma_load(ml_wbz, ga_bias);
  ctx.tdma_load(ml_wbr, ga_bias + 1 * b_size);
  ctx.tdma_load(ml_wbh, ga_bias + 2 * b_size);
  ctx.tdma_load(ml_rbz, ga_bias + 3 * b_size);
  ctx.tdma_load(ml_rbr, ga_bias + 4 * b_size);
  ctx.tdma_load(ml_rbh, ga_bias + 5 * b_size);
  // load initial_h if exist or clear to zeros
  auto gate_shape = ctx.ml_default_shape(batch_size, hidden_size, fmt);
  cvk_ml_t *ml_hidden = ctx.lmem_alloc_matrix(gate_shape, fmt, 1);
  cvk_tl_t tl_hidden;
  matrix_to_tensor(ctx, &tl_hidden, ml_hidden);
  if (with_initial_h) {
    ctx.tdma_load(ml_hidden, ga_initial_h);
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
  auto working_shape = ctx.ml_default_shape(batch_size*2, hidden_size, fmt);
  cvk_ml_t *ml_working = ctx.lmem_alloc_matrix(working_shape, fmt, 1);
  cvk_ml_t *ml_work0 = ctx.lmem_alloc_matrix(gate_shape, fmt, 1);
  cvk_ml_t *ml_work1 = ctx.lmem_alloc_matrix(gate_shape, fmt, 1);
  cvk_ml_t *ml_gate_z = ctx.lmem_alloc_matrix(gate_shape, fmt, 1);
  cvk_ml_t *ml_gate_h = ctx.lmem_alloc_matrix(gate_shape, fmt, 1);
  for (int i = 0; i < seq_len; i++) {
    auto input_shape = ctx.ml_default_shape(batch_size, input_size, fmt);
    cvk_ml_t *ml_input = ctx.lmem_alloc_matrix(input_shape, fmt, 1);
    int i_size = input_size * batch_size * fmt_size;
    ctx.tdma_load(ml_input, ga_input + i * i_size);
    // gate_z => ml_gate_z
    matrix_mul(ctx, layer_id, ml_work0, ml_input, ml_wz, ml_wbz);
    matrix_mul(ctx, layer_id, ml_work1, ml_hidden, ml_rz, ml_rbz);
    eltwise_add(ctx, layer_id, ml_work1, ml_work0);
    cvi_backend_bf16_tl_lut_slope_method(
        ctx, layer_id, ml_work1->start_address, ml_gate_z->start_address,
        ml_working->start_address, tl_sigmoid_lut->start_address,
        tl_sigmoid_slope_lut->start_address, -8, 8, false, gate_shape.n,
        gate_shape.c, 1, ctx.tiu_eu_num(fmt));
    // gate_r => ml_work0
    matrix_mul(ctx, layer_id, ml_work0, ml_input, ml_wr, ml_wbr);
    matrix_mul(ctx, layer_id, ml_work1, ml_hidden, ml_rr, ml_rbr);
    eltwise_add(ctx, layer_id, ml_work1, ml_work0);
    cvi_backend_bf16_tl_lut_slope_method(
        ctx, layer_id, ml_work1->start_address, ml_work0->start_address,
        ml_working->start_address, tl_sigmoid_lut->start_address,
        tl_sigmoid_slope_lut->start_address, -8, 8, false, gate_shape.n,
        gate_shape.c, 1, ctx.tiu_eu_num(fmt));
    // gate_h => ml_gate_h
    matrix_mul(ctx, layer_id, ml_work1, ml_hidden, ml_rh, ml_rbh);
    eltwise_mul(ctx, layer_id, ml_work0, ml_work1);
    matrix_mul(ctx, layer_id, ml_work1, ml_input, ml_wh, ml_wbh);
    eltwise_add(ctx, layer_id, ml_work0, ml_work1);
    cvi_backend_bf16_tl_lut_slope_method(
        ctx, layer_id, ml_work0->start_address, ml_gate_h->start_address,
        ml_working->start_address, tl_tanh_lut->start_address,
        tl_tanh_slope_lut->start_address, -8, 8, false, gate_shape.n,
        gate_shape.c, 1, ctx.tiu_eu_num(fmt));
    // H = (1-gate_z)*gate_h + gate_z * hidden
    eltwise_mul(ctx, layer_id, ml_hidden, ml_gate_z);
    eltwise_add(ctx, layer_id, ml_hidden, ml_gate_h);
    eltwise_mul(ctx, layer_id, ml_gate_z, ml_gate_h);
    eltwise_sub(ctx, layer_id, ml_hidden, ml_gate_z);
    ctx.tdma_store(ml_hidden,
                   ga_output + i * batch_size * hidden_size * fmt_size);
    ctx.lmem_free_matrix(ml_input);
  }

  ctx.lmem_free_matrix(ml_gate_h);
  ctx.lmem_free_matrix(ml_gate_z);
  ctx.lmem_free_matrix(ml_work1);
  ctx.lmem_free_matrix(ml_work0);
  ctx.lmem_free_matrix(ml_working);
  ctx.lmem_free_matrix(ml_hidden);
  ctx.lmem_free_matrix(ml_rbh);
  ctx.lmem_free_matrix(ml_rbr);
  ctx.lmem_free_matrix(ml_rbz);
  ctx.lmem_free_matrix(ml_wbh);
  ctx.lmem_free_matrix(ml_wbr);
  ctx.lmem_free_matrix(ml_wbz);
  ctx.lmem_free_matrix(ml_rh);
  ctx.lmem_free_matrix(ml_rr);
  ctx.lmem_free_matrix(ml_rz);
  ctx.lmem_free_matrix(ml_wh);
  ctx.lmem_free_matrix(ml_wr);
  ctx.lmem_free_matrix(ml_wz);
  ctx.lmem_free_tensor(tl_tanh_slope_lut);
  ctx.lmem_free_tensor(tl_tanh_lut);
  ctx.lmem_free_tensor(tl_sigmoid_slope_lut);
  ctx.lmem_free_tensor(tl_sigmoid_lut);
}