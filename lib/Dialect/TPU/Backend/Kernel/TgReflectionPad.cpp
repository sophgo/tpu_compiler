/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgCropKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "kernel_reflectionpad"
static void matrix_for_tiu(const CviBackendContext &ctx, cvk_ml_t *matrix,
                           cvk_fmt_t fmt) {
  if (matrix->shape.w < ctx.tiu_eu_num(fmt) && matrix->shape.c > 1) {
    matrix->shape.w = ctx.tiu_eu_num(fmt);
    matrix->stride = ctx.ml_default_stride(matrix->shape, fmt, 1);
  }
}

static void matrix_mul(const CviBackendContext &ctx, uint32_t layer_id,
                       const cvk_ml_t *ml_res, const cvk_ml_t *ml_left,
                       const cvk_ml_t *ml_right, cvk_fmt_t fmt) {
  cvk_ml_t ml_res_ = *ml_res;
  cvk_ml_t ml_left_ = *ml_left;
  cvk_ml_t ml_right_ = *ml_right;
  matrix_for_tiu(ctx, &ml_res_, fmt);
  matrix_for_tiu(ctx, &ml_left_, fmt);
  matrix_for_tiu(ctx, &ml_right_, fmt);
  cvk_tiu_matrix_multiplication_param_t p = {0};
  p.res = &ml_res_;
  p.left = &ml_left_;
  p.right = &ml_right_;
  p.res_is_int8 = 1; // dont care in bf16
  p.bias = nullptr;
  p.ps32_mode = 0;
  p.layer_id = layer_id;
  ctx.tiu_matrix_multiplication(&p);
}

static void do_reflection(const CviBackendContext &ctx, uint32_t layer_id,
                          gaddr_t src_addr, gaddr_t dst_addr,
                          gaddr_t weight_addr, cvk_mg_stride_t &src_gstride,
                          cvk_mg_stride_t &dst_gstride, int outer_size, int pad,
                          cvk_fmt_t fmt) {
  if (pad == 0) {
    return;
  }
  auto wshape = ctx.ml_default_shape(pad, pad, fmt);
  auto ml_weight = ctx.lmem_alloc_matrix(wshape, fmt, 1);
  auto lmem_used = ctx.lmem_matrix_to_size(wshape, fmt, 1);
  int step;
  for (step = outer_size; step > 0; step--) {
    auto ishape = ctx.ml_default_shape(step, pad, fmt);
    auto lmem_need = ctx.lmem_matrix_to_size(ishape, fmt, 1) * 2;
    if (lmem_need + lmem_used <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
  }
  if (step == 0) {
    llvm_unreachable("pad slice failed");
  }
  ctx.tdma_load(ml_weight, weight_addr);
  for (int pos = 0; pos < outer_size; pos += step) {
    int in_offset = pos * src_gstride.row;
    int out_offset = pos * dst_gstride.row;
    int N = std::min(step, outer_size - pos);
    auto ishape = ctx.ml_default_shape(N, pad, fmt);
    auto ml_input = ctx.lmem_alloc_matrix(ishape, fmt, 1);
    auto ml_output = ctx.lmem_alloc_matrix(ishape, fmt, 1);
    ctx.tdma_load_stride(ml_input, src_addr + in_offset, src_gstride);
    matrix_mul(ctx, layer_id, ml_output, ml_input, ml_weight, fmt);
    ctx.tdma_store_stride(ml_output, dst_addr + out_offset, dst_gstride);
    ctx.lmem_free_matrix(ml_output);
    ctx.lmem_free_matrix(ml_input);
  }
  ctx.lmem_free_matrix(ml_weight);
}

void cvi_backend_tg_reflectionpad_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, gaddr_t ga_left, gaddr_t ga_right, int outer_size,
    int working_size, std::vector<int> &pads, cvk_fmt_t fmt) {
  ctx.set_layer_id(layer_id);
  auto fmt_size = ctx.bytesize_of_fmt(fmt);
  // copy middle first
  auto src_gaddr = ga_input;
  auto src_shape = ctx.tg_shape_t4(1, outer_size, 1, working_size);
  auto src_stride = ctx.tg_default_stride(src_shape, fmt);
  auto dst_gaddr = ga_output + pads[0] * fmt_size;
  int out_working_size = working_size + pads[0] + pads[1];
  auto dst_shape = ctx.tg_shape_t4(1, outer_size, 1, out_working_size);
  auto dst_stride = ctx.tg_default_stride(dst_shape, fmt);
  ctx.tdma_g2g_tensor_copy(src_gaddr, src_shape, src_stride, fmt, dst_gaddr,
                           src_shape, dst_stride, fmt);

  cvk_mg_stride_t in_gstride = {.row = (uint32_t)working_size * fmt_size};
  cvk_mg_stride_t out_gstride = {.row = (uint32_t)out_working_size * fmt_size};
  do_reflection(ctx, layer_id, ga_input + fmt_size, ga_output, ga_left,
                in_gstride, out_gstride, outer_size, pads[0], fmt);
  ga_input += (working_size - pads[1] - 1) * fmt_size;
  ga_output += (out_working_size - pads[1]) * fmt_size;
  do_reflection(ctx, layer_id, ga_input, ga_output, ga_right, in_gstride,
                out_gstride, outer_size, pads[1], fmt);
}