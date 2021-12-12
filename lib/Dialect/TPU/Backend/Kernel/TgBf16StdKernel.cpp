/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgBf16StdKenrel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_std_kernel"

#define ASSERT(x) assert(x)

void cvi_backend_tg_bf16_std_kernel(const CviBackendContext &ctx,
                                    uint32_t layer_id, gaddr_t ga_input,
                                    gaddr_t ga_table, gaddr_t ga_mantissa_table,
                                    gaddr_t ga_output, int outer_size,
                                    int std_size, bool unbiased) {
  int h, w;
  bool ret = ctx.size_to_hw(std_size, h, w);
  if (ret == false) {
    llvm::errs() << llvm::format("Std inner size[%d] is too large\n", std_size);
    assert(0);
  }
  cvk_fmt_t fmt = CVK_FMT_BF16;
  uint32_t lmem_used = 0;
  cvk_tl_shape_t table_shape = ctx.lut_table_shape(fmt);
  cvk_tl_t *tl_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_lut_mantissa = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  ctx.tdma_load(tl_lut, ga_table);
  ctx.tdma_load(tl_lut_mantissa, ga_mantissa_table);
  lmem_used += 2 * ctx.lmem_tensor_to_size(table_shape, fmt, 1);

  int c_step = std::min(outer_size, MAX_CHANNEL);
  while (c_step > 0) {
    // for input
    uint32_t mem_need = ctx.lmem_tensor_to_size(1, c_step, h, w, fmt, 1);
    // for mean and var
    mem_need += 3 * ctx.lmem_tensor_to_size(1, c_step, 1, 1, fmt, 1);
    if (lmem_used + mem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (c_step % NPU_NUM != 0) {
      c_step -= c_step % NPU_NUM;
    } else {
      c_step -= NPU_NUM;
    }
  }
  if (c_step == 0) {
    llvm::errs() << llvm::format("Tilling Std failed, src shape:[1,%d,%d,%d]\n",
                                 outer_size, h, w);
    assert(0);
  }
  ctx.parallel_disable();
  float fix_div = unbiased ? (float)std_size / ((float)std_size - 1.0f) : 1.0f;
  for (int c_pos = 0; c_pos < outer_size; c_pos += c_step) {
    int c = std::min(c_step, outer_size - c_pos);
    auto input_shape = ctx.tl_shape_t4(1, c, h, w);
    auto *tl_input = ctx.lmem_alloc_tensor(input_shape, fmt, 1);
    uint64_t in_offset = c_pos * h * w * ctx.bytesize_of_fmt(fmt);
    uint64_t out_offset = c_pos * ctx.bytesize_of_fmt(fmt);
    ctx.tdma_load(tl_input, ga_input + in_offset);
    auto mean_shape = ctx.tl_shape_t4(1, c, 1, 1);
    cvk_tl_t *tl_mean = ctx.lmem_alloc_tensor(mean_shape, fmt, 1);
    // input => mean
    cvk_tiu_average_pooling_param_t p1 = {0};
    p1.ofmap = tl_mean;
    p1.ifmap = tl_input;
    p1.kh = h;
    p1.kw = w;
    p1.ins_h = 0;
    p1.ins_last_h = 0;
    p1.ins_w = 0;
    p1.ins_last_w = 0;
    p1.stride_h = 1;
    p1.stride_w = 1;
    p1.avg_pooling_const = ctx.convert_fp32_to_bf16(1.0);
    p1.ins_val = 0;
    p1.ins_fp = ctx.convert_fp32_to_bf16(0.0);
    p1.layer_id = layer_id;
    ctx.tiu_average_pooling(&p1);
    // expand [1,c,1,1] =>[1,c,h,w]
    tl_mean->shape = input_shape;
    tl_mean->stride.w = 0;
    tl_mean->stride.h = 0;
    cvk_tiu_sub_param_t p3 = {0};
    p3.res_high = 0;
    p3.res_low = tl_input;
    p3.a_high = 0;
    p3.a_low = tl_input;
    p3.b_high = 0;
    p3.b_low = tl_mean;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    ctx.tiu_sub(&p3);
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = tl_input;
    p4.a = tl_input;
    p4.b = tl_input;
    p4.b_is_const = 0;
    p4.rshift_bits = 0;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    ctx.tiu_mul(&p4);
    tl_mean->shape = mean_shape;
    tl_mean->stride = ctx.tl_default_stride(mean_shape, fmt, 1);
    cvk_tiu_average_pooling_param_t p5 = {0};
    p5.ofmap = tl_mean;
    p5.ifmap = tl_input;
    p5.kh = h;
    p5.kw = w;
    p5.ins_h = 0;
    p5.ins_last_h = 0;
    p5.ins_w = 0;
    p5.ins_last_w = 0;
    p5.stride_h = 1;
    p5.stride_w = 1;
    p5.avg_pooling_const = ctx.convert_fp32_to_bf16(fix_div);
    p5.ins_val = 0;
    p5.ins_fp = ctx.convert_fp32_to_bf16(0.0);
    p5.layer_id = layer_id;
    ctx.tiu_average_pooling(&p5);

    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(mean_shape, fmt, 1);
    cvk_tl_t *tl_buf = ctx.lmem_alloc_tensor(mean_shape, fmt, 1);

    cvk_tiu_bf16_lookup_interp_table_param_t p6 = {0};
    p6.ifmap = tl_mean;
    p6.buf = tl_buf;
    p6.tbl_answer = tl_lut;
    p6.tbl_answer_mantissa = tl_lut_mantissa;
    p6.ofmap = tl_output;
    p6.is_scientific = 1;
    ctx.tiu_bf16_lookup_interp_table(&p6);

    ctx.tdma_store(tl_output, ga_output + out_offset);
    ctx.lmem_free_tensor(tl_buf);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_mean);
    ctx.lmem_free_tensor(tl_input);
  }
  ctx.lmem_free_tensor(tl_lut_mantissa);
  ctx.lmem_free_tensor(tl_lut);
}