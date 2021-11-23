/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReduceKernel.cpp
 * Description:
 */

#include "TgReduceKernel.hpp"
#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <numeric>

void TgReduceKernel::reshape(std::vector<int64_t> shape,
                             std::vector<int32_t> axes) {
  int num_dims = shape.size();
  int num_axes = axes.size();
  assert(num_axes > 0);
  assert(axes[0] < num_dims && axes[0] >= 0);
  for (int i = 1; i < num_axes; i++) {
    assert(axes[i] == axes[i - 1] + 1);
    assert(axes[i] < num_dims);
  }
  int start_axis = axes[0];
  int end_axis = axes[num_axes - 1] + 1;
  int outer_dims = std::accumulate(shape.begin(), shape.begin() + start_axis, 1,
                                   std::multiplies<int64_t>());
  int axis_dims =
      std::accumulate(shape.begin() + start_axis, shape.begin() + end_axis, 1,
                      std::multiplies<int64_t>());
  int inner_dims = std::accumulate(shape.begin() + end_axis, shape.end(), 1,
                                   std::multiplies<int64_t>());
  if (inner_dims == 1) {
    end_reduce = true;
    n = 1;
    c = outer_dims;
    ctx.size_to_hw(axis_dims, h, w);
    kh = h;
    kw = w;
    in_gstride = ctx.tg_default_stride(c, h, w, fmt);
    out_gstride = ctx.tg_default_stride(c, 1, 1, fmt);
  } else {
    end_reduce = false;
    n = 1;
    c = outer_dims;
    h = axis_dims;
    w = inner_dims;
    kw = 1;
    kh = h;
    in_gstride = ctx.tg_default_stride(c, h, w, fmt);
    out_gstride = ctx.tg_default_stride(c, 1, w, fmt);
  }
}

void TgReduceKernel::reduce_max() {
  cvk_tiu_max_pooling_param_t param = {0};
  param.ofmap = &tl_ofmap;
  param.ifmap = &tl_ifmap;
  param.kh = kh;
  param.kw = kw;
  param.pad_top = 0;
  param.pad_bottom = 0;
  param.pad_left = 0;
  param.pad_right = 0;
  param.stride_h = 1;
  param.stride_w = 1;
  param.ins_val = -128;
  param.ins_fp = 0xff7f;
  param.layer_id = layer_id;
  ctx.tiu_max_pooling(&param);
}

void TgReduceKernel::reduce_min() {
  cvk_tiu_mul_param_t p_in = {0};
  p_in.res_high = NULL;
  p_in.res_low = &tl_ifmap;
  p_in.a = &tl_ifmap;
  p_in.b_is_const = 1;
  p_in.b_const.val =
      (fmt == CVK_FMT_BF16) ? ctx.convert_fp32_to_bf16(-1.0f) : (-1);
  p_in.b_const.is_signed = 1;
  p_in.rshift_bits = 0;
  p_in.relu_enable = 0;
  p_in.layer_id = layer_id;
  ctx.tiu_mul(&p_in);

  cvk_tiu_max_pooling_param_t param = {0};
  param.ofmap = &tl_ofmap;
  param.ifmap = &tl_ifmap;
  param.kh = kh;
  param.kw = kw;
  param.pad_top = 0;
  param.pad_bottom = 0;
  param.pad_left = 0;
  param.pad_right = 0;
  param.stride_h = 1;
  param.stride_w = 1;
  param.ins_val = -128;
  param.ins_fp = 0xff7f;
  param.layer_id = layer_id;
  ctx.tiu_max_pooling(&param);

  if (fmt == CVK_FMT_BF16) {
    cvk_tiu_mul_param_t p_out = {0};
    p_out.res_high = NULL;
    p_out.res_low = &tl_ofmap;
    p_out.a = &tl_ofmap;
    p_out.b_is_const = 1;
    p_out.b_const.val =
        (fmt == CVK_FMT_BF16) ? ctx.convert_fp32_to_bf16(-1.0f) : (-1);
    p_out.b_const.is_signed = 1;
    p_out.rshift_bits = 0;
    p_out.relu_enable = 0;
    p_out.layer_id = layer_id;
    ctx.tiu_mul(&p_out);
  } else {
    cvk_tiu_mul_param_t p_out = {0};
    p_out.res_high = NULL;
    p_out.res_low = &tl_ofmap;
    p_out.a = &tl_ofmap;
    p_out.b_is_const = 1;
    p_out.b_const.val = multiplier * (-1);
    p_out.b_const.is_signed = 1;
    p_out.rshift_bits = rshift;
    p_out.relu_enable = 0;
    p_out.layer_id = layer_id;
    ctx.tiu_mul(&p_out);
  }
}

void TgReduceKernel::reduce_mean() {
  if (fmt == CVK_FMT_I8) {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &tl_ofmap;
    param.ifmap = &tl_ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.avg_pooling_const = multiplier;
    param.rshift_bits = rshift;
    param.layer_id = layer_id;
    param.ins_val = param.avg_pooling_const;
    param.ins_fp = 0;
    ctx.tiu_average_pooling(&param);
  } else {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &tl_ofmap;
    param.ifmap = &tl_ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.avg_pooling_const = ctx.convert_fp32_to_bf16(1.0);
    param.layer_id = layer_id;
    param.ins_val = 0;
    param.ins_fp = param.avg_pooling_const;
    ctx.tiu_average_pooling(&param);
  }
}

void TgReduceKernel::reduce_sum() {
  if (fmt == CVK_FMT_BF16) {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &tl_ofmap;
    param.ifmap = &tl_ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.avg_pooling_const = ctx.convert_fp32_to_bf16(kh * kw);
    param.layer_id = layer_id;
    param.ins_val = 0;
    param.ins_fp = param.avg_pooling_const;
    ctx.tiu_average_pooling(&param);
  } else {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &tl_ofmap;
    param.ifmap = &tl_ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.avg_pooling_const = multiplier;
    param.rshift_bits = rshift;
    param.ins_val = param.avg_pooling_const;
    param.ins_fp = 0;
    ctx.tiu_average_pooling(&param);
  }
}

void TgReduceKernel::reduce_l2() {
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_ifmap;
  p.a = &tl_ifmap;
  p.b_is_const = 0;
  p.b = &tl_ifmap;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  cvk_tl_t tl_buf = tl_ofmap;
  tl_buf.start_address = tl_sum->start_address;
  cvk_tiu_average_pooling_param_t p1 = {0};
  p1.ofmap = &tl_buf;
  p1.ifmap = &tl_ifmap;
  p1.kh = kh;
  p1.kw = kw;
  p1.pad_top = 0;
  p1.pad_bottom = 0;
  p1.pad_left = 0;
  p1.pad_right = 0;
  p1.stride_h = 1;
  p1.stride_w = 1;
  p1.avg_pooling_const = ctx.convert_fp32_to_bf16(kh * kw);
  p1.layer_id = layer_id;
  p1.ins_val = 0;
  p1.ins_fp = p1.avg_pooling_const;
  ctx.tiu_average_pooling(&p1);

  auto &s = tl_ofmap.shape;
  cvi_backend_bf16_tl_lut_mantissa_method(
      ctx, layer_id, tl_sum->start_address, tl_ofmap.start_address,
      tl_ifmap.start_address, tl_lut->start_address,
      tl_lut_mantissa->start_address, s.n, s.c, s.h, s.w);
}

void TgReduceKernel::init(uint32_t layer_id, gaddr_t ga_input,
                          gaddr_t ga_output, std::vector<int64_t> shape,
                          std::vector<int32_t> axes, int multiplier, int rshift,
                          reduce_type_t type, cvk_fmt_t fmt, gaddr_t ga_table,
                          gaddr_t ga_mantissa_table) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_table = ga_table;
  this->ga_mantissa_table = ga_mantissa_table;
  this->multiplier = multiplier;
  this->rshift = rshift;
  this->type = type;
  this->fmt = fmt;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  reshape(shape, axes);
  ctx.set_layer_id(layer_id);
}

void TgReduceKernel::selectTilePolicy() {
  uint32_t lmem_used = 0;
  int out_blob = 2;
  if (type == REDUCE_L2) {
    auto lut_shape = ctx.lut_table_shape(fmt);
    lmem_used += 2 * ctx.lmem_tensor_to_size(lut_shape, fmt, 1);
    out_blob = 3; // + reduce_l2 sum
  }
  if (end_reduce) {
    // only tile c
    int tile_c = std::min(c, MAX_CHANNEL);
    while (tile_c > 0) {
      auto in_shape = ctx.tl_shape_t4(n, tile_c, h, w);
      auto out_shape = ctx.tl_shape_t4(n, tile_c, 1, 1);
      uint32_t lmem_need =
          lmem_used + 2 * ctx.lmem_tensor_to_size(in_shape, fmt, 1) +
          out_blob * ctx.lmem_tensor_to_size(out_shape, fmt, 1);
      if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
        break;
      }
      if (tile_c % NPU_NUM == 0) {
        tile_c -= NPU_NUM;
      } else {
        tile_c -= (tile_c % NPU_NUM);
      }
    }
    if (tile_c == 0) {
      llvm_unreachable("Reduce Op tiling failed\n");
    }
    CviBackendContext::tiling_info_t info = {0};
    info.n = n;
    info.h = h;
    info.w = w;
    for (int pos_c = 0; pos_c < c; pos_c += tile_c) {
      info.pos_c = pos_c;
      info.c = std::min(tile_c, c - pos_c);
      info.offset = pos_c * h * w * fmt_size;
      tiles.emplace_back(info);
    }
    return;
  } else {
    // tile c and tile w
    int max_c = std::min(c, MAX_CHANNEL);
    int max_w = std::min(w, MAX_WIDTH);
    int tile_c, tile_w;
    for (tile_w = max_w; tile_w > 0; tile_w--) {
      for (tile_c = max_c; tile_c > 0;) {
        auto in_shape = ctx.tl_shape_t4(n, tile_c, h, tile_w);
        auto out_shape = ctx.tl_shape_t4(n, tile_c, 1, tile_w);
        uint32_t lmem_need =
            lmem_used + 2 * ctx.lmem_tensor_to_size(in_shape, fmt, 1) +
            out_blob * ctx.lmem_tensor_to_size(out_shape, fmt, 1);
        if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
          goto tiling_success;
        }
        if (tile_c % NPU_NUM == 0) {
          tile_c -= NPU_NUM;
        } else {
          tile_c -= (tile_c % NPU_NUM);
        }
      }
    }
    llvm_unreachable("Reduce Op tiling failed\n");
  tiling_success:
    CviBackendContext::tiling_info_t info = {0};
    info.n = n;
    info.h = h;
    for (int pos_c = 0; pos_c < c; pos_c += tile_c) {
      for (int pos_w = 0; pos_w < w; pos_w += tile_w) {
        info.pos_w = pos_w;
        info.pos_c = pos_c;
        info.c = std::min(tile_c, c - pos_c);
        info.w = std::min(tile_w, w - pos_w);
        info.offset = (pos_c * h * w + pos_w) * fmt_size;
        tiles.emplace_back(info);
      }
    }
  }
}

void TgReduceKernel::allocLmem() {
  auto &tile = tiles[0];
  auto in_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto out_shape = ctx.tl_shape_t4(tile.n, tile.c, 1, 1);
  if (end_reduce == false) {
    out_shape.w = (uint32_t)tile.w;
  }
  if (type == REDUCE_L2) {
    auto lut_shape = ctx.lut_table_shape(fmt);
    tl_lut = ctx.lmem_alloc_tensor(lut_shape, fmt, 1);
    tl_lut_mantissa = ctx.lmem_alloc_tensor(lut_shape, fmt, 1);
    ctx.tdma_load(tl_lut, ga_table);
    ctx.tdma_load(tl_lut_mantissa, ga_mantissa_table);
    tl_sum = ctx.lmem_alloc_tensor(out_shape, fmt, 1);
  }

  tl_mem[0] = ctx.lmem_alloc_tensor(in_shape, fmt, 1);
  tl_mem[1] = ctx.lmem_alloc_tensor(in_shape, fmt, 1);
  tl_mem[2] = ctx.lmem_alloc_tensor(out_shape, fmt, 1);
  tl_mem[3] = ctx.lmem_alloc_tensor(out_shape, fmt, 1);
}

void TgReduceKernel::deallocLmem() {
  for (int i = 3; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
  if (type == REDUCE_L2) {
    ctx.lmem_free_tensor(tl_sum);
    ctx.lmem_free_tensor(tl_lut_mantissa);
    ctx.lmem_free_tensor(tl_lut);
  }
}

void TgReduceKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  auto in_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto out_shape = ctx.tl_shape_t4(tile.n, tile.c, 1, 1);
  if (end_reduce == false) {
    out_shape.w = (uint32_t)tile.w;
  }
  ctx.lmem_init_tensor(&tl_ifmap, in_shape, fmt, 1);
  ctx.lmem_init_tensor(&tl_ofmap, out_shape, fmt, 1);
  tl_ifmap.start_address = tl_mem[step_idx % 2]->start_address;
  tl_ofmap.start_address = tl_mem[step_idx % 2 + 2]->start_address;
}

void TgReduceKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_load_stride(&tl_ifmap, ga_input + tile.offset, in_gstride);
}

void TgReduceKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  int offset;
  if (end_reduce) {
    offset = tile.pos_c * out_gstride.c;
  } else {
    offset = tile.pos_c * out_gstride.c + tile.pos_w * out_gstride.w;
  }
  ctx.tdma_store_stride(&tl_ofmap, ga_output + offset, out_gstride);
}

void TgReduceKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  switch (type) {
  case REDUCE_MAX:
    reduce_max();
    break;
  case REDUCE_MIN:
    reduce_min();
    break;
  case REDUCE_SUM:
    reduce_sum();
    break;
  case REDUCE_MEAN:
    reduce_mean();
    break;
  case REDUCE_L2:
    reduce_l2();
    break;
  default:
    assert(0);
  }
}

void TgReduceKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
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

void cvi_backend_tg_bf16_reduce_max_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id, gaddr_t ga_input,
                                           gaddr_t ga_output,
                                           std::vector<int64_t> shape,
                                           std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_MAX,
              CVK_FMT_BF16);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_reduce_min_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id, gaddr_t ga_input,
                                           gaddr_t ga_output,
                                           std::vector<int64_t> shape,
                                           std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_MIN,
              CVK_FMT_BF16);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_reduce_mean_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_input,
                                            gaddr_t ga_output,
                                            std::vector<int64_t> shape,
                                            std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_MEAN,
              CVK_FMT_BF16);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_reduce_sum_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id, gaddr_t ga_input,
                                           gaddr_t ga_output,
                                           std::vector<int64_t> shape,
                                           std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_SUM,
              CVK_FMT_BF16);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_reduce_max_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_input,
                                            gaddr_t ga_output,
                                            std::vector<int64_t> shape,
                                            std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_MAX,
              CVK_FMT_I8);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_reduce_min_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_input,
                                            gaddr_t ga_output,
                                            std::vector<int64_t> shape,
                                            std::vector<int32_t> axes,
                                            int multiplier, int rshift) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, multiplier, rshift,
              REDUCE_MIN, CVK_FMT_I8);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_reduce_mean_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, std::vector<int64_t> shape, std::vector<int32_t> axes,
    int multiplier, int rshift) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, multiplier, rshift,
              REDUCE_MEAN, CVK_FMT_I8);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_reduce_sum_kernel(const CviBackendContext &ctx,
                                            uint32_t layer_id, gaddr_t ga_input,
                                            gaddr_t ga_output,
                                            std::vector<int64_t> shape,
                                            std::vector<int32_t> axes,
                                            int multiplier, int rshift) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, multiplier, rshift,
              REDUCE_SUM, CVK_FMT_I8);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_reduce_l2_kernel(const CviBackendContext &ctx,
                                          uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output, gaddr_t ga_table,
                                          gaddr_t ga_mantissa_table,
                                          std::vector<int64_t> shape,
                                          std::vector<int32_t> axes) {
  TgReduceKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, shape, axes, 0, 0, REDUCE_L2,
              CVK_FMT_BF16, ga_table, ga_mantissa_table);
  kernel.selectTilePolicy();
  kernel.schedule();
}
