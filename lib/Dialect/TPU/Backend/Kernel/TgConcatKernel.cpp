/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgConcatKernel.cpp
 * Description:
 */

#include "TgConcatKernel.hpp"
#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_concat_kernel"

uint32_t &TgConcatKernel::axis_dim(cvk_tg_shape_t &shape) {
  switch (axis) {
  case 0:
    return shape.n;
  case 1:
    return shape.c;
  case 2:
    return shape.h;
  case 3:
    return shape.w;
  default:
    assert(0);
  }
}

uint64_t TgConcatKernel::axis_size(const cvk_tg_shape_t &shape) const {
  uint64_t size = 1;
  switch (axis) {
  case 0:
    size = shape.n * shape.c * shape.h * shape.w;
    break;
  case 1:
    size = shape.c * shape.h * shape.w;
    break;
  case 2:
    size = shape.h * shape.w;
    break;
  case 3:
    size = shape.w;
    break;
  default:
    assert(0 && "axis should less than 4");
    break;
  }
  return size;
}

void TgConcatKernel::update_output(int output_dim[], int dim_size,
                                   int concat_axis) {
  axis = concat_axis;
  switch (dim_size) {
  case 4:
    output_shape = ctx.tg_shape_t4(output_dim[0], output_dim[1], output_dim[2],
                                   output_dim[3]);

    break;
  case 3: // w = 1
    output_shape =
        ctx.tg_shape_t4(output_dim[0], output_dim[1], output_dim[2], 1);
    break;
  case 2: // n = 1, w = 1
    output_shape = ctx.tg_shape_t4(1, output_dim[0], output_dim[1], 1);
    axis = concat_axis + 1;
    break;
  case 1: // n = 1, c = 1, w = 1
    output_shape = ctx.tg_shape_t4(1, 1, output_dim[0], 1);
    axis = 2;
    break;
  default: // not support
    assert(0 && "dim size large than 4, not supported");
    break;
  }
  output_stride = ctx.tg_default_stride(output_shape, fmt);
  if (axis == 3 && output_shape.h != 1 &&
      output_stride.w >= (uint32_t)0x10000) {
    // hstride should less than 0x10000, c = c*h, h = 1
    output_shape.c *= output_shape.h;
    output_shape.h = output_shape.w;
    output_shape.w = 1;
    output_stride = ctx.tg_default_stride(output_shape, fmt);
    axis = 2;
  }
  // TilingAll is better than TilingNCHW
  for (int i = 0; i < concat_axis; i++) {
    if (output_dim[i] != 1) {
      return;
    }
  }
  tiling_mode = CviBackendContext::TilingAll;
}

void TgConcatKernel::init(uint32_t layer_id, int input_num, int dim_size,
                          int concat_axis, gaddr_t input_gaddrs[],
                          gaddr_t output_gaddr, int axis_dims[],
                          int output_dim[], bool do_relu,
                          const int right_shift_width[],
                          const int threshold_x_quantized[], cvk_fmt_t fmt) {
  ctx.assert_support_fmt(fmt);
  assert(dim_size >= 2 && dim_size <= 4);
  assert(concat_axis < dim_size);
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->do_relu = do_relu;
  this->input_num = input_num;
  this->tiling_mode = CviBackendContext::TilingNCHW;
  update_output(output_dim, dim_size, concat_axis);
  uint64_t axis_addr_offset = 0;
  do_parallel = false;
  for (int i = 0; i < input_num; i++) {
    input_info_t info;
    memset((void *)&info, 0, sizeof(input_info_t));
    if (right_shift_width != nullptr) {
      info.rshift_width = right_shift_width[i];
    }
    if (threshold_x_quantized != nullptr && threshold_x_quantized[i] != 0) {
      info.data_quantized = threshold_x_quantized[i];
    } else {
      info.data_quantized = 1;
    }
    info.shape = output_shape;
    axis_dim(info.shape) = axis_dims[i];
    info.stride = ctx.tg_default_stride(info.shape, fmt);
    info.do_quantize = true;
    if (info.rshift_width == 0 && info.data_quantized == 1 &&
        do_relu == false) {
      info.do_quantize = false;
    }
    if (info.do_quantize && false == do_parallel) {
      do_parallel = true;
    }
    info.ga_input = input_gaddrs[i];
    info.ga_output = output_gaddr + axis_addr_offset;
    axis_addr_offset += ctx.bytesize_of_fmt(fmt) * axis_size(info.shape);
    inputs.emplace_back(info);
  }
}

uint64_t
TgConcatKernel::dst_offset(const CviBackendContext::tiling_info_t &tile) const {
  return tile.pos_w * output_stride.w + tile.pos_h * output_stride.h +
         tile.pos_c * output_stride.c + tile.pos_n * output_stride.n;
}

void TgConcatKernel::selectTilePolicy() {
  total_tiles = 0;
  if (do_parallel) {
    for (auto &input : inputs) {
      input.tile_idx = total_tiles;
      ctx.tiling_packing(input.tiles, input.shape, fmt, 4, 0, tiling_mode,
                         true);
      total_tiles += input.tiles.size();
    }
    // half the lmem
    int lsize = ALIGN(LOCAL_MEM_SIZE / 4, EU_NUM);
    base_addr[0] = 0;
    base_addr[1] = lsize;
    base_addr[2] = base_addr[1] + lsize;
    base_addr[3] = base_addr[2] + lsize;
  } else {
    for (auto &input : inputs) {
      input.tile_idx = total_tiles;
      ctx.tiling_packing(input.tiles, input.shape, fmt, 1, 0, tiling_mode);
      total_tiles += input.tiles.size();
    }
    memset(base_addr, 0, sizeof(base_addr));
  }
}

void TgConcatKernel::prepare(int32_t step_idx,
                             TgConcatKernel::input_info_t *&input,
                             CviBackendContext::tiling_info_t *&tile) {
  for (int idx = input_num - 1; idx >= 0; idx--) {
    input = &inputs[idx];
    if (input->tile_idx <= step_idx) {
      tile = &input->tiles[step_idx - input->tile_idx];
      ctx.lmem_init_tensor(&tl_input,
                           ctx.tl_shape_t4(tile->n, tile->c, tile->h, tile->w),
                           fmt, 1);
      ctx.lmem_init_tensor(&tl_output,
                           ctx.tl_shape_t4(tile->n, tile->c, tile->h, tile->w),
                           fmt, 1);
      tl_input.start_address = base_addr[step_idx % 2];
      tl_output.start_address = base_addr[step_idx % 2 + 2];
      return;
    }
  }
  assert(0 && "tile incorrect");
}

void TgConcatKernel::load(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CviBackendContext::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  if (tiling_mode == CviBackendContext::TilingNCHW) {
    ctx.tdma_load_stride(&tl_input, input->ga_input + tile->offset,
                         input->stride);
  } else {
    ctx.tdma_load(&tl_input, input->ga_input + tile->offset);
  }
}

void TgConcatKernel::store(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CviBackendContext::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  if (tiling_mode == CviBackendContext::TilingNCHW) {
    ctx.tdma_store_stride(&tl_output, input->ga_output + dst_offset(*tile),
                          output_stride);
  } else {
    ctx.tdma_store(&tl_output, input->ga_output + tile->offset);
  }
}

void TgConcatKernel::compute(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CviBackendContext::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  // do quantize
  if (input->do_quantize) {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_output;
    p.a = &tl_input;
    if (fmt == CVK_FMT_BF16) {
      // bf16 no quant now
      p.b_const.val = ctx.convert_fp32_to_bf16(1.0);
      p.rshift_bits = 0;
    } else {
      p.b_const.val = static_cast<int16_t>(input->data_quantized);
      p.rshift_bits = static_cast<uint8_t>(input->rshift_width);
    }
    p.b_const.is_signed = false;
    p.b_is_const = 1;
    p.layer_id = layer_id;
    p.relu_enable = do_relu ? 1 : 0;
    ctx.tiu_mul(&p);
  } else {
    cvk_tiu_copy_param_t p;
    p.layer_id = layer_id;
    p.src = &tl_input;
    p.dst = &tl_output;
    ctx.tiu_copy(&p);
  }
}

void TgConcatKernel::schedule() {
  if (do_parallel) {
    for (int step = 0; step < total_tiles + 2; step++) {
      ctx.parallel_enable();
      if (step > 0 && step - 1 < total_tiles) {
        compute(step - 1);
      }
      if (step < total_tiles) {
        load(step);
      }
      if (step > 1) {
        store(step - 2);
      }
      ctx.parallel_disable();
    }
  } else {
    for (int step = 0; step < total_tiles; step++) {
      load(step);
      store(step);
    }
  }
}

void cvi_backend_tg_concat_kernel(const CviBackendContext &ctx,
                                  uint32_t layer_id, int input_num,
                                  gaddr_t input_gaddrs[], gaddr_t output_gaddr,
                                  int axis_dims[], int concat_axis,
                                  int output_dim_size, int output_dim[],
                                  bool do_relu, const int *right_shift_width,
                                  const int *threshold_x_quantized,
                                  cvk_fmt_t fmt) {
  TgConcatKernel kernel(ctx);
  kernel.init(layer_id, input_num, output_dim_size, concat_axis, input_gaddrs,
              output_gaddr, axis_dims, output_dim, do_relu, right_shift_width,
              threshold_x_quantized, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
