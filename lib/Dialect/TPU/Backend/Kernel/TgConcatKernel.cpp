/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgConcatKernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"
#include "TgConcatKernel.hpp"

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
}

void TgConcatKernel::init(uint32_t layer_id, int input_num, int dim_size,
                          int concat_axis, gaddr_t input_gaddrs[],
                          gaddr_t output_gaddr, int axis_dims[],
                          int output_dim[], bool do_relu,
                          const int8_t right_shift_width[],
                          const int threshold_x_quantized[], cvk_fmt_t fmt) {
  ctx.assert_support_fmt(fmt);
  assert(dim_size >= 2 && dim_size <= 4);
  assert(concat_axis < dim_size);
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->do_relu = do_relu;
  update_output(output_dim, dim_size, concat_axis);
  uint64_t axis_addr_offset = 0;

  for (int i = 0; i < input_num; i++) {
    input_info_t info;
    memset((void *)&info, 0, sizeof(input_info_t));
    if (right_shift_width != nullptr) {
      info.rshift_width = right_shift_width[i];
    }
    if (threshold_x_quantized != nullptr) {
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
    info.ga_input = input_gaddrs[i];
    info.ga_output = output_gaddr + axis_addr_offset;
    axis_addr_offset += ctx.bytesize_of_fmt(fmt) * axis_size(info.shape);
    inputs.emplace_back(info);
  }
}

void TgConcatKernel::doTileForNormalCase() {
  for (auto &input : inputs) {
    int n = input.shape.n, c = input.shape.c, h = input.shape.h,
        w = input.shape.w;
    int step_w, step_h, step_c, step_n;
    int max_w = std::min(w, MAX_WIDTH);
    int max_h = std::min(h, MAX_HEIGHT);
    int max_c = std::min(c, MAX_CHANNEL);
    int max_n = std::min(n, MAX_CHANNEL);

    uint32_t lmem_required = (uint32_t)LOCAL_MEM_SIZE + 1;
    for (step_w = max_w; step_w > 0; --step_w) {
      for (step_h = max_h; step_h > 0; --step_h) {
        for (step_n = max_n; step_n > 0; --step_n) {
          for (step_c = max_c; step_c > 0; step_c -= NPU_NUM) {
            auto max_shape = ctx.tl_shape_t4(step_n, step_c, step_h, step_w);
            lmem_required = ctx.lmem_tensor_to_size(max_shape, fmt, 1);
            if (lmem_required <= (uint32_t)LOCAL_MEM_SIZE) {
              goto after_loop;
            }
          }
        }
      }
    }
  after_loop:
    if (lmem_required > (uint32_t)LOCAL_MEM_SIZE) {
      llvm::errs() << llvm::format("Tilling failed, src shape:(%d,%d,%d,%d)\n",
                                   n, c, h, w);
      assert(0);
    }

    tile_info_t tile;
    cvk_tg_stride_t input_stride = ctx.tg_default_stride(input.shape, fmt);
    for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
      tile.n = std::min(n - tile.pos_n, step_n);
      for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
        tile.c = std::min(c - tile.pos_c, step_c);
        for (tile.pos_h = 0; tile.pos_h < h; tile.pos_h += step_h) {
          tile.h = std::min(h - tile.pos_h, step_h);
          for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
            tile.w = std::min(w - tile.pos_w, step_w);
            tile.src_offset =
                tile.pos_w * input_stride.w + tile.pos_h * input_stride.h +
                tile.pos_c * input_stride.c + tile.pos_n * input_stride.n;
            tile.dst_offset =
                tile.pos_w * output_stride.w + tile.pos_h * output_stride.h +
                tile.pos_c * output_stride.c + tile.pos_n * output_stride.n;
            input.tiles.push_back(tile);
          }
        }
      }
    }
  }
}

void TgConcatKernel::selectTilePolicy() { doTileForNormalCase(); }

void TgConcatKernel::schedule() {
  for (auto &input : inputs) {
    cvk_tl_shape_t max_shape = ctx.tl_shape_t4(
        input.tiles[0].n, input.tiles[0].c, input.tiles[0].h, input.tiles[0].w);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(max_shape, fmt, 1);

    for (auto &tile : input.tiles) {
      cvk_tl_t tl_ifmap = {0};
      // load
      tl_ifmap.start_address = tl_input->start_address;
      tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
      tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
      tl_ifmap.fmt = fmt;
      ctx.tdma_load_stride(&tl_ifmap, input.ga_input + tile.src_offset,
                           input.stride);
      // do quantize
      if (input.do_quantize) {
        cvk_tiu_mul_param_t p = {0};
        p.res_high = nullptr;
        p.res_low = &tl_ifmap;
        p.a = &tl_ifmap;
        if (fmt == CVK_FMT_BF16) {
          // bf16 no quant now
          p.b_const.val = ctx.convert_fp32_to_bf16(1.0);
          p.rshift_bits = 0;
        } else {
          p.b_const.val = static_cast<int16_t>(input.data_quantized);
          p.rshift_bits = static_cast<uint8_t>(input.rshift_width);
        }
        p.b_const.is_signed = false;
        p.b_is_const = 1;
        p.layer_id = layer_id;
        p.relu_enable = do_relu ? 1 : 0;
        ctx.tiu_mul(&p);
      }
      // store
      ctx.tdma_store_stride(&tl_ifmap, input.ga_output + tile.dst_offset,
                            output_stride);
    }
    ctx.lmem_free_tensor(tl_input);
  }
}

void cvi_backend_tg_concat_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    int input_num, gaddr_t input_gaddrs[], gaddr_t output_gaddr,
    int axis_dims[], int concat_axis, int output_dim_size, int output_dim[],
    bool do_relu, const int8_t *right_shift_width,
    const int *threshold_x_quantized, cvk_fmt_t fmt) {
  TgConcatKernel kernel(ctx);
  kernel.init(layer_id, input_num, output_dim_size, concat_axis, input_gaddrs,
              output_gaddr, axis_dims, output_dim, do_relu, right_shift_width,
              threshold_x_quantized, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
