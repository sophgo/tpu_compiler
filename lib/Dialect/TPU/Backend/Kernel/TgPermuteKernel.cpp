/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgPermuteKernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"
#include "TgPermuteKernel.hpp"

#define DEBUG_TYPE "cvi_backend_permute_kernel"

//
// permute xxx3:
//   TDMA does not has the stride of width(ws).
//   Since the width of destination is unchanged, use tensor store to write one
//   hight to the correct position with ns, cs, hs.
//   It is tricky that destination shape in tensor store is the same as source
//   shape.
//
// Permute 0321:
//   (N, C, H, W) -> (N, H, W, C)
//   tensor load
//   tensor store, cw transpose
//   e.g. reduceOp with axis [1] needs CW transpose in global memory
//
// Permute 0213:
//   (N,C,1,H*W), permute(0321) ->(N,H*W,1,C)
//   e.g. ssd300
//
// Permute 0312:
//   (N,C*H,1,W), permute(0321) ->(N,W,1,C*H)
//

//
// order supported now
//
void TgPermuteKernel::assert_order_support() const {
  if (order[3] == 3) { // xxx3
  } else if (is_order(0, 3, 2, 1)) {
  } else if (is_order(0, 2, 3, 1)) {
  } else if (is_order(0, 3, 1, 2)) {
  } else {
    llvm::errs() << "Not support order (" << order[0] << ", " << order[1]
                 << ", " << order[2] << ", " << order[3] << ") permute case\n";
    assert(0);
  }
}

bool TgPermuteKernel::is_order(int32_t order_n, int32_t order_c,
                               int32_t order_h, int32_t order_w) const {
  return (order_n == order[0] && order_c == order[1] && order_h == order[2] &&
          order_w == order[3]);
}

void TgPermuteKernel::update_order(int32_t order_n, int32_t order_c,
                                   int32_t order_h, int32_t order_w) {
  order[0] = order_n;
  order[1] = order_c;
  order[2] = order_h;
  order[3] = order_w;
}

void TgPermuteKernel::init(uint32_t layer_id, gaddr_t ga_input,
                           gaddr_t ga_output, int32_t n, int32_t c, int32_t h,
                           int32_t w, int32_t order_n, int32_t order_c,
                           int32_t order_h, int32_t order_w, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  update_order(order_n, order_c, order_h, order_w);
  assert_order_support();

  if (is_order(0, 2, 3, 1)) {
    this->w = h * w;
    this->h = 1;
    update_order(0, 3, 2, 1);
  } else if (is_order(0, 3, 1, 2)) {
    this->c = c * h;
    this->h = 1;
    update_order(0, 3, 2, 1);
  }

  int32_t i_s[4] = {this->n, this->c, this->h, this->w};
  int32_t o_s[4] = {i_s[order[0]], i_s[order[1]], i_s[order[2]], i_s[order[3]]};
  src_stride = ctx.tg_default_stride(this->c, this->h, this->w, fmt);
  dst_stride = ctx.tg_default_stride(o_s[1], o_s[2], o_s[3], fmt);
  uint32_t o_stride[4];
  o_stride[order[0]] = dst_stride.n;
  o_stride[order[1]] = dst_stride.c;
  o_stride[order[2]] = dst_stride.h;
  o_stride[order[3]] = dst_stride.w;
  dst_stride_order = {o_stride[0], o_stride[1], o_stride[2], o_stride[3]};
  fmt_size = ctx.bytesize_of_fmt(fmt);
  ctx.set_layer_id(layer_id);
}

void TgPermuteKernel::doTileForNormalCase() {
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
          max_shape = ctx.tl_shape_t4(step_n, step_c, step_h, step_w);
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
    llvm::errs() << llvm::format(
        "Tilling failed, src shape:(%d,%d,%d,%d), order:(%d,%d,%d,%d)\n", n, c,
        h, w, order[0], order[1], order[2], order[3]);
    assert(0);
  }
  PermuteTile tile;
  for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
    tile.n = std::min(n - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
      tile.c = std::min(c - tile.pos_c, step_c);
      for (tile.pos_h = 0; tile.pos_h < h; tile.pos_h += step_h) {
        tile.h = std::min(h - tile.pos_h, step_h);
        for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
          tile.w = std::min(w - tile.pos_w, step_w);
          tile.src_offset =
              tile.pos_w * src_stride.w + tile.pos_h * src_stride.h +
              tile.pos_c * src_stride.c + tile.pos_n * src_stride.n;
          tile.dst_offset = tile.pos_w * dst_stride_order.w +
                            tile.pos_h * dst_stride_order.h +
                            tile.pos_c * dst_stride_order.c +
                            tile.pos_n * dst_stride_order.n;
          tiles.push_back(tile);
        }
      }
    }
  }
}

void TgPermuteKernel::selectTilePolicy() { doTileForNormalCase(); }

void TgPermuteKernel::load(int32_t step_idx, cvk_tl_t &tl_ifmap) const {
  const PermuteTile &tile = tiles[step_idx];
  tl_ifmap.start_address = tl_input->start_address;
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_ifmap.fmt = fmt;
  ctx.tdma_load_stride(&tl_ifmap, ga_input + tile.src_offset, src_stride);
}

void TgPermuteKernel::store_xxx3(int32_t step_idx, cvk_tl_t &tl_ifmap) const {
  const PermuteTile &tile = tiles[step_idx];
  ctx.tdma_store_stride(&tl_ifmap, ga_output + tile.dst_offset,
                        dst_stride_order);
}

void TgPermuteKernel::store_0321(int32_t step_idx, cvk_tl_t &tl_ifmap) const {
  const PermuteTile &tile = tiles[step_idx];
  cvk_tg_t ofmap = {0};
  ofmap.start_address = ga_output + tile.dst_offset;
  ofmap.shape = ctx.tg_shape_t4(tile.n, tile.w, tile.h, tile.c);
  ofmap.stride = dst_stride;
  ofmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_output);
  ofmap.fmt = fmt;
  cvk_tdma_l2g_tensor_copy_cw_transposed_param_t param;
  param.src = &tl_ifmap;
  param.dst = &ofmap;
  ctx.tdma_l2g_tensor_copy_cw_transposed(&param);
}

void TgPermuteKernel::schedule() {
  cvk_tl_shape_t max_shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  tl_input = ctx.lmem_alloc_tensor(max_shape, fmt, 1);
  cvk_tl_t tl_ifmap;
  for (uint32_t idx = 0; idx < tiles.size(); idx++) {
    load(idx, tl_ifmap);
    if (order[3] == 3) {
      store_xxx3(idx, tl_ifmap);
    } else if (is_order(0, 3, 2, 1)) {
      store_0321(idx, tl_ifmap);
    } else {
      assert(0);
    }
  }
  ctx.lmem_free_tensor(tl_input);
}

void cvi_backend_tg_permute_kernel(const CviBackendContext &ctx,
                                   uint32_t stream_id, uint32_t inst_id,
                                   uint32_t layer_id, const uint32_t *depends,
                                   uint32_t depends_len, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int order_n, int order_c, int order_h,
                                   int order_w, cvk_fmt_t fmt) {
  TgPermuteKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, order_n, order_c,
              order_h, order_w, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
