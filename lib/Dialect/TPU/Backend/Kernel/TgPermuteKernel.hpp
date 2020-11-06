/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-22
 */
#ifndef TG_PERMUTE_KERNEL_HPP
#define TG_PERMUTE_KERNEL_HPP

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

class TgPermuteKernel {
public:
  TgPermuteKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int32_t n,
            int32_t c, int32_t h, int32_t w, int32_t order_n, int32_t order_c,
            int32_t order_h, int32_t order_w, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  bool is_order(int32_t order_n, int32_t order_c, int32_t order_h,
                int32_t order_w) const;
  void update_order(int32_t order_n, int32_t order_c, int32_t order_h,
                    int32_t order_w);
  void update_NCHW(int32_t n, int32_t c, int32_t h, int32_t w);
  void convert_order();
  bool check_stride_h();
  void load(int32_t step_idx, cvk_tl_t &tl_ifmap) const;
  void store(int32_t step_idx, cvk_tl_t &tl_ifmap) const;
  void store_normal(int32_t step_idx, cvk_tl_t &tl_ifmap) const;
  void store_0321(int32_t step_idx, cvk_tl_t &tl_ifmap) const;
  uint64_t dst_offset(const CviBackendContext::tiling_info_t &tile) const;
  const CviBackendContext &ctx;

  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_tl_t *tl_input;

  int32_t n, c, h, w; // input_shape
  int32_t n_loop;
  int32_t n_index;
  uint64_t n_offset;
  cvk_tl_shape_t max_shape;
  cvk_tg_stride_t src_stride;
  cvk_tg_stride_t dst_stride;
  cvk_tg_stride_t dst_stride_order;
  int32_t order[4];
  cvk_fmt_t fmt;
  int32_t layer_id;
  typedef enum {
    STORE_NORMAL,
    STORE_321,
  } store_mode_t;
  store_mode_t store_mode;

  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
