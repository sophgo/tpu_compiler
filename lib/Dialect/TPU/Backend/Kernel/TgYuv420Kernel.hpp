/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_YUV420_KERNEL_HPP
#define TG_YUV420_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgYuv420Kernel {
public:
  TgYuv420Kernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, const std::vector<int> &order, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void load_u8_to_bf16(cvk_tl_t *dst, uint64_t src_gaddr,
                       cvk_tg_stride_t stride);
  void store_bf16_to_u8(cvk_tl_t *src, uint64_t dst_gaddr,
                        cvk_tg_stride_t stride);

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_output;
  int32_t n, c, h, w;
  cvk_fmt_t fmt;
  int fmt_size;
  static const int BLOB_NUM = 12; // yuvrgb * 2 for flip
  // current step yuv,rgb
  cvk_tl_t tl_y, tl_u, tl_v, tl_r, tl_g, tl_b;
  // lmem alloc
  cvk_tl_t *tl_mem[BLOB_NUM];
  cvk_tg_stride_t y_gstride, uv_gstride, rgb_gstride;
  std::vector<int> order;
  int32_t layer_id;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
