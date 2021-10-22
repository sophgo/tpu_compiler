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

//enum YuvType { YUV_UNKNOWN = 0, YUV420_PLANAR = 1, YUV_NV12 = 2, YUV_NV21 = 3 };

class TgYuv420Kernel {
public:
  TgYuv420Kernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, const std::vector<int> &order,
            int32_t pixel_type, int32_t channel_align, int32_t y_align,
            int32_t w_align, cvk_fmt_t fmt);

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
  gaddr_t ga_y, ga_u, ga_v;
  int32_t n, c, h, w;
  cvk_fmt_t fmt;
  int n_stride;
  int32_t y_w_aligned;
  int32_t uv_w_aligned;
  cvk_tl_shape_t kernel_shape;
  cvk_tl_t *tl_mem_kernel;
  uint32_t BLOB_NUM; // yuvrgb * 2 for flip, + uv2 * 2
  // current step yuv,rgb
  cvk_tl_t tl_y, tl_u, tl_v, tl_r, tl_g, tl_b, tl_4u, tl_4v, tl_uv, tl_uv_cache;
  cvk_tl_t tl_kernel;
  // lmem alloc
  std::vector<cvk_tl_t *> tl_mem;
  cvk_tg_stride_t y_gstride, uv_gstride, rgb_gstride;
  std::vector<int> order;
  int32_t layer_id;
  int32_t step_n, step_c, step_h, step_w; // for tiling step
  int32_t yuv_type;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
