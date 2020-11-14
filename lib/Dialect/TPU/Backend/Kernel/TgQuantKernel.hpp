/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_QUANT_KERNEL_HPP
#define TG_QUANT_KERNEL_HPP

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

class TgQuantKernel {
public:
  TgQuantKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, cvk_fmt_t from, cvk_fmt_t to, gaddr_t ga_input,
            gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w,
            float const_scale);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void allocLmem();
  void deallocLmem();
  cvk_tl_t *alloc_lmem(const cvk_tl_shape_t &shape, bool clean) const;
  cvk_tl_stride_t tl_fp32_stride(const cvk_tl_shape_t &shape,
                                 int eu_align) const;

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  int32_t n, c, h, w;
  cvk_fmt_t from;
  cvk_fmt_t to;
  int32_t from_byte;
  int32_t to_byte;
  int32_t load_unit;
  int32_t store_unit;
  int32_t layer_id;
  int32_t flip = 0;
  float const_scale;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
