/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_ZERO_MASK_KERNEL_HPP
#define TG_ZERO_MASK_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgZeroMaskKernel {
public:
  TgZeroMaskKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, bool positive, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void compute_bf16();
  void compute_int8();
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();

protected:
  const CviBackendContext &ctx;
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_fmt_t fmt;
  int n, c, h, w;
  bool positive;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tg_stride_t gstride;
  cvk_tl_t *tl_mem[4];
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
