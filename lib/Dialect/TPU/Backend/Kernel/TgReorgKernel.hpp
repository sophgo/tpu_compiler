/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_REORG_KERNEL_HPP
#define TG_REORG_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgReorgKernel {
public:
  TgReorgKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, int stride, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void reshape();

protected:
  const CviBackendContext &ctx;
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_fmt_t fmt;
  int fmt_bytes;
  int n_loop, n_offset;
  int n, c, h, w, oh, ow, r;
  cvk_tg_stride_t src_gstride;
  cvk_tg_stride_t dst_gstride;
  cvk_tl_t *tl_mem[5];
  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tl_t tl_middle;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
