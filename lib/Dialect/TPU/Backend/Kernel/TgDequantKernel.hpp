/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_DEQUANT_KERNEL_HPP
#define TG_DEQUANT_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

typedef enum {
  AXIS_ALL,
  AXIS_W,
  AXIS_C,
} axis_mode_t;

class TgDequantKernel {
public:
  TgDequantKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_scale,
            gaddr_t ga_zeropoint, gaddr_t ga_output, int axis, int n, int c,
            int h, int w);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void refresh(int32_t step_idx);
  void reshape();
  void tiling_axis_w();

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_output;
  gaddr_t ga_scale;
  gaddr_t ga_zeropoint;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_scale;
  cvk_tl_t tl_zeropoint;
  cvk_tl_t *tl_mem[4];

  int32_t n, c, h, w;
  int32_t axis;
  int32_t axis_dim;
  int32_t layer_id;
  cvk_fmt_t fmt;
  axis_mode_t mode;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
