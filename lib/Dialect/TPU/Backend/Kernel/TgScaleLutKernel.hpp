/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_SCALE_LUT_KERNEL_HPP
#define TG_SCALE_LUT_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgScaleLutKernel {
public:
  TgScaleLutKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
            gaddr_t table_gaddr, int n, int c, int h, int w, cvk_fmt_t fmt);

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
  gaddr_t ga_lut;
  cvk_fmt_t fmt;
  int n, c, h, w, c_times;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tg_stride_t gstride;
  cvk_tl_shape_t lut_shape;
  static const int BLOB_NUM = 4;
  cvk_tl_t *tl_mem[BLOB_NUM];
  cvk_tl_t *tl_lut;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
