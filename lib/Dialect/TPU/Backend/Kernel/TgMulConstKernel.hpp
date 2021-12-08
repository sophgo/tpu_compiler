/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */
#pragma once

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgMulConstKernel {
public:
  TgMulConstKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int32_t n,
            int32_t c, int32_t h, int32_t w, bool do_relu, float const_val,
            cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();

protected:
  const CviBackendContext &ctx;
  bool do_relu;
  gaddr_t ga_input;
  gaddr_t ga_output;
  int32_t n, c, h, w;
  cvk_fmt_t fmt;
  int fmt_size;

  cvk_tl_t tl_input;
  cvk_tl_t *tl_mem[2];
  uint32_t layer_id;
  float const_val;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};
