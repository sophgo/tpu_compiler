/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_TILE_KERNEL_HPP
#define TG_TILE_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgTileKernel {
public:
  TgTileKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, int axis, int factor, cvk_fmt_t fmt);

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
  void g2g_tile_N();
  void g2g_tile_W();

protected:
  typedef enum {
    TILE_N,
    TILE_W,
  } tile_mode_t;
  typedef struct {
    int pos_c;
    int c;
  } tiling_t;
  const CviBackendContext &ctx;
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_fmt_t fmt;
  int fmt_bytes;
  int n, c, h, w, factor;
  cvk_tg_stride_t src_gstride;
  cvk_tl_t *tl_mem[4];
  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  tile_mode_t mode;
  int step_c;
  std::vector<tiling_t> tiles;
};

#endif
