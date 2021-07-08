/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */
#ifndef TG_FIXED_POOLING_KERNEL_HPP
#define TG_FIXED_POOLING_KERNEL_HPP

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

typedef struct {
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t c_pos;
  int32_t oh;
  int32_t ow;
  int32_t ih_pos;
  int32_t oh_pos;
  int32_t pad[4];
  uint64_t input_offset;
  uint64_t output_offset;
} PoolingTile;

class TgInt8PoolingKernel {
public:
  TgInt8PoolingKernel(const CviBackendContext &ctx)
    : ctx(ctx) {}

  void init(uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t pad_t, int32_t pad_b, int32_t pad_l, int32_t pad_r,
    int32_t kh, int32_t kw, int32_t stride_h, int32_t stride_w,
    bool is_avg_pooling, bool do_relu, int32_t rshift,
    int32_t multipliers, bool ceil_mode,
    int32_t store_cmpr_act = 0, int32_t load_cmpr_act = 0,
    int32_t store_cmpr_act_c_step = 0, int32_t load_cmpr_act_c_step = 0);

  void selectTilePolicy();
  void schedule();

protected:

  void allocLmem(
      cvk_tl_shape_t &input_shape,
      cvk_tl_shape_t &output_shape);
  void deallocLmem();
  void doTileForNormalCase();
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void loadDecompressed(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void storeCompressed(int32_t step_idx, int32_t flip);
  void adjustPadding();
  const CviBackendContext &ctx;

  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t oh;
  int32_t ow;
  int32_t pad_t;
  int32_t pad_b;
  int32_t pad_l;
  int32_t pad_r;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t rshift;
  int32_t multiplier;
  int32_t layer_id;
  bool do_relu;
  bool is_avg_pooling;
  bool ceil_mode;

  int32_t flip = 0;

  std::vector<PoolingTile> tiles;

  int32_t store_cmpr_act = 0;
  int32_t load_cmpr_act = 0;
  int32_t store_cmpr_act_c_step = 0;
  int32_t load_cmpr_act_c_step = 0;
};

#endif
