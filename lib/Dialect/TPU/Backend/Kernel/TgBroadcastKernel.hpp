/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_BROADCAST_KERNEL_HPP
#define TG_BROADCAST_KERNEL_HPP

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

typedef enum {
  BCAST_HW,
  BCAST_C,
  BCAST_ALL,
} bcast_mode_t;
typedef enum _bcast_t {
  BCAST_ADD,
  BCAST_SUB,
  BCAST_MUL,
} bcast_t;
bcast_mode_t mode;

class TgBcastKernel {
public:
  TgBcastKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_a, gaddr_t ga_b, gaddr_t ga_output,
            int an, int ac, int ah, int aw, int bn, int bc, int bh, int bw,
            bool do_relu, int32_t rshift, const int32_t *multipliers,
            bcast_t type, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void convert_shape(int an, int ac, int ah, int aw, int bn, int bc, int bh,
                     int bw);
  void schedule_bcast_all();
  void schedule_bcast_c();
  void schedule_bcast_hw();
  void tile_all();
  void tile_other();
  void tiu_compute(cvk_tl_t *tl_result, cvk_tl_t *tl_left, cvk_tl_t *tl_right,
                   cvk_tl_t *tl_buff = nullptr);

protected:
  const CviBackendContext &ctx;
  uint32_t layer_id;
  gaddr_t ga_a;
  gaddr_t ga_b;
  gaddr_t ga_output;
  int shape_a[4];
  int shape_b[4];
  bool do_relu;
  int32_t rshift;
  const int32_t *multipliers;
  bcast_mode_t mode;
  bcast_t type;
  cvk_fmt_t fmt;
  int fmt_bytes;
  int index_bcast;
  int num_blobs;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
