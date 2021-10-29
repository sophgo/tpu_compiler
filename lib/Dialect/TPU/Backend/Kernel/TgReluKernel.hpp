/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-17
 */
#ifndef TG_RELU_KERNEL_HPP
#define TG_RELU_KERNEL_HPP

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

class TgReluKernel {
public:
  TgReluKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  typedef enum { RELU, LEAKY_RELU, PRELU } mode_t;

  void init(uint32_t layer_id, int32_t n, int32_t c, int32_t h, int32_t w,
            gaddr_t ga_input, gaddr_t ga_output, gaddr_t ga_negative_slope,
            float negative_slope, int GT_rshift, int GT_scale, int LE_rshift,
            int LE_scale, cvk_fmt_t fmt, mode_t mode);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void allocLmem();
  void deallocLmem();
  void compute_relu(int32_t step_idx, int32_t flip);
  void compute_leaky_relu_fixed_sym(int32_t step_idx, int32_t flip);
  void compute_leaky_relu_bf16(int32_t step_idx, int32_t flip);
  void compute_prelu_fixed(int32_t step_idx, int32_t flip);
  void compute_prelu_bf16(int32_t step_idx, int32_t flip);
  cvk_tl_t get_input(int32_t step_idx, int32_t flip);
  cvk_tl_t get_output(int32_t step_idx, int32_t flip);
  void change_workspace_size(int32_t step_idx);

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];
  cvk_tl_t *tl_slope; // for prelu
  cvk_tl_t *tl_working[2];
  cvk_tl_t *tl_pos_neg_map;
  cvk_tg_stride_t gstride;

  int32_t n, c, h, w;
  int32_t layer_id;
  cvk_fmt_t fmt;
  mode_t mode;

  gaddr_t ga_slope; // for prelu
  int GT_rshift;    // for i8
  int GT_scale;     // for i8
  int LE_rshift;    // for i8
  int LE_scale;     // for i8
  float negative_slope;

  int32_t flip = 0;
  std::vector<CviBackendContext::tiling_info_t> tiles;
};

#endif
