/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-01
 */
#ifndef TG_CONCAT_KERNEL_HPP
#define TG_CONCAT_KERNEL_HPP

#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgConcatKernel {
public:
  TgConcatKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, int input_num, int dim_size, int concat_axis,
            gaddr_t input_gaddrs[], gaddr_t output_gaddr, int axis_dims[],
            int output_dim[], bool do_relu, const int right_shift_width[],
            const int threshold_x_quantized[], cvk_fmt_t fmt);
  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void update_output(int output_dim[], int dim_size, int concat_axis);
  uint32_t &axis_dim(cvk_tg_shape_t &shape);
  uint64_t axis_size(const cvk_tg_shape_t &shape) const;
  uint64_t dst_offset(const CviBackendContext::tiling_info_t &tile) const;

protected:
  const CviBackendContext &ctx;

  gaddr_t ga_output;
  cvk_tg_shape_t output_shape;
  cvk_tg_stride_t output_stride;
  bool do_relu;
  int dim_size;
  int axis;
  int input_num;
  cvk_fmt_t fmt;
  int32_t layer_id;
  uint32_t base_addr[4]; // do flip, 2 input, 2 output
  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  bool do_parallel;

  typedef struct {
    bool do_quantize;
    gaddr_t ga_input;
    gaddr_t ga_output;
    cvk_tg_shape_t shape;
    cvk_tg_stride_t stride;
    int rshift_width;
    int data_quantized;
    int tile_idx;
    std::vector<CviBackendContext::tiling_info_t> tiles;
  } input_info_t;
  std::vector<input_info_t> inputs;
  CviBackendContext::tiling_mode_t tiling_mode;
  int total_tiles;
  void prepare(int32_t step_idx, input_info_t *&input,
               CviBackendContext::tiling_info_t *&tile);
};

#endif
