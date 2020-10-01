/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 */
#ifndef TG_BF16_SQUARE_KERNEL_HPP
#define TG_BF16_SQUARE_KERNEL_HPP

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
  uint64_t input_offset;
  uint64_t output_offset;
} SquareTile;

class TgBf16SquareKernel {
public:
  TgBf16SquareKernel(const CviBackendContext &ctx)
    : ctx(ctx) {}

  void init(uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_output,
    int32_t n, int32_t c, int32_t h, int32_t w);

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
  void store(int32_t step_idx, int32_t flip);

  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t layer_id;

  int32_t flip = 0;

  std::vector<SquareTile> tiles;
};

#endif
