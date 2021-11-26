/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */
#pragma once

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
  int32_t n_pos;
  int32_t c_pos;
  int32_t h_pos;
  uint64_t input_offset;
} EltwiseConstTile;

class TgEltwiseConstKernel {
public:
  TgEltwiseConstKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int32_t n,
            int32_t c, int32_t h, int32_t w, bool do_relu, float const_val,
            int32_t coeff, int32_t rshift, std::vector<int8_t> &multiplier);

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int32_t n,
            int32_t c, int32_t h, int32_t w, bool do_relu, float const_val);
  void selectTilePolicy();
  void schedule();

protected:
  virtual void compute(int32_t step_idx) = 0;
  virtual void load(int32_t step_idx);
  virtual void store(int32_t step_idx);
  virtual void allocLmem();
  virtual void deallocLmem();

protected:
  const CviBackendContext &ctx;
  bool do_relu;
  gaddr_t ga_input;
  gaddr_t ga_output;
  int32_t n, c, h, w;
  cvk_fmt_t fmt;
  uint32_t block_num;
  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output_h;
  uint32_t layer_id;
  uint32_t elementSize;
  int32_t rshift;
  std::vector<int8_t> multipliers;
  float const_val;
  int32_t coeff;
  cvk_tl_shape_t load_shape;
  std::vector<EltwiseConstTile> tiles;
};

class TgInt8EltwiseConstAddKernel : public TgEltwiseConstKernel {
public:
  TgInt8EltwiseConstAddKernel(const CviBackendContext &ctx)
    : TgEltwiseConstKernel(ctx) {
      block_num = 3;
  }

protected:
  void compute(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
};

class TgInt8EltwiseConstMulKernel : public TgEltwiseConstKernel {
public:
  TgInt8EltwiseConstMulKernel(const CviBackendContext &ctx)
    : TgEltwiseConstKernel(ctx) {
      block_num = 2;
  }

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseConstAddKernel : public TgEltwiseConstKernel {
public:
  TgBf16EltwiseConstAddKernel(const CviBackendContext &ctx)
    : TgEltwiseConstKernel(ctx) {
      block_num = 2;
  }

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseConstMulKernel : public TgEltwiseConstKernel {
public:
  TgBf16EltwiseConstMulKernel(const CviBackendContext &ctx)
    : TgEltwiseConstKernel(ctx) {
      block_num = 2;
  }

protected:
  void compute(int32_t step_idx);
};