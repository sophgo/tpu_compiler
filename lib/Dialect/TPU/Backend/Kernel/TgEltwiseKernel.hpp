/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 */
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
} TILE;

class TgInt8EltwiseKernel {
public:
  TgInt8EltwiseKernel(const CviBackendContext &ctx)
    : ctx(ctx) {}

  void init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);

  void selectTilePolicy();
  void schedule();

protected:
  virtual void compute(int32_t step_idx) = 0;

  void allocLmem(
      cvk_tl_shape_t &input_shape,
      cvk_tl_shape_t &output_shape);
  void deallocLmem();
  void doTileForNormalCase();
  void doTileForStrideCase();
  void load(int32_t step_idx);
  void store(int32_t step_idx);

  const CviBackendContext &ctx;

  gaddr_t *ga_inputs;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];
  cvk_tl_t *tl_output_h;

  int32_t operand_num;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  bool do_early_stride;
  int32_t stride_h;
  int32_t stride_w;
  int32_t rshift;
  const int32_t *multipliers;
  const int32_t *coeffs;
  int32_t layer_id;
  bool do_relu;

  int32_t input_flip = 0;
  int32_t output_flip = 0;

  std::vector<TILE> tiles;
};

class TgInt8EltwiseAddKernel : public TgInt8EltwiseKernel {
public:
  TgInt8EltwiseAddKernel(const CviBackendContext &ctx)
    : TgInt8EltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMaxKernel : public TgInt8EltwiseKernel {
public:
  TgInt8EltwiseMaxKernel(const CviBackendContext &ctx)
    : TgInt8EltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMinKernel : public TgInt8EltwiseKernel {
public:
  TgInt8EltwiseMinKernel(const CviBackendContext &ctx)
    : TgInt8EltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMulKernel : public TgInt8EltwiseKernel {
public:
  TgInt8EltwiseMulKernel(const CviBackendContext &ctx)
    : TgInt8EltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};
