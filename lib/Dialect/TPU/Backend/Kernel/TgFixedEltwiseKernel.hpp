/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */
#ifndef TG_FIXED_ELTWISE_KERNEL_HPP
#define TG_FIXED_ELTWISE_KERNEL_HPP

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
} EltwiseTile;

class TgEltwiseKernel {
public:
  TgEltwiseKernel(const CviBackendContext &ctx)
    : ctx(ctx) {}

  void init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs,
    int32_t *inputs_offset = nullptr,
    int32_t output_offset = 0);

  void init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w,
    const float *coeffs);

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
  cvk_tl_t *tl_output_h[2];

  // low/high working space to do asymmetric quant
  cvk_tl_t *tl_working[2];

  int32_t operand_num;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  bool do_early_stride;
  int32_t stride_h;
  int32_t stride_w;
  int32_t rshift;
  int32_t output_offset;
  const int32_t *multipliers;
  const int32_t *coeffs;
  int32_t *inputs_offset;
  const float *coeffs_float;
  int32_t layer_id;
  bool do_relu;
  bool is_asymmetric = false;
  cvk_fmt_t fmt;
  uint32_t elementSize;

  int32_t input_flip = 0;
  int32_t output_flip = 0;

  std::vector<EltwiseTile> tiles;
};

class TgInt8EltwiseAddKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseAddKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
private:
  void symmetric_compute(const int opd_idx,
                            cvk_tl_t &input,
                            cvk_tl_t &output,
                            cvk_tl_t &output_high);
  void asymmetric_compute(const int opd_idx,
                            cvk_tl_t &input,
                            cvk_tl_t &output,
                            cvk_tl_t &output_high,
                            bool last_step);
  void reset(cvk_tl_t *tl);
};

class TgInt8EltwiseMaxKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMaxKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMinKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMinKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMulKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMulKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseAddKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseAddKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMaxKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMaxKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMinKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMinKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMulKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMulKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMinMaxKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMinMaxKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

class TgEltwiseAbsKernel : public TgEltwiseKernel {
public:
  TgEltwiseAbsKernel(const CviBackendContext &ctx)
    : TgEltwiseKernel(ctx) {}

protected:
  void compute(int32_t step_idx);
};

#endif
