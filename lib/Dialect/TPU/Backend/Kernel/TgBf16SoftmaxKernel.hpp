/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 */
#ifndef TG_SOFTMAX_KERNEL_HPP
#define TG_SOFTMAX_KERNEL_HPP

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

class TgSoftmaxKernel {
public:
  TgSoftmaxKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id,
            gaddr_t ga_input,
            gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
            gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
            gaddr_t ga_output,
            int64_t* shape, int axis, int dimension);
  void selectTilePolicy();
  void schedule();

protected:
  typedef struct {
    int pos_h;
    int h;
  } tiling_t;
  enum SoftmaxMode {Softmax2D, Softmax4D};
  void selectSoftmaxMode(int64_t* shape);
  void fillOneAsGolden();
  void matrixToTensor(cvk_tl_t *tensor, const cvk_ml_t &matrix);
  unsigned int doSplitHeightBf16softmax2DParallelInnerSize();
  unsigned int doSplitHeightBf16softmax2DParallelOuterSize();
  void softmaxLargeSizeHandler();
  void bf16_softmax_kernel_2d_parallel_inner_size();
  void bf16_softmax_kernel_2d_parallel_outer_size();
  int doSplitHeightBf16softmax4D();
  void bf16_softmax_kernel_4d();
  void bf16_softmax_kernel_2d();
  void exponential(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work);
  void reciprocal(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work);
  
  void init_table();
  void free_table();
  void assign_matrix(cvk_ml_t *ml_mem, const cvk_ml_shape_t &shape);
  void fill_matrix(cvk_ml_t *ml_mem, uint32_t row, uint32_t col, uint32_t addr);
  void assign_addr(cvk_tl_t *tl_mem, uint32_t size);
  void matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix);
  void matrix_for_tiu(cvk_ml_t *matrix);
  void matrix_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                  const cvk_ml_t &ml_right, const cvk_ml_t &ml_bias,
                  uint8_t ps32_mode = 0);
  void matrix_recurrence(const cvk_ml_t &ml_res, int flip, const tiling_t &tile,
                         gaddr_t ga_weight, gaddr_t ga_bias);
  void zeros(const cvk_ml_t &matrix);

  void max_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out);
  void accumulate_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out);
  void exponential(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
               const cvk_ml_t &ml_buff);
  void reciprocal(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
            const cvk_ml_t &ml_buff);

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_exponential_table_data_lut;
  gaddr_t ga_exponential_slope_table_data_lut;
  gaddr_t ga_reciprocal_table_data_lut;
  gaddr_t ga_reciprocal_table_mantissa_data_lut;
  gaddr_t ga_output;
  int axis; 
  int dimension;
  int outer_size;
  int inner_size;
  cvk_fmt_t fmt;
  int fmt_size;
  SoftmaxMode functionMode;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t layer_id;

  // for lmem addr alloc
  cvk_tl_shape_t table_shape;
  uint32_t table_size;
  cvk_tl_t *tl_exponential_table_answer;
  cvk_tl_t *tl_exponential_table_answer_slope;
  cvk_tl_t *tl_reciprocal_table_answer;
  cvk_tl_t *tl_reciprocal_mantissa_table_answer;
  uint32_t lmem_used;

  // for tiling
  int step_size;
  int step_num;
  std::vector<tiling_t> tiles;         // tilng hidden_size
  std::vector<cvk_ml_t> ml_hiddens[2]; // one for backup
  uint32_t addr_recurrence;            // for recurrence
  uint32_t addr_bias;
  uint32_t addr_work0; // for lut buffer and ps32 bias buffer
  uint32_t addr_work1; // for dot(h_t,r) result
  uint32_t addr_xz;    // for z
  uint32_t addr_xh;    // for r/h
  cvk_mg_stride_t x_gstride;
  cvk_mg_stride_t h_gstride;
};

#endif
