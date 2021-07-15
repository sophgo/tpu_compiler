/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#ifndef TG_FC_KERNEL_HPP
#define TG_FC_KERNEL_HPP

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

// Y[M, N] = L[M,K] * R[K,N] + B[4,N]
class TgFcKernel {
public:
  TgFcKernel(const CviBackendContext &ctx) : ctx(ctx) {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight,
            gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
            bool do_bias, bool do_relu, std::vector<int> *rshift_width,
            std::vector<int> *multiplier, int batch_high, int batch_low,
            bool lstride, bool rstride, bool ostride,
            std::vector<int> compressed_pos, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  uint32_t lmem_matrix_size(uint32_t row, uint32_t col,
                            bool ps32 = false) const;

  void update_tl_matrix(int32_t step_idx);
  void set_laddr();
  void matrix_for_tiu();
  bool is_last_k(int32_t step_idx) const;
  inline uint32_t slice_m() const { return (M + tile_M - 1) / tile_M; }
  inline uint32_t slice_k() const { return (K + tile_K - 1) / tile_K; }
  inline uint32_t slice_n() const { return (N + tile_N - 1) / tile_N; }
  typedef struct {
    uint32_t L, R, B, Y;
    uint32_t blob_L, blob_R, blob_B, blob_Y;
  } lmem_size_t;
  lmem_size_t get_lmem_size() const;
  uint32_t total_lmem_size() const;
  void update_batch_info(int high_idx, int low_idx);

  bool try_tiling_group_parallel();
  bool try_no_tiling();
  bool try_tiling_parallel();
  bool try_tiling_no_parallel();
  void tiling_generic();
  void schedule_parallel();
  void schedule_group_parallel();
  void schedule_no_parallel();

protected:
  const CviBackendContext &ctx;
  gaddr_t ga_input;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  gaddr_t ga_output;

  gaddr_t ga_i, ga_w, ga_o, ga_b; // for origin addr

  uint32_t M;
  uint32_t K;
  uint32_t N;

  bool do_bias;
  bool do_relu;
  std::vector<int> rshift;
  std::vector<int> multiplier;
  int cur_rshift;
  int cur_multiplier;
  std::vector<int> compressed_pos;
  cvk_fmt_t fmt;
  int fmt_size;
  uint32_t layer_id;

  cvk_ml_t tl_Y;
  cvk_ml_t tl_L;
  cvk_ml_t tl_B;
  cvk_ml_t tl_R;

  int batch_high;
  int batch_low;
  int batch; // high * low
  bool lstride;
  bool rstride;
  bool ostride;
  cvk_mg_stride_t left_gstride;
  cvk_mg_stride_t right_gstride;
  cvk_mg_stride_t output_gstride;

  bool do_parallel;
  uint32_t maxM, maxK, maxN;
  uint32_t TOTAL_EU;
  uint32_t tile_M;
  uint32_t tile_K;
  uint32_t tile_N;
  int compress_offset; // for batch compress pos
  typedef struct TileInfo {
    uint32_t pos_m;
    uint32_t pos_k;
    uint32_t pos_n;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    int RB_idx;
    int L_idx;
    int Y_idx;
    int batch_high;
    int batch_low;
    int compress_idx; // compress pos
  } tile_info_t;
  std::vector<tile_info_t> tiles;
  typedef enum {
    FC_NO_TILING,
    FC_GROUP_PARALLEL,
    FC_PARALLEL,
    FC_NO_PARALLEL,
  } fc_mode_t;
  fc_mode_t mode;
  int total_steps;
  std::vector<uint32_t> Y_laddr; // [M, tile_N]
  uint32_t L_laddr[2];           // [M, tile_K]
  uint32_t R_laddr[2];           // [tile_K, tile_N]
  uint32_t B_laddr[2];           // [4, tile_N]
};

#endif
