/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: fc_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "cvi_backend_fc_kernel"

//
// New shape/stride to align EU_NUM
// adjust w as EU_NUM
//
static void initLHSForTiu(const CviBackendContext &ctx, cvk_ml_t *dst,
    const cvk_ml_t *src) {
  ctx.lmem_init_matrix(
      dst,
      {src->shape.n, src->shape.c, static_cast<uint32_t>(EU_NUM),
       src->shape.col},
      src->fmt,
      1);
  dst->start_address = src->start_address;
}

static void initRHSForTiu(const CviBackendContext &ctx, cvk_ml_t *dst,
    const cvk_ml_t *src) {
  initLHSForTiu(ctx, dst, src);
}

static void initBiasForTiu(const CviBackendContext &ctx, cvk_ml_t *dst,
    const cvk_ml_t *src) {
  if (src)
    initLHSForTiu(ctx, dst, src);
}

//
// Shape/stride used in TDMA may not the same as in TIU.
// Adjust shape/stride for TIU.
//
// E.g.
//   Y(0, 4) = L(1, 256) * R(256, 4) + B(1, 4)
//
//   TDMA:
//      L(0, 16, 1, 16)
//      R(255, 1, 1, 4)
//      B(0, 1, 1, 4)
//
//   TIU:
//       Y res0(1, 1, 1, 16)
//       L opd0(1, 16, 1, 16)
//       R opd1(256, 1, 1, 16)
//       B opd2(1, 1, 1, 16)
//
static void matrix_multiplication(const CviBackendContext &ctx,
                                  cvk_tiu_matrix_multiplication_param_t &p) {
  // No need to adjust shape/stride
  if (p.res->shape.w >= static_cast<uint32_t>(EU_NUM)) {
    LLVM_DEBUG(llvm::errs() << llvm::format(
        "  L(%d, %d), R(%d, %d)\n", p.left->shape.n,
        p.left->shape.col, p.right->shape.n, p.right->shape.col));
    ctx.tiu_matrix_multiplication(&p);

    return;
  }

  //
  // New shape/stride to align EU_NUM
  // adjust w as EU_NUM
  //
  cvk_ml_t tl_res;
  initLHSForTiu(ctx, &tl_res, p.res);

  cvk_ml_t tl_right;
  initRHSForTiu(ctx, &tl_right, p.right);

  cvk_ml_t tl_bias = {0};
  initBiasForTiu(ctx, &tl_bias, p.bias);

  cvk_tiu_matrix_multiplication_param_t p2 = p;
  p2.res = &tl_res;
  p2.left = p.left;
  p2.right = &tl_right;
  p2.bias = p.bias ? &tl_bias : nullptr;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  Modified L(%d, %d), R(%d, %d)\n", p2.left->shape.n,
      p2.left->shape.col, p2.right->shape.n,
      p2.right->shape.col));

  ctx.tiu_matrix_multiplication(&p2);
}

static void matrix_multiplication(
    const CviBackendContext &ctx,
    cvk_tiu_matrix_multiplication_qm_param_t &p) {
  // No need to adjust shape/stride
  if (p.res->shape.w >= static_cast<uint32_t>(EU_NUM)) {
    LLVM_DEBUG(llvm::dbgs() << llvm::format(
        "  L(%d, %d), R(%d, %d)\n",
        p.left->shape.n, p.left->shape.col, p.right->shape.n,
        p.right->shape.col));

    ctx.tiu_matrix_multiplication_qm(&p);

    return;
  }

  cvk_ml_t tl_res;
  initLHSForTiu(ctx, &tl_res, p.res);

  cvk_ml_t tl_right;
  initRHSForTiu(ctx, &tl_right, p.right);

  cvk_ml_t tl_bias = {0};
  initBiasForTiu(ctx, &tl_bias, p.bias);

  cvk_tiu_matrix_multiplication_qm_param_t p2 = p;
  p2.res = &tl_res;
  p2.left = p.left;
  p2.right = &tl_right;
  p2.bias = p.bias ? &tl_bias : nullptr;

  LLVM_DEBUG(llvm::dbgs() << llvm::format(
      "  Modified L(%d, %d), R(%d, %d)\n",
      p2.left->shape.n, p2.left->shape.col, p2.right->shape.n,
      p2.right->shape.col));

  ctx.tiu_matrix_multiplication_qm(&p2);
}

static void fc_slicing_multi_dimention(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_ofmap,
    uint32_t input_row_num, uint32_t input_col_num, uint32_t weight_col_num,
    bool do_bias, bool do_relu, bool weight_tp, int quant_rshift,
    uint32_t quant_multiplier) {
  // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
  uint32_t M = static_cast<uint32_t>(input_row_num);
  uint32_t K = static_cast<uint32_t>(input_col_num);
  uint32_t N = static_cast<uint32_t>(weight_col_num);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "fc_slicing_multi_dimension\n"
      "  Y(%d, %d) = L(%d, %d) * R(%d, %d) + B(%d, %d)\n",
      M, N, M, K, K, N, 1, N));

  // Split N <= max total eu number
  uint32_t total_eu = NPU_NUM * EU_NUM;
  uint32_t tiled_N = (N >= total_eu) ? total_eu : N;

  // Split K based on lane size
  uint32_t lane_size = LOCAL_MEM_SIZE;
  uint32_t max_k = (1 << 12) - 1;  // 1880v2: 12 bit
  uint32_t tiled_K = (K >= max_k) ? max_k : K;

  // Tiled Y
  cvk_ml_t tl_tiled_Y = {0};
  tl_tiled_Y.fmt = CVK_FMT_I8;

  // Tiled L
  cvk_ml_t tl_tiled_L = {0};
  tl_tiled_L.fmt = CVK_FMT_I8;

  // Tiled R
  cvk_ml_t tl_tiled_R = {0};
  tl_tiled_R.fmt = CVK_FMT_I8;

  // Tiled B
  cvk_ml_t tl_tiled_B = {0};
  if (do_bias) {
    // TIU opd2_n = 1, H/W use b_stride to read upper 8-bit
    // But bmk demand n = 2, but assign opd2_n = 1.
    // Let dma load and tiu use different shape.
    tl_tiled_B.fmt = CVK_FMT_I8;
    tl_tiled_B.shape = ctx.ml_default_shape(2, tiled_N, CVK_FMT_I8);  // 16bit
    tl_tiled_B.stride = ctx.ml_default_stride(tl_tiled_B.shape, tl_tiled_B.fmt, 1);
  }

  // Tiled local memory layout:
  //   Y at fixed position since last tiled ones may be smaller
  //
  //   tiled Y, [7:0]
  //   tiled Y, [15:8]
  //   tiled Y, [23:16]
  //   tiled Y, [31:24]
  //   tiled L
  //   tiled R
  //   tiled B, [7:0], if existed
  //   tiled B, [15:8], if existed

  // Find max tiled K
  uint32_t required_size = 0;
  do {
    required_size = 0;  // Start of LMEM

    // Not split M since we don't want to reload L(weight)
    // or reload partial result of different M.
    //
    // Y(M, N) = L(M, K) * R(K, N) + B(1, N)
    // tiled_Y(M, tiled_N) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) +
    //                       tiled_B(1, tiled_N)

    // tiled Y, 32bit
    tl_tiled_Y.start_address = required_size;
    tl_tiled_Y.shape = ctx.ml_default_shape(M, tiled_N, CVK_FMT_I8);
    tl_tiled_Y.stride = ctx.ml_default_stride(tl_tiled_Y.shape, tl_tiled_Y.fmt, 1);
    required_size += 4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n;

    // tiled L
    tl_tiled_L.start_address = required_size;
    tl_tiled_L.shape = ctx.ml_default_shape(M, tiled_K, CVK_FMT_I8);
    tl_tiled_L.stride = ctx.ml_default_stride(tl_tiled_L.shape, tl_tiled_L.fmt, 1);
    required_size += tl_tiled_L.shape.n * tl_tiled_L.stride.n;  // assume n = 2

    // tiled R
    tl_tiled_R.start_address = required_size;
    tl_tiled_R.shape = ctx.ml_default_shape(tiled_K, tiled_N, CVK_FMT_I8);
    tl_tiled_R.stride = ctx.ml_default_stride(tl_tiled_R.shape, tl_tiled_R.fmt, 1);
    required_size += tl_tiled_R.shape.n * tl_tiled_R.stride.n;

    // tiled B, 16bit
    if (do_bias) {
      tl_tiled_B.start_address = required_size;
      required_size += tl_tiled_B.shape.n * tl_tiled_B.stride.n;
    }

    if (required_size <= lane_size) {
      LLVM_DEBUG(llvm::errs() << llvm::format(
          "  tiled_Y %d, tiled_L %d, tiled_R %d, tiled_B %d\n",
          4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n,
          tl_tiled_L.shape.n * tl_tiled_L.stride.n,
          tl_tiled_R.shape.n * tl_tiled_R.stride.n,
          2 * tl_tiled_B.shape.n * tl_tiled_B.stride.n));
      break;
    }

  } while (--tiled_K);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  tiled_Y(%d, %d) = tiled_L(%d, %d) * tiled_R(%d, %d) "
      "+ tiled_B(%d, %d), required_size %d kB\n",
      M, tiled_N, M, tiled_K, tiled_K, tiled_N, 1, tiled_N,
      required_size / 1024));

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  tiled_Y shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
      "  tiled_L shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
      "  tiled_R shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
      "  tiled_B shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n",
      tl_tiled_Y.shape.n, tl_tiled_Y.shape.c, tl_tiled_Y.shape.w,
      tl_tiled_Y.shape.col, tl_tiled_Y.stride.n, tl_tiled_Y.stride.c,
      tl_tiled_Y.stride.h, tl_tiled_L.shape.n, tl_tiled_L.shape.c,
      tl_tiled_L.shape.w, tl_tiled_L.shape.col, tl_tiled_L.stride.n,
      tl_tiled_L.stride.c, tl_tiled_L.stride.h, tl_tiled_R.shape.n,
      tl_tiled_R.shape.c, tl_tiled_R.shape.w, tl_tiled_R.shape.col,
      tl_tiled_R.stride.n, tl_tiled_R.stride.c, tl_tiled_R.stride.h,
      tl_tiled_B.shape.n, tl_tiled_B.shape.c, tl_tiled_B.shape.w,
      tl_tiled_B.shape.col, tl_tiled_B.stride.n, tl_tiled_B.stride.c,
      tl_tiled_B.stride.h));

  assert(tiled_K && "Expect tiled column of left matrix");
  if (!tiled_K) {
    return;
  }

  bool loadInputOnce = false;
  if ((tl_tiled_L.shape.n == input_row_num) &&
      (tl_tiled_L.shape.col == input_col_num))
    loadInputOnce = true;

  // Each tiled_R(weight) is only loaded once.
  // tiled_L(input) reload is reload once tiled_weight moves right.
  //
  // for each tiled N
  for (uint32_t offset_N = 0; offset_N < N; offset_N += tiled_N) {
    // Y = [Y0, Y1, ... Yn-1]

    // Actual width
    uint32_t width_N = ((offset_N + tiled_N) <= N) ? tiled_N : (N - offset_N);
    width_N = ALIGN(width_N, 16); //Align 16 for better performance

    // for each tiled K
    for (uint32_t offset_K = 0; offset_K < K; offset_K += tiled_K) {
      // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
      // tiled_Y(M, tiled_K) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) +
      //                       tiled_B(1, tiled_N)
      //
      // L = [L0, L1, ... Lk-1]
      // R = [R0,0,   R0,1,   ..., R0,n-1
      //      R1,0,
      //
      //      Rk-1,0, Rk-1,1, ..., Rk-1,n-1]
      // B = [B0, B1, ... Bn-1]
      //
      // tiled_y,i += L0 * R0,i + L1 * R1,i + ... + Ln-1 * Rk-1,i + Bi

      // Actual width
      uint32_t width_K = ((offset_K + tiled_K) <= K) ? tiled_K : (K - offset_K);

      required_size = 0;  // Start of LMEM

      // tiled Y, 32bit
      tl_tiled_Y.start_address = required_size;
      tl_tiled_Y.shape = ctx.ml_default_shape(M, width_N, CVK_FMT_I8);  // actual width
      tl_tiled_Y.stride =
          ctx.ml_default_stride(tl_tiled_Y.shape, tl_tiled_Y.fmt, 1);
      required_size += 4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n;

      // Load tiled L from global memory, input
      bool inputNeeded = (offset_N && loadInputOnce) ? false : true;
      tl_tiled_L.start_address = required_size;
      tl_tiled_L.shape = ctx.ml_default_shape(M, width_K, CVK_FMT_I8);  // actual width
      tl_tiled_L.stride =
          ctx.ml_default_stride(tl_tiled_L.shape, tl_tiled_L.fmt, 1);
      required_size += tl_tiled_L.shape.n * tl_tiled_L.stride.n;
      if (inputNeeded) {
        ctx.tdma_load_stride(&tl_tiled_L, ga_ifmap + offset_K,
                            {K} // original column width
                            );
      }

      // Load tiled R from global memory, weight
      tl_tiled_R.start_address = required_size;
      tl_tiled_R.shape = ctx.ml_default_shape(width_K, width_N, CVK_FMT_I8); // actual width
      tl_tiled_R.stride =
          ctx.ml_default_stride(tl_tiled_R.shape, tl_tiled_R.fmt, 1);
      required_size += tl_tiled_R.shape.n * tl_tiled_R.stride.n;
      ctx.tdma_load_stride(&tl_tiled_R, ga_weight + offset_K * N + offset_N,
                           {N}  // original column width
                           );

      // Load tiled B from gobale memory at last time, bias
      // we need temporary shape to load lower 8bit and upper 8bit
      bool is_last_tile = ((offset_K + tiled_K) >= K) ? true : false;
      bool B_needed = (is_last_tile && do_bias) ? true : false;
      if (B_needed) {
        tl_tiled_B.start_address = required_size;

        tl_tiled_B.shape = ctx.ml_default_shape(2, width_N, CVK_FMT_I8);  // actual width
        tl_tiled_B.stride =
            ctx.ml_default_stride(tl_tiled_B.shape, tl_tiled_B.fmt, 1);

        ctx.tdma_load_stride(&tl_tiled_B, ga_bias + offset_N,
                             {N}  // original column width
                             );
      }

      uint32_t ps32_mode = 0;    // normal mode
      uint32_t relu_enable = 0;  // 1880v2 relu can be used in ps32_mode
      if (tiled_K < K) {
        if (offset_K == 0) {        // first tile
          ps32_mode = 2;            // write 32b result at the first time
        } else if (is_last_tile) {  // last tile
          ps32_mode = 1;            // load previous 32-bit result
        } else {
          ps32_mode = 3;  // init & write 32bits partial sum
        }
      }

      // No tiling or last tile
      if ((ps32_mode == 0 || ps32_mode == 1) && do_relu) {
        relu_enable = 1;
      }

      // New multiplier and 32bit bias are only used in final post data
      // processing stage.
      // So, only set chan_quan = 1 if no tiling or last tile.
      // And when chan_quan is enabled, des_opt_res0_int8 must be 1
      if (quant_multiplier) {
        cvk_tiu_matrix_multiplication_qm_param_t p = {0};
        p.res = &tl_tiled_Y;
        p.left = &tl_tiled_L;
        p.right = &tl_tiled_R;
        p.bias = B_needed ? &tl_tiled_B : nullptr;
        p.rshift_bits = is_last_tile ? quant_rshift : 0;  // quantization down
        p.relu_enable = relu_enable;
        p.ps32_mode = ps32_mode;
        p.quan_m = quant_multiplier;
        p.layer_id = layer_id;

        LLVM_DEBUG(llvm::errs() << llvm::format(
            "  offset_N %d, offset_K %d, L(%d, %d), R(%d, %d)\n",
            offset_N, offset_K, p.left->shape.n, p.left->shape.col,
            p.right->shape.n, p.right->shape.col));

        matrix_multiplication(ctx, p);

      } else {
        cvk_tiu_matrix_multiplication_param_t p = {0};
        p.res = &tl_tiled_Y;
        p.left = &tl_tiled_L;
        p.right = &tl_tiled_R;
        p.bias = B_needed ? &tl_tiled_B : nullptr;
        p.lshift_bits = 0;                                // deprecated
        p.rshift_bits = is_last_tile ? quant_rshift : 0;  // quantization down
        p.res_is_int8 = is_last_tile ? 1 : 0;             // output 8bit
        p.add_result = 0;                                 // deprecated
        p.relu_enable = relu_enable;
        p.ps32_mode = ps32_mode;
        p.layer_id = layer_id;

        LLVM_DEBUG(llvm::errs() << llvm::format(
            "  offset_N %d, offset_K %d, L(%d, %d), R(%d, %d)\n",
            offset_N, offset_K, p.left->shape.n, p.left->shape.col,
            p.right->shape.n, p.right->shape.col));

        matrix_multiplication(ctx, p);
      }

      // Store tiled_Y to global memory
      if (is_last_tile) {
        uint32_t real_width_N = ((offset_N + tiled_N) <= N) ? tiled_N : (N - offset_N);
        tl_tiled_Y.shape = ctx.ml_default_shape(M, real_width_N, CVK_FMT_I8);  // actual width
        tl_tiled_Y.stride = ctx.ml_default_stride(tl_tiled_Y.shape, tl_tiled_Y.fmt, 1);
        ctx.tdma_store_stride(&tl_tiled_Y, ga_ofmap + offset_N,
                              {N} // original column width
        );
      }

    }  // for (uint32_t offset_K = 0; offset_K < K; offset_K += tiled_K)
  }    // for (uint32_t offst_N = 0; offset_N < N; offset_N += tiled_N)
}

void cvi_backend_tg_fixed_fc(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_ofmap, int input_row,
    int input_col, int output_col, bool do_bias, bool do_relu, bool weight_tp,
    int quant_rshift, uint32_t quant_multiplier) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
             "cvi_backend_tg_fixed_fc\n"
             "    in (%d, %d), out (%d), do_bias %d, do_relu %d, "
             "weight_tp %d, quant_rshift %d\n",
             input_row, input_col, output_col, do_bias, do_relu, weight_tp,
             quant_rshift, quant_multiplier));

  ctx.set_layer_id(layer_id); // pmu used

  fc_slicing_multi_dimention(ctx, layer_id, ga_ifmap, ga_weight, ga_bias,
      ga_ofmap, static_cast<uint32_t>(input_row),
      static_cast<uint32_t>(input_col), static_cast<uint32_t>(output_col),
      do_bias, do_relu, weight_tp, quant_rshift, quant_multiplier);
}

void cvi_backend_tg_fixed_fc_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_data_gaddr, gaddr_t weight_data_gaddr, gaddr_t bias_data_gaddr,
    gaddr_t top_data_gaddr, int in_row, int in_col, int out_col, int have_bias, int do_activation,
    int activation_method, gaddr_t activation_ga_slope, int activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, bool weight_tp, int left_shift_width, int right_shift_width,
    int threshold_x_quantized_len, const int *threshold_x_quantized, const int *right_shift_array) {
  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tg_fixed_fc_kernel\n"
                                        "    in (%d, %d), out (%d), has_bias %d, do_activation %d, "
                                        "activation_method %d, weight_tp %d\n",
                                        in_row, in_col, out_col, have_bias, do_activation,
                                        activation_method, weight_tp));

  assert(!weight_tp && "weight transpose deprecated");

  cvi_backend_tg_fixed_fc(ctx, layer_id, bottom_data_gaddr, weight_data_gaddr,
      bias_data_gaddr, top_data_gaddr, in_row, in_col, out_col,
      have_bias ? true : false,
      do_activation ? true : false,
      weight_tp, right_shift_width,
      right_shift_array ? right_shift_array[0] : 0);
}
