/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_fc_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bf16_fc_kernel"

#define RELU   0
#define PRELU  1

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
  if (p.res->shape.w >= (uint32_t)EU_NUM / 2) {
    // LLVM_DEBUG(llvm::errs() << llvm::format("    L(%d, %d), R(%d, %d)\n", p.left->shape.n,
    //                                       p.left->shape.col, p.right->shape.n, p.right->shape.col););
    ctx.tiu_matrix_multiplication(&p);

    return;
  }

  //
  // New shape/stride to align EU_NUM / 2
  // adjust w as EU_NUM / 2
  //
  cvk_ml_t tl_res;
  tl_res.start_address = p.res->start_address;
  tl_res.fmt = p.res->fmt;
  tl_res.shape = {p.res->shape.n, p.res->shape.c, static_cast<uint32_t>(EU_NUM / 2),
                  p.res->shape.col};
  tl_res.stride = ctx.ml_default_stride(tl_res.shape, CVK_FMT_BF16, /*eu_align=*/1);

  cvk_ml_t tl_right;
  tl_right.start_address = p.right->start_address;
  tl_right.fmt = p.right->fmt;
  tl_right.shape = {p.right->shape.n, p.right->shape.c, static_cast<uint32_t>(EU_NUM / 2),
                    p.right->shape.col};
  tl_right.stride = ctx.ml_default_stride(tl_right.shape, CVK_FMT_BF16, /*eu_align=*/1);

  cvk_ml_t tl_bias = {0};
  if (p.bias) {
    tl_bias.start_address = p.bias->start_address;
    tl_bias.fmt = p.bias->fmt;
    tl_bias.shape = {p.bias->shape.n, p.bias->shape.c, static_cast<uint32_t>(EU_NUM / 2),
                     p.bias->shape.col};
    tl_bias.stride = ctx.ml_default_stride(tl_bias.shape, CVK_FMT_BF16, /*eu_align=*/1);
  }

  cvk_tiu_matrix_multiplication_param_t p2 = p;
  p2.res = &tl_res;
  p2.left = p.left;
  p2.right = &tl_right;
  p2.bias = p.bias ? &tl_bias : nullptr;

  LLVM_DEBUG(llvm::errs() << llvm::format("    Modified L(%d, %d), R(%d, %d)\n", p2.left->shape.n,
                                        p2.left->shape.col, p2.right->shape.n,
                                        p2.right->shape.col););

  ctx.tiu_matrix_multiplication(&p2);
}


static void fc_slicing_multi_dimention(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation, int activation_method) {
  // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
  uint32_t M = static_cast<uint32_t>(input_row_num);
  uint32_t K = static_cast<uint32_t>(input_col_num);
  uint32_t N = static_cast<uint32_t>(weight_col_num);

  LLVM_DEBUG(llvm::errs() << llvm::format("fc_slicing_multi_dimension\n"
                               "  Y(%d, %d) = L(%d, %d) * R(%d, %d) + B(%d, %d)\n",
                               M, N, M, K, K, N, 1, N););

  // Split N <= max total eu number
  uint32_t total_eu = NPU_NUM * EU_NUM / 2;
  uint32_t tiled_N = (N >= total_eu) ? total_eu : N;

  // Split K based on lane size
  uint32_t lane_size = LOCAL_MEM_SIZE;
  uint32_t max_k = (1 << 12) - 1;  // 1880v2: 12 bit
  uint32_t tiled_K = (K >= max_k) ? max_k : K;

  // Tiled Y
  cvk_ml_t tl_tiled_Y = {0};
  tl_tiled_Y.fmt = CVK_FMT_BF16;

  // Tiled L
  cvk_ml_t tl_tiled_L = {0};
  tl_tiled_L.fmt = CVK_FMT_BF16;

  // Tiled R
  cvk_ml_t tl_tiled_R = {0};
  tl_tiled_R.fmt = CVK_FMT_BF16;

  // Tiled B
  cvk_ml_t tl_tiled_B = {0};
  if (have_bias) {
    // tiu_matrix_multiplication will change shape.n from 2 to 1
    // So we use the shape for both dma load and local memory allocation.

    // Upper16 [31:16] then Lower16 [15:0] separated by b_stride
    tl_tiled_B.fmt = CVK_FMT_BF16;
    tl_tiled_B.shape = ctx.ml_default_shape(sizeof(uint32_t)/sizeof(uint16_t), tiled_N, CVK_FMT_BF16);  // 2 x 16bit
    tl_tiled_B.stride = ctx.ml_default_stride(tl_tiled_B.shape, CVK_FMT_BF16, /*eu_align=*/1);
  }

  // Tiled local memory layout:
  //   Y at fixed position since last tiled ones may be smaller
  //
  //   tiled Y, [7:0]
  //   tiled Y, [15:8]
  //   tiled Y, [23:16]
  //   tiled Y, [31:24]
  //   tiled L  [15:0]
  //   tiled R  [15:0]
  //   tiled B, [31:16], if existed
  //   tiled B, [15:0], if existed

  // Find max tiled K
  uint32_t required_size = 0;
  do {
    required_size = 0;  // Start of LMEM

    // Not split M since we don't want to reload L(weight)
    // or reload partial result of different M.
    //
    // Y(M, N) = L(M, K) * R(K, N) + B(1, N)
    // tiled_Y(M, tiled_N) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) + tiled_B(1, tiled_N)

    // tiled Y, 2 * 16bit
    tl_tiled_Y.start_address = required_size;
    tl_tiled_Y.shape = ctx.ml_default_shape(M, tiled_N, CVK_FMT_BF16);
    tl_tiled_Y.stride = ctx.ml_default_stride(tl_tiled_Y.shape, CVK_FMT_BF16, /*eu_align=*/1);
    required_size += ctx.lmem_ps32_matrix_to_size(tl_tiled_Y.shape, CVK_FMT_BF16, /*eu_align=*/1);

    // tiled L, 16bit
    tl_tiled_L.start_address = required_size;
    tl_tiled_L.shape = ctx.ml_default_shape(M, tiled_K, CVK_FMT_BF16);
    tl_tiled_L.stride = ctx.ml_default_stride(tl_tiled_L.shape, CVK_FMT_BF16, /*eu_align=*/1);
    required_size += ctx.lmem_matrix_to_size(tl_tiled_L.shape, CVK_FMT_BF16, /*eu_align=*/1);

    // tiled R, 16bit
    tl_tiled_R.start_address = required_size;
    tl_tiled_R.shape = ctx.ml_default_shape(tiled_K, tiled_N, CVK_FMT_BF16);
    tl_tiled_R.stride = ctx.ml_default_stride(tl_tiled_R.shape, CVK_FMT_BF16, /*eu_align=*/1);
    required_size += ctx.lmem_matrix_to_size(tl_tiled_R.shape, CVK_FMT_BF16, /*eu_align=*/1);

    // tiled B, 2 * 16bit
    if (have_bias) {
      tl_tiled_B.start_address = required_size;
      required_size += ctx.lmem_matrix_to_size(tl_tiled_B.shape, CVK_FMT_BF16, /*eu_align=*/1);
    }

    if (required_size <= lane_size) {
      // LLVM_DEBUG(llvm::errs() << llvm::format("  tiled_Y %d, tiled_L %d, tiled_R %d, tiled_B %d, required_size %d\n",
      //                              ctx.lmem_ps32_matrix_to_size(tl_tiled_Y.shape, CVK_FMT_BF16, /*eu_align=*/1),
      //                              ctx.lmem_matrix_to_size(tl_tiled_L.shape, CVK_FMT_BF16, /*eu_align=*/1),
      //                              ctx.lmem_matrix_to_size(tl_tiled_R.shape, CVK_FMT_BF16, /*eu_align=*/1),
      //                              ctx.lmem_matrix_to_size(tl_tiled_B.shape, CVK_FMT_BF16, /*eu_align=*/1),
      //                              required_size););

      break;
    }

  } while (--tiled_K);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "  tiled_Y(%d, %d) = tiled_L(%d, %d) * tiled_R(%d, %d) + tiled_B(%d, %d),"
                  " required_size %d kB\n",
                  M, tiled_N, M, tiled_K, tiled_K, tiled_N, 1, tiled_N, required_size / 1024););

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "  tiled_Y shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_L shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_R shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_B shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n",
                  tl_tiled_Y.shape.n, tl_tiled_Y.shape.c, tl_tiled_Y.shape.w, tl_tiled_Y.shape.col,
                  tl_tiled_Y.stride.n, tl_tiled_Y.stride.c, tl_tiled_Y.stride.h, tl_tiled_L.shape.n,
                  tl_tiled_L.shape.c, tl_tiled_L.shape.w, tl_tiled_L.shape.col, tl_tiled_L.stride.n,
                  tl_tiled_L.stride.c, tl_tiled_L.stride.h, tl_tiled_R.shape.n, tl_tiled_R.shape.c,
                  tl_tiled_R.shape.w, tl_tiled_R.shape.col, tl_tiled_R.stride.n,
                  tl_tiled_R.stride.c, tl_tiled_R.stride.h, tl_tiled_B.shape.n, tl_tiled_B.shape.c,
                  tl_tiled_B.shape.w, tl_tiled_B.shape.col, tl_tiled_B.stride.n,
                  tl_tiled_B.stride.c, tl_tiled_B.stride.h););

  assert(tiled_K);
  if (!tiled_K) {
    return;
  }

  // Each tiled_R(weight) is only loaded once.
  // tiled_L(input) reload is reload once tiled_weight moves right.
  //
  // for each tiled N
  for (uint32_t offset_N = 0; offset_N < N; offset_N += tiled_N) {
    // Y = [Y0, Y1, ... Yn-1]

    // Actual width
    uint32_t width_N = ((offset_N + tiled_N) <= N) ? tiled_N : (N - offset_N);

    // for each tiled K
    for (uint32_t offset_K = 0; offset_K < K; offset_K += tiled_K) {
      // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
      // tiled_Y(M, tiled_K) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) + tiled_B(1, tiled_N)
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
      tl_tiled_Y.shape = ctx.ml_default_shape(M, width_N, CVK_FMT_BF16);  // actual width
      tl_tiled_Y.stride = ctx.ml_default_stride(tl_tiled_Y.shape, tl_tiled_Y.fmt, 1);
      required_size += ctx.lmem_ps32_matrix_to_size(tl_tiled_Y.shape, CVK_FMT_BF16, /*eu_align=*/1);

      // Load tiled L from global memory, input
      tl_tiled_L.start_address = required_size;
      tl_tiled_L.shape = ctx.ml_default_shape(M, width_K, CVK_FMT_BF16);  // actual width
      tl_tiled_L.stride = ctx.ml_default_stride(tl_tiled_L.shape, CVK_FMT_BF16, /*eu_align=*/1);
      required_size += ctx.lmem_matrix_to_size(tl_tiled_L.shape, CVK_FMT_BF16, /*eu_align=*/1);
      ctx.tdma_load_stride_bf16(&tl_tiled_L, global_offset_bottom_data + offset_K * sizeof(uint16_t),
                               {(uint32_t)(K * sizeof(uint16_t))}  // original column width
                               );

      // Load tiled R from global memory, weight
      tl_tiled_R.start_address = required_size;
      tl_tiled_R.shape = ctx.ml_default_shape(width_K, width_N, CVK_FMT_BF16);  // actual width
      tl_tiled_R.stride = ctx.ml_default_stride(tl_tiled_R.shape, CVK_FMT_BF16, /*eu_align=*/1);
      required_size += ctx.lmem_matrix_to_size(tl_tiled_R.shape, CVK_FMT_BF16, /*eu_aligned=*/1);

      ctx.tdma_load_stride_bf16(&tl_tiled_R, global_offset_weight_data + (offset_K * N + offset_N) * sizeof(uint16_t),
                                {(uint32_t)(N * sizeof(uint16_t))} // original column width
                                );

      // Load tiled B(bias) from gobale memory at last time as H/W does
      // we need temporary shape to load uppper 16bit and lower 16bit
      bool is_last_tile = ((offset_K + tiled_K) >= K) ? true : false;
      bool B_needed = (is_last_tile && have_bias) ? true : false;
      if (B_needed) {
        tl_tiled_B.start_address = required_size;

        tl_tiled_B.shape = ctx.ml_default_shape(sizeof(uint32_t)/sizeof(uint16_t), width_N, CVK_FMT_BF16);  // 2 x 16bit, actual width
        tl_tiled_B.stride = ctx.ml_default_stride(tl_tiled_B.shape, CVK_FMT_BF16, /*eu_align=*/1);
        required_size += ctx.lmem_matrix_to_size(tl_tiled_B.shape, CVK_FMT_BF16, /*eu_aligned=*/1);
        assert(required_size <= lane_size);

        ctx.tdma_load_stride_bf16(&tl_tiled_B, global_offset_bias_data + offset_N * sizeof(uint16_t),
                                  {(uint32_t)(N * sizeof(uint16_t))} // original column width
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
      if ((ps32_mode == 0 || ps32_mode == 1) && do_activation && activation_method == RELU) {
        relu_enable = 1;
      }

      {
        cvk_tiu_matrix_multiplication_param_t p = {0};
        p.res = &tl_tiled_Y;
        p.left = &tl_tiled_L;
        p.right = &tl_tiled_R;
        p.bias = B_needed ? &tl_tiled_B : nullptr;
        p.lshift_bits = 0;                                     // deprecated
        p.rshift_bits = 0;
        p.res_is_int8 = 1;  // H/W constraint
        p.add_result = 0;   // H/W constraint
        p.relu_enable = relu_enable;
        p.ps32_mode = ps32_mode;
        p.layer_id = layer_id;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "  [offset_N=%d][offset_K=%d] L(%d, %d), R(%d, %d)\n", offset_N, offset_K,
                        p.left->shape.n, p.left->shape.col, p.right->shape.n, p.right->shape.col););

        matrix_multiplication(ctx, p);
      }

      // Store tiled_Y to global memory
      if (is_last_tile) {
        ctx.tdma_store_stride_bf16(&tl_tiled_Y, global_offset_top_data + offset_N * sizeof(uint16_t),
                                   {(uint32_t)(N*sizeof(uint16_t))}// original column width
        );
      }

    }  // for (uint32_t offset_K = 0; offset_K < K; offset_K += tiled_K)

  }    // for (uint32_t offset_N = 0; offset_N < N; offset_N += tiled_N)

}

void cvi_backend_tg_bf16_fc_kernelxxx(
    const CviBackendContext &ctx,
    uint32_t layer_id,
    gaddr_t bottom_data_gaddr,
    gaddr_t weight_data_gaddr,
    gaddr_t bias_data_gaddr,
    gaddr_t top_data_gaddr,
    int in_row,
    int in_col,
    int out_col,
    int have_bias,
    int do_activation,
    int activation_method)
{
  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tg_bf16_fc_kernel\n"
                               "    bottom_gaddr 0x%lx, weight_gaddr 0x%lx, bias_gaddr 0x%lx, top_gaddr 0x%lx\n"
                               "    in (%d, %d), out (%d)\n"
                               "    has_bias %d, do_activation %d, activation_method %d\n",
                               bottom_data_gaddr, weight_data_gaddr, bias_data_gaddr, top_data_gaddr,
                               in_row, in_col, out_col, have_bias, do_activation,
                               activation_method););

  fc_slicing_multi_dimention(ctx, layer_id, bottom_data_gaddr, weight_data_gaddr, bias_data_gaddr,
                             top_data_gaddr, in_row, in_col, out_col, have_bias, do_activation,
                             activation_method);

}
