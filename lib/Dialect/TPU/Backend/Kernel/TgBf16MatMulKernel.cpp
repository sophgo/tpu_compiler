/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_fc_kernel.cpp
 * Description:
 *
 * refined 2020-10-12
 *
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bf16_fc_kernel"

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
    // LLVM_DEBUG(llvm::errs() << llvm::format("    L(%d, %d), R(%d, %d)\n",
    // p.left->shape.n,
    //                                       p.left->shape.col, p.right->shape.n,
    //                                       p.right->shape.col););
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

  LLVM_DEBUG(llvm::errs() << llvm::format("    Modified L(%d, %d), R(%d, %d)\n",
                                          p2.left->shape.n, p2.left->shape.col,
                                          p2.right->shape.n, p2.right->shape.col););

  ctx.tiu_matrix_multiplication(&p2);
}

void cvi_backend_tg_bf16_fc_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                   gaddr_t ga_left, gaddr_t ga_right, gaddr_t ga_bias,
                                   gaddr_t ga_output, int in_row, int in_col, int out_col,
                                   bool have_bias, bool do_relu,
                                   bool compressed_weight, std::vector<int> compr_weight_poss) {

  int element_size = 2;

  int m = in_row;
  int k = in_col;
  int n = out_col;

  int max_m = std::min(m, MAX_ROW);
  int max_k = std::min(k, MAX_COL);
  int max_n = std::min(n, MAX_COL);
  int step_m = max_m;
  int step_k = max_k;
  int step_n = max_n;
  uint32_t lmem_required = 0;

  cvk_ml_shape_t shape_l, shape_r, shape_b, shape_y;
  for (step_k = max_k; step_k > 0; --step_k) {
    for (step_n = max_n; step_n > 0; --step_n) {
      for (step_m = max_m; step_m > 0; --step_m) {
        shape_l = ctx.ml_default_shape(step_m, step_k, CVK_FMT_BF16);
        shape_r = ctx.ml_default_shape(step_k, step_n, CVK_FMT_BF16);
        shape_y = ctx.ml_default_shape(step_m, step_n, CVK_FMT_BF16);
        lmem_required = ctx.lmem_matrix_to_size(shape_l, CVK_FMT_BF16, 1) +
                        ctx.lmem_matrix_to_size(shape_r, CVK_FMT_BF16, 1);
        if (have_bias) {
          shape_b = ctx.ml_default_shape(2, step_n, CVK_FMT_BF16);
          lmem_required += ctx.lmem_matrix_to_size(shape_b, CVK_FMT_BF16, 1);
        }
        if (step_k != k) {
          lmem_required += ctx.lmem_ps32_matrix_to_size(shape_y, CVK_FMT_BF16, 1);
        } else {
          lmem_required += ctx.lmem_matrix_to_size(shape_y, CVK_FMT_BF16, 1);
        }
        if (lmem_required <= (uint32_t)LOCAL_MEM_SIZE) {
          goto after_loop;
        }
      }
    }
  }

after_loop:
  if (lmem_required > (uint32_t)LOCAL_MEM_SIZE) {
    llvm::errs() << llvm::format(
        "matmul, failed, m:%d, k:%d, n:%d, step:%d,%d,%d, lmem:%d, bias:%d, relu:%d\n", m,
        k, n, step_m, step_k, step_n, (int)lmem_required, have_bias, do_relu);
    assert(0);
  }

  llvm::errs() << llvm::format(
      "matmul, m:%d, k:%d, n:%d, step:%d,%d,%d, lmem:%d, bias:%d, relu:%d\n", m, k, n,
      step_m, step_k, step_n, (int)lmem_required, have_bias, do_relu);

  cvk_ml_t *tl_l, *tl_r, *tl_y;
  cvk_ml_t *tl_b = nullptr;
  tl_l = ctx.lmem_alloc_matrix(shape_l, CVK_FMT_BF16, 1);
  tl_r = ctx.lmem_alloc_matrix(shape_r, CVK_FMT_BF16, 1);
  tl_y = (step_k != k) ? ctx.lmem_alloc_ps32_matrix(shape_y, CVK_FMT_BF16, 1)
                       : ctx.lmem_alloc_matrix(shape_y, CVK_FMT_BF16, 1);
  if (have_bias) {
    tl_b = ctx.lmem_alloc_matrix(shape_b, CVK_FMT_BF16, 1);
  }

  uint64_t offset = 0;
  cvk_mg_stride_t gstride_l = {(uint32_t)(k * element_size)};
  cvk_mg_stride_t gstride_r = {(uint32_t)(n * element_size)};
  cvk_mg_stride_t gstride_y = {(uint32_t)(n * element_size)};

  if (step_k == k) {
    for (int pos_n = 0; pos_n < n; pos_n += step_n) {
      int cur_n = std::min(n - pos_n, step_n);

      cvk_ml_t opd_r, opd_b;
      shape_r = ctx.ml_default_shape(k, cur_n, CVK_FMT_BF16);
      opd_r.start_address = tl_r->start_address;
      opd_r.shape = shape_r;
      opd_r.fmt = CVK_FMT_BF16;
      opd_r.stride = ctx.ml_default_stride(shape_r, CVK_FMT_BF16, 1);
      offset = pos_n * element_size;
      if (!compressed_weight) {
        ctx.tdma_load_stride(&opd_r, ga_right + offset, gstride_r);
      } else {
        int cmpr_wgt_idx =  pos_n / step_n;
        cvi_backend_ml_load_stride(ctx,
                                   layer_id,
                                   ga_right + compr_weight_poss[cmpr_wgt_idx],
                                   tl_r->start_address,
                                   k, cur_n,
                                   cur_n,
                                   false, // DoTranspose
                                   true,  // DoAligned
                                   CVK_FMT_BF16,
                                   CVK_FMT_BF16,
                                   true   // DoDecompress
                                   );
      }

      if (have_bias) {
        shape_b = ctx.ml_default_shape(2, cur_n, CVK_FMT_BF16);
        opd_b.start_address = tl_b->start_address;
        opd_b.shape = shape_b;
        opd_b.fmt = CVK_FMT_BF16;
        opd_b.stride = ctx.ml_default_stride(shape_b, CVK_FMT_BF16, 1);
        offset = pos_n * element_size;
        ctx.tdma_load_stride(&opd_b, ga_bias + offset, gstride_r);
      }

      for (int pos_m = 0; pos_m < m; pos_m += step_m) {
        int cur_m = std::min(m - pos_m, step_m);

        cvk_ml_t opd_l;
        shape_l = ctx.ml_default_shape(cur_m, k, CVK_FMT_BF16);
        opd_l.start_address = tl_l->start_address;
        opd_l.shape = shape_l;
        opd_l.fmt = CVK_FMT_BF16;
        opd_l.stride = ctx.ml_default_stride(shape_l, CVK_FMT_BF16, 1);

        offset = pos_m * k * element_size;
        ctx.tdma_load_stride(&opd_l, ga_left + offset, gstride_l);

        cvk_ml_t opd_y;
        shape_y = ctx.ml_default_shape(cur_m, cur_n, CVK_FMT_BF16);
        opd_y.start_address = tl_y->start_address;
        opd_y.shape = shape_y;
        opd_y.fmt = CVK_FMT_BF16;
        opd_y.stride = ctx.ml_default_stride(shape_y, CVK_FMT_BF16, 1);

        cvk_tiu_matrix_multiplication_param_t p = {0};
        p.res = &opd_y;
        p.left = &opd_l;
        p.right = &opd_r;
        p.bias = have_bias ? &opd_b : nullptr;
        p.lshift_bits = 0; // deprecated
        p.rshift_bits = 0;
        p.res_is_int8 = 1; // H/W constraint
        p.add_result = 0;  // H/W constraint
        p.relu_enable = do_relu;
        p.ps32_mode = 0;
        p.layer_id = layer_id;

        matrix_multiplication(ctx, p);

        // store
        offset = (pos_n + pos_m * n) * element_size;
        ctx.tdma_store_stride(&opd_y, ga_output + offset, gstride_y);
      }
    }
  } else {

    for (int pos_n = 0; pos_n < n; pos_n += step_n) {
      int cur_n = std::min(n - pos_n, step_n);
      for (int pos_m = 0; pos_m < m; pos_m += step_m) {
        int cur_m = std::min(m - pos_m, step_m);
        for (int pos_k = 0; pos_k < k; pos_k += step_k) {
          int cur_k = std::min(k - pos_k, step_k);

          cvk_ml_t opd_r, opd_b;
          shape_r = ctx.ml_default_shape(cur_k, cur_n, CVK_FMT_BF16);
          opd_r.start_address = tl_r->start_address;
          opd_r.shape = shape_r;
          opd_r.fmt = CVK_FMT_BF16;
          opd_r.stride = ctx.ml_default_stride(shape_r, CVK_FMT_BF16, 1);
          offset = (pos_n + pos_k * n) * element_size;
          if (!compressed_weight) {
            ctx.tdma_load_stride(&opd_r, ga_right + offset, gstride_r);
          } else {
            // column major
            const int num_k = llvm::divideCeil(k, step_k);
            int cmpr_wgt_idx =  (pos_n / step_n) * num_k + (pos_k / step_k);
            cvi_backend_ml_load_stride(ctx,
                                       layer_id,
                                       ga_right + compr_weight_poss[cmpr_wgt_idx],
                                       tl_r->start_address,
                                       cur_k, cur_n,
                                       cur_n,
                                       false, // DoTranspose
                                       true,  // DoAligned
                                       CVK_FMT_BF16,
                                       CVK_FMT_BF16,
                                       true   // DoDecompress
                                      );
          }

          cvk_ml_t opd_l;
          shape_l = ctx.ml_default_shape(cur_m, cur_k, CVK_FMT_BF16);
          opd_l.start_address = tl_l->start_address;
          opd_l.shape = shape_l;
          opd_l.fmt = CVK_FMT_BF16;
          opd_l.stride = ctx.ml_default_stride(shape_l, CVK_FMT_BF16, 1);

          offset = (pos_k + pos_m * k) * element_size;
          ctx.tdma_load_stride(&opd_l, ga_left + offset, gstride_l);

          cvk_ml_t opd_y;
          shape_y = ctx.ml_default_shape(cur_m, cur_n, CVK_FMT_BF16);
          opd_y.start_address = tl_y->start_address;
          opd_y.shape = shape_y;
          opd_y.fmt = CVK_FMT_BF16;
          opd_y.stride = ctx.ml_default_stride(shape_y, CVK_FMT_BF16, 1);

          bool is_last_k_tile = (pos_k + step_k >= k) ? true : false;
          bool add_bias = (is_last_k_tile && have_bias) ? true : false;
          if (add_bias) {
            shape_b = ctx.ml_default_shape(2, cur_n, CVK_FMT_BF16);
            opd_b.start_address = tl_b->start_address;
            opd_b.shape = shape_b;
            opd_b.fmt = CVK_FMT_BF16;
            opd_b.stride = ctx.ml_default_stride(shape_b, CVK_FMT_BF16, 1);
            offset = pos_n * element_size;
            ctx.tdma_load_stride(&opd_b, ga_bias + offset, gstride_r);
          }

          int ps32_mode = 0;           // normal mode
          if (pos_k == 0) {            // first tile
            ps32_mode = 2;             // write 32b result at the first time
          } else if (is_last_k_tile) { // last tile
            ps32_mode = 1;             // load previous 32-bit result
          } else {
            ps32_mode = 3; // init & write 32bits partial sum
          }

          bool _do_relu = false;
          if ((ps32_mode == 0 || ps32_mode == 1) && do_relu) {
            _do_relu = true;
          }

          cvk_tiu_matrix_multiplication_param_t p = {0};
          p.res = &opd_y;
          p.left = &opd_l;
          p.right = &opd_r;
          p.bias = add_bias ? &opd_b : nullptr;
          p.lshift_bits = 0; // deprecated
          p.rshift_bits = 0;
          p.res_is_int8 = 1; // H/W constraint
          p.add_result = 0;  // H/W constraint
          p.relu_enable = _do_relu;
          p.ps32_mode = ps32_mode;
          p.layer_id = layer_id;

          matrix_multiplication(ctx, p);

          // store
          if (is_last_k_tile) {
            offset = (pos_n + pos_m * n) * element_size;
            ctx.tdma_store_stride(&opd_y, ga_output + offset, gstride_y);
          }
        }
      }
    }
  }

  if (have_bias)
    ctx.lmem_free_matrix(tl_b);
  ctx.lmem_free_matrix(tl_y);
  ctx.lmem_free_matrix(tl_r);
  ctx.lmem_free_matrix(tl_l);
}
