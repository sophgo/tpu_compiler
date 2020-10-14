/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedLrnKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tg_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>


static void find_best_slice(const CviBackendContext &ctx, int n, int c, int h,
                            int w, int blob_num, int *n_slices, int *h_slices) {
  *h_slices = 1;
  *n_slices = 1;

  uint32_t total_lmem_needs = blob_num * ctx.get_lmem_usage(n, c, h, w);
  // total_lmem_needs += 2 * 32 * 256;  // sq_lut_table and power_lut_table
  total_lmem_needs +=
      2 * NPU_NUM * 256; // sq_lut_table and power_lut_table
  if (total_lmem_needs <= (uint32_t)LOCAL_MEM_SIZE) {
    return;
  }

  // split h if lmem usage per image is larger than LOCAL_MEM_SIZE
  if (n == 1 || total_lmem_needs > (uint32_t)LOCAL_MEM_SIZE * n) {
    *n_slices = n;
    total_lmem_needs = total_lmem_needs / n;

    *h_slices = ceiling_func(total_lmem_needs, LOCAL_MEM_SIZE);
    int h_units_per_slice = ceiling_func(h, *h_slices);

    while (blob_num * ctx.get_lmem_usage(1, c, h_units_per_slice, w) >
           (uint32_t)LOCAL_MEM_SIZE) {
      *h_slices += 1;
      h_units_per_slice = ceiling_func(h, *h_slices);
    }
  } else { // split n if local memory can store more than on image
    *n_slices = ceiling_func(total_lmem_needs, (uint32_t)LOCAL_MEM_SIZE);
    int n_units_per_slice = ceiling_func(n, *n_slices);

    while (blob_num * ctx.get_lmem_usage(n_units_per_slice, c, h, w) >
           (uint32_t)LOCAL_MEM_SIZE) {
      *n_slices += 1;
      n_units_per_slice = ceiling_func(n, *n_slices);
    }
  }
}

void cvi_backend_tg_fixed_lrn_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                              uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
                              uint32_t depends_len, gaddr_t input_gaddr,
                              gaddr_t output_gaddr, gaddr_t sqr_lut_gaddr,
                              gaddr_t power_lut_gaddr, int input_n, int input_c,
                              int input_h, int input_w, int local_size,
                              int sum_right_shift_width,
                              int lrn_right_shift_width, int quant_data0,
                              int quant_data1) {
  int blob_num = 6;

  uint32_t global_Nstride = static_cast<uint32_t>(input_c * input_h * input_w);
  uint32_t global_Cstride = static_cast<uint32_t>(input_h * input_w);
  const cvk_tg_stride_t gstride = {
      global_Nstride, global_Cstride, static_cast<uint32_t>(input_w)};
  int n_slices, h_slices;
  find_best_slice(ctx, input_n, input_c, input_h, input_w, blob_num, &n_slices,
                  &h_slices);
  assert(n_slices <= input_n);
  assert(h_slices <= input_h);

  int n_units_per_slice = input_n / n_slices;
  int h_units_per_slice = input_h / h_slices;

  int move_counts = (local_size - 1) / 2;
  assert(move_counts <= input_c);

  // cvk_tl_shape_t table_shape = {1, 32, 16, 16};
  cvk_tl_shape_t table_shape = ctx.tl_shape_t4(1,NPU_NUM,16,16);

  cvk_tl_t *sqr_lut_table =
      ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);
  cvk_tl_t *power_lut_table =
      ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);

  for (int nidx = 0, nstart = 0; nidx < n_slices; nidx++) {
    int n_units = n_units_per_slice + (nidx < input_n % n_slices);
    for (int hidx = 0, hstart = 0; hidx < h_slices; hidx++) {
      int h_units = h_units_per_slice + (hidx < input_h % h_slices);
      cvk_tl_shape_t lshape = ctx.tl_shape_t4(n_units, input_c,h_units,input_w);

      cvk_tl_t *bottom =
          ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
      cvk_tl_t *sum = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
      cvk_tl_t *sum_high =
          ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
      cvk_tl_t *top = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
      cvk_tl_t *top_high =
          ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
      cvk_tl_t *top_high_high =
          ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);

      uint64_t offset = (nstart * global_Nstride + hstart * input_w) * sizeof(uint8_t);
      uint64_t slice_bottom_gaddr = input_gaddr + offset;
      uint64_t slice_top_gaddr = output_gaddr + offset;

      ctx.tdma_load_stride(bottom, slice_bottom_gaddr, gstride);

      // lut:x^2*alpha/local_size
      // move data from gmem to lmem
      ctx.tdma_load(sqr_lut_table, sqr_lut_gaddr);

      cvk_tiu_lookup_table_param_t p12 = {0};
      p12.ofmap = top;
      p12.ifmap = bottom;
      p12.table = sqr_lut_table;
      p12.layer_id = layer_id;
      ctx.tiu_lookup_table(&p12);

      cvk_tiu_copy_param_t p10 = {0};
      p10.dst = sum;
      p10.src = top;
      p10.layer_id = layer_id;
      ctx.tiu_copy(&p10);

      cvk_tiu_mul_param_t p = {0};
      p.res_high = top_high_high;
      p.res_low = sum_high;
      p.a = top;
      p.b_const.val = 0;
      p.b_const.is_signed = 0;
      p.b_is_const = 1;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      ctx.tiu_mul(&p); // sum_high initialize 0

      // use lrn shift to add different feature map data
      // sum(x^2*alpha/local_size),16bits
      for (int step = 1; step <= move_counts && step < input_c; step++) {
        cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
        lrn_shift_p.dst = top_high_high;
        lrn_shift_p.src = top;
        lrn_shift_p.right_shift = false;
        lrn_shift_p.lrn_step = step;
        ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

        cvk_tiu_mac_param_t p3 = {0};
        p3.res_high = sum_high;
        p3.res_low = sum;
        p3.res_is_int8 = 0;
        p3.a = top_high_high;
        p3.b_const.val = 1;
        p3.b_is_const = 1;
        p3.b_const.is_signed = 0;
        p3.lshift_bits = 0;
        p3.rshift_bits = 0;
        p3.layer_id = layer_id;
        p3.relu_enable = 0;
        ctx.tiu_mac(&p3);

        lrn_shift_p.dst = top_high;
        lrn_shift_p.src = top;
        lrn_shift_p.right_shift = true;
        lrn_shift_p.lrn_step = step;
        ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

        p3.res_high = sum_high;
        p3.res_low = sum;
        p3.res_is_int8 = 0;
        p3.a = top_high;
        p3.b_const.val = 1;
        p3.b_is_const = 1;
        p3.b_const.is_signed = 0;
        p3.lshift_bits = 0;
        p3.rshift_bits = 0;
        p3.layer_id = layer_id;
        p3.relu_enable = 0;
        ctx.tiu_mac(&p3);
      }
      // 16bits higher  8bits,
      p.res_high = top_high;
      p.res_low = sum_high;
      p.a = sum_high;
      p.b_const.val = quant_data0;
      p.b_const.is_signed = 0;
      p.b_is_const = 1;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      ctx.tiu_mul(&p);

      cvk_tiu_mac_param_t p3 = {0};
      p3.res_high = top_high;
      p3.res_low = sum_high;
      p3.res_is_int8 = true;
      p3.a = sum;
      p3.b_const.val = quant_data0;
      p3.b_is_const = 1;
      p3.b_const.is_signed = 0;
      p3.lshift_bits = 8;
      p3.rshift_bits = sum_right_shift_width;
      p3.layer_id = layer_id;
      p3.relu_enable = 0;
      ctx.tiu_mac(&p3);

      // scale=lut:(k+sum)^(-beta)
      // move data from gmem to lmem
      ctx.tdma_load(power_lut_table, power_lut_gaddr);

      p12.ofmap = top;
      p12.ifmap = sum_high;
      p12.table = power_lut_table;
      p12.layer_id = layer_id;
      ctx.tiu_lookup_table(&p12);

      // Y=x*scale*threshold_x_quantized[1]>>lrn_right_shift_width
      cvk_tiu_mul_param_t p1 = {0};
      p1.res_high = top_high;
      p1.res_low = top;
      p1.a = bottom;
      p1.b = top;
      p1.b_is_const = 0;
      p1.rshift_bits = 0;
      p1.layer_id = layer_id;
      p1.relu_enable = 0;
      ctx.tiu_mul(&p1);

      // 16bits higher  8bits,
      p.res_high = top_high_high;
      p.res_low = top_high;
      p.a = top_high;
      p.b_const.val = quant_data1;
      p.b_const.is_signed = 0;
      p.b_is_const = 1;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      ctx.tiu_mul(&p);

      p3.res_high = top_high_high;
      p3.res_low = top_high;
      p3.res_is_int8 = true;
      p3.a = top;
      p3.b_const.val = quant_data1;
      p3.b_is_const = 1;
      p3.b_const.is_signed = 0;
      p3.lshift_bits = 8;
      p3.rshift_bits = lrn_right_shift_width;
      p3.layer_id = layer_id;
      p3.relu_enable = 0;
      ctx.tiu_mac(&p3);

      cvk_tiu_min_param_t p7 = {0};
      p7.min = top_high;
      p7.a = top_high;
      p7.b_is_const = 1;
      p7.b_const.val = 127;
      p7.b_const.is_signed = 0;
      p7.layer_id = layer_id;
      ctx.tiu_min(&p7);

      // Original global memory shape used to calculate global stride
      // Asign global memory shape as local memory's
      ctx.tdma_store_stride(top_high, slice_top_gaddr, gstride);

      ctx.lmem_free_tensor(top_high_high);
      ctx.lmem_free_tensor(top_high);
      ctx.lmem_free_tensor(top);
      ctx.lmem_free_tensor(sum_high);
      ctx.lmem_free_tensor(sum);
      ctx.lmem_free_tensor(bottom);

      hstart += h_units;
    }

    nstart += n_units;
  }
  ctx.lmem_free_tensor(power_lut_table);
  ctx.lmem_free_tensor(sqr_lut_table);
}
