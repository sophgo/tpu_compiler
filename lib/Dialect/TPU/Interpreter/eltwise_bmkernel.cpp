/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
// Liang Zeng <liang.zeng@bitmain.com>
//#include <support/Debug.h>
//#include <support/Format.h>
//#include <targets/plat-bm188x/bmkernel/bmkernel_api.h>
//#include <targets/Target.hpp>
//#include <targets/plat-bm188x/BM1880v2BackendContext.hpp>
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880v2_eltwise"

#define DEBUG_BMNET(x) LLVM_DEBUG(x)
#define LOG(x) llvm::errs()
#define ASSERT(x) assert(x)

#define NODECHIP_SHIFT (ctx.hw.nodechip_shift)

static int get_csize_local(const BackendContext &ctx, int h, int w)
{
  int unit_size = ctx.hw.unit_size;
  ASSERT(unit_size > 0);
  return ALIGN(h * w, ctx.hw.eu_num) * unit_size;
}

//namespace bmnet {

// for 1880v2, max of opd0 shoud be 1024.
#define MAX_1880v2_W (1 << 10)

static int __split(const BackendContext &ctx, int blob_num, int count) {
  int slice_num = 1;
  int W_param = ctx.hw.eu_num;
  int C_param = ceiling_func_shift(count, ctx.hw.eu_shift);
  int aligned_csize = get_csize_local(ctx, 1, W_param);
  int c_per_npu = ceiling_func_shift(C_param, ctx.hw.npu_shift);
  int local_mem_usage = c_per_npu * aligned_csize * blob_num;
  const int local_mem_size = (1 << ctx.hw.local_mem_shift);
  int proportion_mem_usage = ceiling_func(local_mem_usage, local_mem_size);

  if (proportion_mem_usage == 1 && c_per_npu < 0x1000) {
    return slice_num;
  } else {
    slice_num = proportion_mem_usage;
  }

  while (true) {
    int count_slice = count / slice_num + 1;
    C_param = ceiling_func_shift(count_slice, ctx.hw.eu_shift);
    c_per_npu = ceiling_func_shift(C_param, ctx.hw.npu_shift);
    local_mem_usage = c_per_npu * aligned_csize * blob_num;
    if (local_mem_usage <= local_mem_size && c_per_npu < 0x1000) {
      return slice_num;
    } else if (slice_num < count) {
      slice_num++;
    } else {
      ASSERT(0);
    }
  }
}

void bmnet_eltwise_fixed_forward_bmkernel(const BM1880v2BackendContext &ctx, u32 stream_id,
                                          u32 inst_id, u32 layer_id, const u32 *depends,
                                          u32 depends_len, gaddr_t ga_input[], gaddr_t ga_output,
                                          int input_size, int op, int input_n, int input_c,
                                          int input_h, int input_w, bool do_relu, float relu_slope,
                                          int right_shift_width, const int *threshold_x_quantized,
                                          const int *coeffs) {
  DEBUG_BMNET(
    llvm::errs() << llvm::format(
        "   in NCHW (%d, %d, %d, %d), op %d\n",
        input_n, input_c, input_h, input_w, op);
  );

  ASSERT(input_size >= 2);

  int sec_n = ceiling_func_shift(input_n, NODECHIP_SHIFT);
  int count = ALIGN(sec_n * input_c * input_h * input_w, MAX_1880v2_W);
  int blob_num = input_size + 2;  // output may be 16bt: low and high
  int slice_num = __split(ctx, blob_num, count);
  int step = ALIGN(ceiling_func(count, slice_num), MAX_1880v2_W);
  int offset = 0;
  DEBUG_BMNET(
    llvm::errs() << llvm::format(
        "sec_n %d count %d blob_num %d slice_num %d step %d\n",
        sec_n, count, blob_num, slice_num, step);
  );

  for (int pos = 0; pos < count; pos += step) {
    int slice = math_min(count - pos, step);
    LOG(INFO) << "slice=" << slice;
    bmk1880v2_matrix_lmem_t *tl_input[2];
    bmk1880v2_matrix_lmem_shape_t matrix_shape_t1 = ctx.matrix_lmem_shape_t1(slice);

    bmk1880v2_tensor_lmem_t tl_tensor_lmem_output_h, tl_tensor_lmem_output_l, tl_tensor_lmem_input0,
        tl_tensor_lmem_input1;

    tl_input[0] = ctx.lmem_alloc_matrix(matrix_shape_t1, FMT_I8, 1);
    tl_input[1] = ctx.lmem_alloc_matrix(matrix_shape_t1, FMT_I8, 1);

    // TODO, does the input & output need to be in different memory?
    bmk1880v2_matrix_lmem_t *tl_output_l = ctx.lmem_alloc_matrix(matrix_shape_t1, FMT_I8, 1);
    bmk1880v2_matrix_lmem_t *tl_output_h = ctx.lmem_alloc_matrix(matrix_shape_t1, FMT_I8, 1);

    // convert matrix shape and stride to tensor shape and stride for mul operation
    tl_tensor_lmem_output_l.start_address = tl_output_l->start_address;
    tl_tensor_lmem_output_l.fmt = tl_output_l->fmt;
    tl_tensor_lmem_output_l.shape = {tl_output_l->shape.n, tl_output_l->shape.c,
                                     static_cast<u32>(1), tl_output_l->shape.w};
    tl_tensor_lmem_output_l.stride = {tl_output_l->stride.n, tl_output_l->stride.c,
                                      tl_output_l->stride.h, 1};

    tl_tensor_lmem_output_h.start_address = tl_output_h->start_address;
    tl_tensor_lmem_output_h.fmt = tl_output_h->fmt;
    tl_tensor_lmem_output_h.shape = {tl_output_h->shape.n, tl_output_h->shape.c,
                                     static_cast<u32>(1), tl_output_h->shape.w};
    tl_tensor_lmem_output_h.stride = {tl_output_h->stride.n, tl_output_h->stride.c,
                                      tl_output_h->stride.h, 1};

    tl_tensor_lmem_input0.start_address = tl_input[0]->start_address;
    tl_tensor_lmem_input0.fmt = tl_input[0]->fmt;
    tl_tensor_lmem_input0.shape = {tl_input[0]->shape.n, tl_input[0]->shape.c, static_cast<u32>(1),
                                   tl_input[0]->shape.w};
    tl_tensor_lmem_input0.stride = {tl_input[0]->stride.n, tl_input[0]->stride.c,
                                    tl_input[0]->stride.h, 1};

    tl_tensor_lmem_input1.start_address = tl_input[1]->start_address;
    tl_tensor_lmem_input1.fmt = tl_input[1]->fmt;
    tl_tensor_lmem_input1.shape = {tl_input[1]->shape.n, tl_input[1]->shape.c, static_cast<u32>(1),
                                   tl_input[1]->shape.w};
    tl_tensor_lmem_input1.stride = {tl_input[1]->stride.n, tl_input[1]->stride.c,
                                    tl_input[1]->stride.h, 1};

    ctx.tdma_load(tl_input[0], ga_input[0] + offset, CTRL_NEURON);

    switch (op) {
      case 0: {  // production
        // TODO yuming: not supoort op product yet
        ASSERT(0);
        // tl_load(tl_input[1], ga_input[1] + offset, CTRL_NULL);
        // tl_mul(tl_output, tl_input[0], tl_input[1], CTRL_NULL);
        // for (int i = 2; i < input_size; ++i) {
        //  bmk1682_tl_load(tl_input[1], ga_input[i] + offset, CTRL_NULL);
        //  bmk1682_tl_mul(tl_output, tl_output, tl_input[1], CTRL_NULL);
        //}
        break;
      }
      case 1: {  // sum
        bmk1880v2_tiu_element_wise_mul_param_t p;
        p.res_high = &tl_tensor_lmem_output_h;
        p.res_low = &tl_tensor_lmem_output_l;
        p.a = &tl_tensor_lmem_input0;
        p.b_val = threshold_x_quantized[0] * coeffs[0];
        p.b_is_signed = true;
        p.b_is_const = true;
        p.rshift_bits = 0;
        p.layer_id = layer_id;
        p.relu_enable = 0;
        ctx.tiu_element_wise_mul(&p);

        for (int i = 1; i < input_size - 1; ++i) {
          ctx.tdma_load(tl_input[1], ga_input[i] + offset, CTRL_NEURON);

          bmk1880v2_tiu_element_wise_mac_param_t p3;
          p3.res_high = &tl_tensor_lmem_output_h;
          p3.res_low = &tl_tensor_lmem_output_l;
          p3.a = &tl_tensor_lmem_input1;
          p3.res_is_int8 = false;
          p3.b_val = threshold_x_quantized[i] * coeffs[i];
          p3.b_is_const = 1;
          p3.b_is_signed = true;
          p3.lshift_bits = 0;
          p3.rshift_bits = 0;
          p3.layer_id = layer_id;
          p3.relu_enable = 0;
          ctx.tiu_element_wise_mac(&p3);
        }

        ctx.tdma_load(tl_input[1], ga_input[input_size - 1] + offset, CTRL_NEURON);

        bmk1880v2_tiu_element_wise_mac_param_t p3;
        p3.res_high = &tl_tensor_lmem_output_h;
        p3.res_low = &tl_tensor_lmem_output_l;
        p3.a = &tl_tensor_lmem_input1;
        p3.res_is_int8 = true;
        p3.b_val = threshold_x_quantized[input_size - 1] * coeffs[input_size - 1];
        p3.b_is_const = 1;
        p3.b_is_signed = true;
        p3.lshift_bits = 0;
        p3.rshift_bits = right_shift_width;
        p3.layer_id = layer_id;
        p3.relu_enable = 0;
        ctx.tiu_element_wise_mac(&p3);
        break;
      }
      case 2: {  // max
        // TODO yuming: not supoort op max yet
        ASSERT(0);
        // tl_load(tl_input[1], ga_input[1] + offset, CTRL_NULL);
        // tl_max(tl_output, tl_input[0], tl_input[1]);
        // for (int i = 2; i < input_size; ++i) {
        //  bmk1682_tl_load(tl_input[1], ga_input[i] + offset, CTRL_NULL);
        //  bmk1682_tl_max(tl_output, tl_output, tl_input[1]);
        //}
        break;
      }
    }

    if (do_relu) {
      if (relu_slope == 0.0f) {
        bmk1880v2_tiu_element_wise_max_param_t p13;
        p13.max = &tl_tensor_lmem_output_l;
        p13.a = &tl_tensor_lmem_output_l;
        p13.b_is_const = 1;
        p13.b_is_signed = 1;
        p13.b_val = 0;
        p13.layer_id = layer_id;
        ctx.tiu_element_wise_max(&p13);
      } else {
        ASSERT(0);  // should not reach here
        // tensor_lmem *activ = tl_input[0];
        // tensor_lmem *slope = bmk1682_tl_alloc_const(relu_slope);
        // tl_output keep elements > 0, activ keep elements < 0
        // tl_cmp(tl_output, activ, tl_output, zero, zero, tl_output);
        // using result_add
        // tl_mul(tl_output, activ, slope, CTRL_RA);
        // ctx.tl_free(slope);
      }
    }

    ctx.tdma_store(tl_output_l, ga_output + offset, CTRL_NEURON);

    ctx.lmem_free_matrix(tl_output_h);
    ctx.lmem_free_matrix(tl_output_l);
    ctx.lmem_free_matrix(tl_input[1]);
    ctx.lmem_free_matrix(tl_input[0]);

    offset += slice * INT8_SIZE;
  }
}

//}
