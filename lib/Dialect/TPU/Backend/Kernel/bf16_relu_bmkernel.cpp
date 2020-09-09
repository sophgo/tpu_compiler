/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_relu_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880v2_leakyrelu"

#define ASSERT(x) assert(x)
//#define LOCAL_MEM_ADDRWIDTH (ctx.hw.local_mem_shift)
//#define LOCAL_MEM_SIZE (1 << LOCAL_MEM_ADDRWIDTH)


void bf16_leakyrelu_forward_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                      gaddr_t ga_bottom, gaddr_t ga_top, float ga_negative_slope,
                                      int input_n, int input_c, int input_h, int input_w) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "bmnet_bf16_leakyrelu_forward_kernel:\n"
      "    layer_id %d\n"
      "    bottom = %lx, top = %lx, negative_slope = %f\n"
      "    nchw = (%d, %d, %d, %d)\n",
      layer_id, ga_bottom, ga_top, ga_negative_slope, input_n, input_c, input_h, input_w););

  /* BF16 Condition */
  int nsecs = 1, hsecs = 1;
  uint32_t global_Nstride = static_cast<uint32_t>(input_c) * input_h * input_w;
  int blob_num = 4; //2 buffer x pingpong(2)
  _split_nh(ctx, input_n, input_c, input_h, input_w, blob_num,
            __get_lmem_usage(ctx, 1, input_c, 1, 1) * 2, &nsecs, &hsecs);
  LLVM_DEBUG(llvm::errs() << llvm::format(
          "leakyrelu inference, <%d,%d,%d,%d>, nsecs:%d, hsecs:%d\n\n",
          input_n, input_c, input_h, input_w, nsecs, hsecs));

  int nslice = input_n / nsecs;
  int hslice = input_h / hsecs;
  int nresidual = input_n - nslice * nsecs;
  int hresidual = input_h - hslice * hsecs;

  LLVM_DEBUG(llvm::errs() << "[ nsecs = " << nsecs << ", hsecs = " << hsecs << " ]\n";);
  for (int nidx = 0, nstart = 0; nidx < nsecs; nidx++) {
    int sec_len_n = nslice + (nidx < nresidual);
    for (int hidx = 0, hstart = 0; hidx < hsecs; hidx++) {
      int sec_len_h = hslice + (hidx < hresidual);
      // set shape
      cvk_tl_shape_t input_shape =
          ctx.shape_t4(sec_len_n, input_c, sec_len_h, input_w);
      cvk_tl_t *bottom =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);  // EU-aligned
      cvk_tl_t *relu =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);  // EU-aligned

      LLVM_DEBUG(
        if (bottom == nullptr)  llvm::errs() << "      unable to alloc bottom\n";
        if (relu == nullptr)    llvm::errs() << "      unable to alloc relu\n";
      );

      uint64_t offset = (nstart * global_Nstride + hstart * input_w) * sizeof(uint16_t);

      cvk_tg_stride_t stride = {
          (uint32_t)(global_Nstride * sizeof(uint16_t)),
          (uint32_t)(input_h * input_w * sizeof(uint16_t)),
          (uint32_t)(input_w * sizeof(uint16_t))
      };
      ctx.tdma_load_stride_bf16(bottom, ga_bottom + offset, stride);

      LLVM_DEBUG(llvm::errs() << llvm::format(
          "loop, nstart:%d,hstart:%d, sec_len_n:%d,sec_len_h:%d, offset:%lu, "
          "global_Nstride:%u\n",
          nstart, hstart, sec_len_n, sec_len_h, offset, global_Nstride));

      // 0. relu = bottom * slope
      // 1. relu = max(bottom, relu)

      // 0. relu = bottom * slope
      cvk_tiu_mul_param_t p1 = {0};
      p1.res_high = nullptr; //useless
      p1.res_low = relu;
      p1.a = bottom;
      p1.b_const.val = ctx.convert_fp32_to_bf16(ga_negative_slope);
      p1.b_const.is_signed = true;
      p1.b_is_const = true;
      p1.rshift_bits = 0;
      p1.layer_id = layer_id;
      p1.relu_enable = 0;
      ctx.tiu_mul(&p1);


      // 1. relu = max(bottom, relu)
      if(ga_negative_slope <= 1) {
        cvk_tiu_max_param_t p13 = {0};
        p13.max = relu;
        p13.a = bottom;
        p13.b_is_const = 0;
        p13.b_const.is_signed = 1;
        p13.b = relu;
        p13.layer_id = layer_id;
        ctx.tiu_max(&p13);
      } else {
        cvk_tiu_min_param_t p13 = {0};
        p13.min = relu;
        p13.a = bottom;
        p13.b_is_const = 0;
        p13.b_const.is_signed = 1;
        p13.b = relu;
        p13.layer_id = layer_id;
        ctx.tiu_min(&p13);
      }
      // move result to global
      ctx.tdma_store_stride_bf16(relu, ga_top + offset, stride);

      // free
      ctx.lmem_free_tensor(relu);
      ctx.lmem_free_tensor(bottom);

      hstart += sec_len_h;
    }
    nstart += sec_len_n;
  }
}
