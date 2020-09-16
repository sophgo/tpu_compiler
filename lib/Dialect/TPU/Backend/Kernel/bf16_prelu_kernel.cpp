/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_prelu_kernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <iostream>
#include "CviBackendContext.h"

#define DEBUG_TYPE "bmnet_bm1880v2_prelu"

#define ASSERT(x) assert(x)
//#define LOCAL_MEM_ADDRWIDTH (ctx.hw.local_mem_shift)
//#define LOCAL_MEM_SIZE (1 << LOCAL_MEM_ADDRWIDTH)

void cvi_backend_tg_bf16_prelu_kernel(const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_bottom,
                               gaddr_t ga_top, gaddr_t ga_negative_slope, int input_n, int input_c,
                               int input_h, int input_w) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "bmnet_cvi_backend_tg_bf16_prelu_kernel:\n"
      "    layer_id %d\n"
      "    bottom = %lx, top = %lx, negative_slope = %lx\n"
      "    nchw = (%d, %d, %d, %d)\n",
      layer_id, ga_bottom, ga_top, ga_negative_slope, input_n, input_c, input_h, input_w););
#if 0
  cvk_tl_shape_t tl_shape =
        ctx.shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t *tl_input =
      ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_t *tl_output =
      ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  // negative slope
  cvk_tl_shape_t slope_shape = {1, static_cast<uint32_t>(input_c), 1, 1};
  cvk_tl_t *slope =
      ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_BF16, /*eu_align=*/1);


  if (tl_input == nullptr)  llvm::errs() << "      unable to alloc tl_input\n";
  if (tl_output == nullptr) llvm::errs() << "      unable to alloc tl_output\n";
  if (slope == nullptr)     llvm::errs() << "      unable to alloc slope\n";

  ASSERT(tl_input && tl_output && slope && "ERROR: ctx.lmem_alloc_tensor");

  ctx.tdma_load_bf16(tl_input, ga_bottom);
  ctx.tdma_load_bf16(slope, ga_negative_slope);
#endif
  /* BF16 Condition */
  int nsecs = 1, hsecs = 1;
  uint32_t global_Nstride = static_cast<uint32_t>(input_c) * input_h * input_w;
  int blob_num = 2 * 2; /* twice as much as INT8, former 2 for input/output, buf*/
  _split_nh(ctx, input_n, input_c, input_h, input_w, blob_num,
            __get_lmem_usage(ctx, 1, input_c, 1, 1) * 2, &nsecs, &hsecs);
  LLVM_DEBUG(llvm::errs() << llvm::format("prelu inference, <%d,%d,%d,%d>, nsecs:%d, hsecs:%d\n\n",
                                          input_n, input_c, input_h, input_w, nsecs, hsecs));

  int nslice = input_n / nsecs;
  int hslice = input_h / hsecs;
  int nresidual = input_n - nslice * nsecs;
  int hresidual = input_h - hslice * hsecs;

  cvk_tl_shape_t slope_shape = ctx.shape_t4(1, static_cast<uint32_t>(input_c), 1, 1);
  cvk_tl_t *tl_slope =
      ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_BF16, 1);  // EU-aligned
  LLVM_DEBUG(
    if (tl_slope == nullptr) llvm::errs() << "      unable to alloc slope\n";
    llvm::errs() << "tl_slope->{n,c,h,w} = { " << tl_slope->stride.n << ", " << tl_slope->stride.c
               << ", " << tl_slope->stride.h << ", " << tl_slope->stride.w << " }\n";
  );

  ctx.tdma_load_bf16(tl_slope, ga_negative_slope);

  LLVM_DEBUG(
      llvm::errs() << "[ nsecs = " << nsecs << ", hsecs = " << hsecs << " ]\n";);
  for (int nidx = 0, nstart = 0; nidx < nsecs; nidx++) {
    int sec_len_n = nslice + (nidx < nresidual);
    for (int hidx = 0, hstart = 0; hidx < hsecs; hidx++) {
      int sec_len_h = hslice + (hidx < hresidual);
      // set shape
      cvk_tl_shape_t input_shape =
          ctx.shape_t4(sec_len_n, input_c, sec_len_h, input_w);
      cvk_tl_t *bottom =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);  // EU-aligned
      cvk_tl_t *neg =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);  // EU-aligned

      LLVM_DEBUG(
          if (bottom == nullptr) llvm::errs() << "      unable to alloc bottom\n";
          if (neg == nullptr) llvm::errs() << "      unable to alloc neg\n";
      );

      uint64_t offset = (nstart * global_Nstride + hstart * input_w) * sizeof(uint16_t);

      // load with stride, because needs to split h.
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

      // 1. neg = min(0, botom)
      // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_right_shift_width
      // 3. relu = relu(bottom), dirty input
      // 4. bottom = or relu, neg

      // 0. neg = min(0, botom)
      cvk_tiu_min_param_t p6 = {0};
      p6.min = neg;
      p6.a = bottom;
      p6.b_is_const = 1;
      p6.b_const.val = ctx.convert_fp32_to_bf16(0.0);
      p6.b_const.is_signed = 1;
      p6.layer_id = layer_id;
      ctx.tiu_min(&p6);

      // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_right_shift_width
      cvk_tiu_depthwise_pt_convolution_param_t param = {0};
      param.ins_h = 0;
      param.ins_last_h = 0;
      param.ins_w = 0;
      param.ins_last_w = 0;
      param.pad_top = 0;
      param.pad_bottom = 0;
      param.pad_left = 0;
      param.pad_right = 0;
      param.stride_h = 1;
      param.stride_w = 1;
      param.dilation_h = 1;
      param.dilation_w = 1;
      param.ofmap = neg;
      param.ifmap = neg;
      param.weight = tl_slope;
      param.bias = nullptr;
      param.rshift_bits = 0;
      param.relu_enable = 0;
      param.layer_id = layer_id;
      ctx.tiu_pt_depthwise_convolution(&param);

      // 3. relu = relu(bottom), dirty it
      cvk_tiu_max_param_t p13 = {0};
      p13.max = bottom;
      p13.a = bottom;
      p13.b_is_const = 1;
      p13.b_const.is_signed = 1;
      p13.b_const.val = ctx.convert_fp32_to_bf16(0.0);
      p13.layer_id = layer_id;
      ctx.tiu_max(&p13);

      // 4. bottom = or relu, neg
      cvk_tiu_add_param_t p9 = {0};
      p9.res_high = nullptr;
      p9.res_low = bottom;
      p9.a_high = nullptr;
      p9.a_low = bottom;
      p9.b_is_const = false;
      p9.b.high = nullptr;
      p9.b.low = neg;
      p9.rshift_bits = 0;
      p9.layer_id = layer_id;
      p9.relu_enable = 0;
      ctx.tiu_add(&p9);

      // move result to global
      ctx.tdma_store_stride_bf16(bottom, ga_top + offset, stride);
      // free
      ctx.lmem_free_tensor(neg);
      ctx.lmem_free_tensor(bottom);

      hstart += sec_len_h;
    }
    nstart += sec_len_n;
  }

  ctx.lmem_free_tensor(tl_slope);

// reference bf16_eltwise_sum_forward_kernel
#if 0
static void bf16_eltwise_sum_forward_kernel(const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input[],
                                            gaddr_t ga_output, int input_size, int input_n, int input_c,
                                            int input_h, int input_w, bool do_relu, float relu_slope,
                                            const float coeffs[]) {
  int blob_num = 3;
  int tensor_size = input_n * input_c * input_h * input_w;
  int total_lmem_size = LOCAL_MEM_SIZE * NPU_NUM;

  cvk_tl_shape_t tl_shape = ctx.shape_t4(input_n, input_c, input_h, input_w);
  int total_tiu_lmem_size = blob_num * ctx.lmem_tensor_to_size(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);

  llvm::errs() << llvm::format("bf16_eltwise_sum_forward_kernel:\n"
                               "    shape (%d, %d, %d, %d)\n"
                               "    input_size %d, do_relu %d, relu_slope %f\n"
                               "    total_lmem_size %d, total_tiu_lmem_size %d\n",
                               input_n, input_c, input_h, input_w,
                               input_size, do_relu, relu_slope,
                               total_lmem_size, total_tiu_lmem_size);

  gaddr_t gaddr_offset = 0;
  int remain_size = tensor_size;

  bool fused_relu = (do_relu && (relu_slope == 0.0f)) ? true : false;

  // Use all lanes
  if (remain_size >= (NPU_NUM * EU_NUM)) {
    do {
      // Find height
      int height = remain_size / (NPU_NUM * EU_NUM);
      do {
        cvk_tl_shape_t tmp_shape = ctx.shape_t4(1, NPU_NUM, height, EU_NUM);
        int required_size = blob_num * ctx.lmem_tensor_to_size(tmp_shape, CVK_FMT_BF16, /*eu_align=*/1);

        if (required_size <= LOCAL_MEM_SIZE)
          break;
      } while (--height);

      int step_size = height * NPU_NUM * EU_NUM;

      llvm::errs() << llvm::format("    step_size %d, remain_size %d, gaddr_offset 0x%lx\n",
                                            step_size, remain_size, gaddr_offset);

      elt_sum_one_step(ctx, layer_id, ga_input, ga_output, input_size, 1, NPU_NUM, height, EU_NUM,
                       coeffs, gaddr_offset, fused_relu);

      // Update step
      remain_size -= step_size;
      gaddr_offset += step_size * sizeof(uint16_t);
    } while (remain_size >= ((NPU_NUM * EU_NUM)));
  }

  // Use one lane to handle remaining
  if (remain_size) {
    int step_size = remain_size;

    elt_sum_one_step(ctx, layer_id, ga_input, ga_output, input_size, 1, 1, 1, step_size,
                     coeffs, gaddr_offset, fused_relu);

    llvm::errs() << llvm::format("    step_size %d, remain_size %d, gaddr_offset 0x%lx\n",
                                          step_size, remain_size, gaddr_offset);
    remain_size -= step_size;
    gaddr_offset += step_size * sizeof(uint16_t);
  }

  tatic void elt_sum_one_step(const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input[],
                             gaddr_t ga_output, int input_size, int input_n, int input_c,
                             int input_h, int input_w, const float coeffs[],
                             gaddr_t gaddr_offset, bool fused_relu) {
  cvk_tl_shape_t tl_shape = ctx.shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t *tl_input = ctx.(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_t *tl_output_h = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);

  llvm::errs() << "      elt_sum_one_step: (" << input_n << ", " << input_c
                           << ", " << input_h << ", " << input_w << ")\n";
  if (tl_input == nullptr) {
    llvm::errs() << "      unable to alloc tl_input\n";
  }
  if (tl_output == nullptr) {
    llvm::errs() << "      unable to alloc tl_output\n";
  }
  if (tl_output_h == nullptr) {
    llvm::errs() << "      unable to alloc tl_output_h\n";
  }

  ASSERT(tl_input && tl_output && tl_output_h);

  ctx.tdma_load_bf16(tl_input, ga_input[0] + gaddr_offset);

  cvk_tiu_mul_param_t p = {0};
  p.res_high = tl_output_h;
  p.res_low = tl_output;
  p.a = tl_input;
  p.b_const.val = ctx.convert_fp32_to_bf16(coeffs[0]);
  p.b_const.is_signed = true;
  p.b_is_const = true;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  for (int i = 1; i < input_size - 1; ++i) {
    ctx.tdma_load_bf16(tl_input, ga_input[i] + gaddr_offset);

    cvk_tiu_mac_param_t p3 = {0};
    p3.res_high = tl_output_h;
    p3.res_low = tl_output;
    p3.a = tl_input;
    p3.res_is_int8 = false;
    p3.b_const.val = ctx.convert_fp32_to_bf16(coeffs[i]);
    p3.b_is_const = 1;
    p3.b_const.is_signed = true;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    ctx.tiu_mac(&p3);
  }

  ctx.tdma_load_bf16(tl_input, ga_input[input_size - 1] + gaddr_offset);

  cvk_tiu_mac_param_t p3 = {0};
  p3.res_high = tl_output_h;
  p3.res_low = tl_output;
  p3.a = tl_input;
  p3.res_is_int8 = true;
  p3.b_const.val = ctx.convert_fp32_to_bf16(coeffs[input_size - 1]);
  p3.b_is_const = 1;
  p3.b_const.is_signed = true;
  p3.lshift_bits = 0;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = fused_relu;
  ctx.tiu_mac(&p3);

  ctx.tdma_store_bf16(tl_output, ga_output + gaddr_offset);

  // Reverse order
  ctx.lmem_free_tensor(tl_output_h);
  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}
#endif
}
