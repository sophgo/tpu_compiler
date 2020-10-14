/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReluKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <map>
#include <vector>

#define DEBUG_TYPE "bm1880v2_kernel_relu"
#define ASSERT(x) assert(x)

#define RELU   1
#define PRELU  2
//namespace {

//using bmnet::CviBackendContext;

//}  // namespace

//namespace bmnet {

void cvi_backend_tg_fixed_relu_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                       uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
                                       uint32_t depends_len, uint64_t bottom_gaddr, uint64_t top_gaddr,
                                       float negative_slope, int input_n, int input_c, int input_h,
                                       int input_w, int threshold_x_quantized_len,
                                       const int *threshold_x_quantized,
                                       const int *right_shift_array,
                                       cvk_fmt_t fmt) {
  // for (int i = 0; i < threshold_x_quantized_len; i++) {
  //  VLOG(3) << "threshold_x_quantized/right_shift_array[" << i << "]:" << threshold_x_quantized[i]
  //          << "/" << right_shift_array[i];
  //}
  // TODO: ParallelBanks
  //_forward_internel(ctx, layer_id, bottom_gaddr, top_gaddr, negative_slope, input_n, input_c,
  //                  input_h, input_w, threshold_x_quantized_len, threshold_x_quantized,
  //                  right_shift_array);
  // TODO: slicing
  // TODO: not use much if/else condition for uint8_t/bf16
  // TODO: implement tl
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 0;
  int blob_num = 2; // 3 means we allocate input / output

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, (cvk_fmt_t)fmt, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n, c, h, w);

    // load input
    cvk_tl_t *tl_input;

    if ((cvk_fmt_t)fmt == CVK_FMT_BF16) {
      tl_input = ctx.lmem_alloc_tensor(tl_shape, (cvk_fmt_t)fmt, /*eu_align=*/1);
      ctx.tdma_load_bf16(tl_input, bottom_gaddr + gaddr_offset);
    }
    else {
      tl_input = ctx.lmem_alloc_tensor(tl_shape, (cvk_fmt_t)fmt, /*eu_align=*/1);
      ctx.tdma_load(tl_input, bottom_gaddr + gaddr_offset);
    }

    // 0. top = relu(bottom)
    cvk_tiu_max_param_t p13 = {0};
    p13.max = tl_input;
    p13.a = tl_input;
    p13.b_is_const = 1;
    p13.layer_id = layer_id;

    if (fmt == CVK_FMT_BF16) {
      p13.b_const.val = ctx.convert_fp32_to_bf16(0);
    }
    else {
      p13.b_const.val = (0);
      if (fmt == CVK_FMT_I8) {
        p13.b_const.is_signed = 1;
      }
      else if (fmt == CVK_FMT_U8) {
        p13.b_const.is_signed = 0;
      }
      else {
        // only support fmt = CVK_FMT_I8/CVK_FMT_U8
        ASSERT(0);
      }
    }


    if (fmt == CVK_FMT_BF16) {
      ctx.tiu_max(&p13);
      ctx.tdma_store_bf16(tl_input, top_gaddr + gaddr_offset);
    }
    else {
      ctx.tiu_max(&p13);
      ctx.tdma_store(tl_input, top_gaddr + gaddr_offset);
    }

    ctx.lmem_free_tensor(tl_input);

    // TODO checke tfma/tiu pipeline
    // store
    //cvi_backend_tl_store_tensor(ctx, layer_id, tl_ofmap, top_gaddr + gaddr_offset, /*eu_align=*/1);
  }
}

void cvi_backend_tg_fixed_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, uint64_t bottom_gaddr,
    uint64_t top_gaddr, uint64_t negative_scope_gaddr, int input_n, int input_c,
    int input_h, int input_w, int threshold_x_quantized_len,
    const int *threshold_x_quantized, const int *right_shift_array, cvk_fmt_t fmt) {


  #if 0
  cvk_tl_shape_t tl_shape =
      ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t *tl_input =
      ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_t *tl_output =
      ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
  ctx.tdma_load_bf16(tl_input, bottom_gaddr);

  // slope
  cvk_tl_shape_t slope_shape = {1, static_cast<uint32_t>(input_c), 1,
                                               1};
  cvk_tl_t *slope =
      ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_BF16, /*eu_align=*/1);
  ctx.tdma_load_bf16(slope, negative_scope_gaddr);
  #endif

  cvk_tl_shape_t tl_shape;
  cvk_tl_t *tl_input;
  cvk_tl_t *tl_output;
  cvk_tl_shape_t slope_shape;
  cvk_tl_t *slope;

  if (fmt == CVK_FMT_I8){
    tl_shape =
        ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
    tl_input =
        ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    tl_output =
        ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    ctx.tdma_load(tl_input, bottom_gaddr);
    // slope
    slope_shape = {1, static_cast<uint32_t>(input_c), 1, 1};
    slope =
        ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_I8, /*eu_align=*/1);
    ctx.tdma_load(slope, negative_scope_gaddr);
  } else if (fmt == CVK_FMT_BF16){
    tl_shape =
      ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
    tl_input =
        ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
    tl_output =
        ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
    ctx.tdma_load_bf16(tl_input, bottom_gaddr);
    // slope
    slope_shape = {1, static_cast<uint32_t>(input_c), 1, 1};
    slope =
        ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_BF16, /*eu_align=*/1);
    ctx.tdma_load_bf16(slope, negative_scope_gaddr);
  } else {
    assert(0 && "Not Support Format Type.");
  }

  #if 0
  // 0. top = relu(bottom)
  cvk_tiu_max_param_t p13 = {0};
  p13.max = tl_output;
  p13.a = tl_input;
  p13.b_is_const = 1;
  p13.b_const.val = ctx.convert_fp32_to_bf16(0);
  p13.layer_id = layer_id;
  ctx.tiu_max(&p13);
  #endif

  // 0. top = relu(bottom)
  cvk_tiu_max_param_t p13 = {0};
  p13.max = tl_output;
  p13.a = tl_input;
  p13.b_is_const = 1;
  p13.b_const.val = 0;
  p13.layer_id = layer_id;
  ctx.tiu_max(&p13);

  // 1. top = (top * gt_scale) >> right_shift_array
  // cvk_tiu_mul_param_t p = {0};
  // p.res_high = nullptr;
  // p.res_low = tl_output;
  // p.a = tl_output;
  // p.b_const.val = 0;
  // p.b_is_const = 1;
  // p.rshift_bits = 0;
  // p.layer_id = layer_id;
  // p.relu_enable = 0;
  // ctx.tiu_mul(&p);

  // 2. bottom = neg(0, bottom)
  cvk_tiu_min_param_t p7 = {0};
  p7.min = tl_input;
  p7.a = tl_input;
  p7.b_is_const = 1;
  p7.b_const.val = ctx.convert_fp32_to_bf16(0);
  p7.layer_id = layer_id;
  ctx.tiu_min(&p7);

  // 3. bottom (n,c,h,w) = (bottom(n,c,h,w) * slope(1,c,1,1)) >>
  // le_right_shift_width
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
  param.ofmap = tl_input;
  param.ifmap = tl_input;
  param.weight = slope;
  param.bias = nullptr;
  param.rshift_bits = 0;
  param.relu_enable = 0;
  param.layer_id = layer_id;
  ctx.tiu_pt_depthwise_convolution(&param);

  //4. top = or(top, bottom)
  cvk_tiu_add_param_t p9 = {0};
  p9.res_low = tl_output;
  p9.res_high=0;
  p9.a_low = tl_output;
  p9.a_high = 0;
  p9.b.low = tl_input;
  p9.b.high = 0;
  p9.rshift_bits = 0;
  p9.b_is_const=0;
  ctx.tiu_add(&p9);
  ctx.tdma_store_bf16(tl_output, top_gaddr);
  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}

void cvi_backend_tg_fixed_prelu_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, uint64_t bottom_gaddr,
    uint64_t top_gaddr, uint64_t negative_scope_gaddr, int input_n, int input_c,
    int input_h, int input_w,
    int GT_right_shift_width,int GT_scale,
    int LE_right_shift_width, cvk_fmt_t fmt) {

  int nsecs = 1, hsecs = 1;
  cvk_tg_stride_t gstride =
      ctx.tg_default_stride({
          static_cast<uint32_t>(input_n),
          static_cast<uint32_t>(input_c),
          static_cast<uint32_t>(input_h),
          static_cast<uint32_t>(input_w)},
          CVK_FMT_I8);

  cvk_tl_shape_t slope_shape = ctx.tl_shape_t4(1, input_c, 1, 1);
  cvk_tl_t *tl_slope =
      ctx.lmem_alloc_tensor(slope_shape, CVK_FMT_I8, 1);  // EU-aligned
  ctx.tdma_load(tl_slope, negative_scope_gaddr);

  uint32_t reserved = ctx.lmem_tensor_to_size(slope_shape, CVK_FMT_I8, 0);
  reserved = align_up(reserved, EU_NUM);
  ctx.split_nh(input_n, input_c, input_h, input_w, 3, reserved, &nsecs, &hsecs);
  LLVM_DEBUG(llvm::errs() << llvm::format(
          "prelu inference, <%d,%d,%d,%d>, nsecs:%d, hsecs:%d\n\n",
          input_n, input_c, input_h, input_w, nsecs, hsecs););

  int n_step = ceiling_func(input_n, nsecs);
  int h_step = ceiling_func(input_h, hsecs);

  for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
    int cur_n = std::min(n_step, input_n - n_pos);
    for (int h_pos = 0; h_pos < input_h; h_pos += h_step) {
      int cur_h = std::min(h_step, input_h - h_pos);
      // set shape
      cvk_tl_shape_t input_shape =
          ctx.tl_shape_t4(cur_n, input_c, cur_h, input_w);
      cvk_tl_t *bottom =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);  // EU-aligned
      cvk_tl_t *relu =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);  // EU-aligned
      cvk_tl_t *neg =
          ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);  // EU-aligned
      assert(bottom && relu && neg);
      // cvk_tl_t *zero =
      //     ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);  // EU-aligned

      // cvk_tdma_g2l_tensor_fill_constant_param_t p1 = {0};
      // p1.constant = 0;
      // p1.dst = zero;
      // ctx.tdma_tg2l_tensor_fill_constant(&p1);

      uint64_t offset = n_pos * gstride.n + h_pos * input_w;

      // load with stride, because needs to split h.
      ctx.tdma_load_stride(bottom, bottom_gaddr + offset, gstride);
      LLVM_DEBUG(llvm::errs() << llvm::format(
          "loop, n_pos:%d,h_pos:%d, cur_n:%d,cur_h:%d, offset:%lu, "
          "global_Nstride:%u\n",
          n_pos, h_pos, cur_n, cur_h, offset, gstride.n););


      // 0. relu = relu(bottom)
      // 1. relu = (relu * GT_scale) >> GT_right_shift_width
      // 2. neg = neg(0, botom)
      // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_right_shift_width
      // 4. bottom = or relu, neg

      // 0. relu = relu(bottom)
      cvk_tiu_max_param_t p13 = {0};
      p13.max = relu;
      p13.a = bottom;
      p13.b_is_const = 1;
      p13.b_const.is_signed = 1;
      p13.b_const.val = 0;
      p13.layer_id = layer_id;
      ctx.tiu_max(&p13);

      // 1. relu = (relu * GT_scale) >> GT_right_shift_width
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = relu;
      p.a = relu;
      p.b_const.val = GT_scale;
      p.b_const.is_signed = true;
      p.b_is_const = 1;
      p.rshift_bits = GT_right_shift_width;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      ctx.tiu_mul(&p);

      // 2. neg = neg(0, botom)
      cvk_tiu_min_param_t p6 = {0};
      p6.min = neg;
      p6.a = bottom;
      p6.b_is_const = 1;
      p6.b_const.val = 0;
      p6.b_const.is_signed = 1;
      p6.layer_id = layer_id;
      ctx.tiu_min(&p6);

      // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_right_shift_width
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
      param.rshift_bits = LE_right_shift_width;
      param.relu_enable = 0;
      param.layer_id = layer_id;
      ctx.tiu_pt_depthwise_convolution(&param);

      // 4. bottom = or relu, neg
      cvk_tiu_or_int8_param_t p9 = {0};
      p9.res = bottom;
      p9.a = relu;
      p9.b = neg;
      p9.layer_id = layer_id;
      ctx.tiu_or_int8(&p9);

      // move result to global
      ctx.tdma_store_stride(bottom, top_gaddr + offset, gstride);

      // free
      ctx.lmem_free_tensor(neg);
      ctx.lmem_free_tensor(relu);
      ctx.lmem_free_tensor(bottom);
    }
  }

  ctx.lmem_free_tensor(tl_slope);
}


#if 1
//
// Output: bottom
//
static void tl_leaky_relu(const CviBackendContext &ctx, uint32_t layer_id,
                          cvk_tl_t &bottom, cvk_tl_t &relu,
                          cvk_tl_t &neg, int GT_right_shift_width,
                          int LE_right_shift_width, int GT_scale, int LE_scale) {
  bool isIgnorePosPart = (GT_scale == 0);
  bool isSlopeSmallerThanOne = ((LE_scale >> LE_right_shift_width) == 0);

  if(isIgnorePosPart) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &relu;
    p4.a = &bottom;
    p4.b_const.val = LE_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = LE_right_shift_width;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    ctx.tiu_mul(&p4);

    if(isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p1 = {0};
      p1.max = &bottom;
      p1.a = &bottom;
      p1.b = &relu;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      ctx.tiu_max(&p1);
    } else {
      cvk_tiu_min_param_t p1 = {0};
      p1.min = &bottom;
      p1.a = &bottom;
      p1.b = &relu;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      ctx.tiu_min(&p1);
    }
  } else {
    // 0. relu = relu(bottom)
    cvk_tiu_max_param_t p13 = {0};
    p13.max = &relu;
    p13.a = &bottom;
    p13.b_is_const = 1;
    p13.b_const.is_signed = 1;
    p13.b_const.val = 0;
    p13.layer_id = layer_id;
    ctx.tiu_max(&p13);

    // 1. relu = (relu * GT_scale) >> GT_right_shift_width
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &relu;
    p.a = &relu;
    p.b_const.val = GT_scale;
    p.b_const.is_signed = true;
    p.b_is_const = 1;
    p.rshift_bits = GT_right_shift_width;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    // 2. neg = neg(0, botom)
    cvk_tiu_min_param_t p7 = {0};
    p7.min = &neg;
    p7.a = &bottom;
    p7.b_is_const = 1;
    p7.b_const.val = 0;
    p7.b_const.is_signed = 1;
    p7.layer_id = layer_id;
    ctx.tiu_min(&p7);

    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope) >> LE_right_shift_width
    cvk_tiu_mul_param_t p8 = {0};
    p8.res_high = nullptr;
    p8.res_low = &neg;
    p8.a = &neg;
    p8.b_const.val = LE_scale;
    p8.b_const.is_signed = true;
    p8.b_is_const = 1;
    p8.rshift_bits = LE_right_shift_width;
    p8.layer_id = layer_id;
    p8.relu_enable = 0;
    ctx.tiu_mul(&p8);

    // 4. bottom = or relu, neg
    cvk_tiu_or_int8_param_t p9 = {0};
    p9.res = &bottom;
    p9.a = &relu;
    p9.b = &neg;
    p9.layer_id = layer_id;
    ctx.tiu_or_int8(&p9);
  }
}

void cvi_backend_tg_fixed_leakyrelu_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, uint64_t input_gaddr, uint64_t output_gaddr, int input_n, int input_c, int input_h,
    int input_w, int GT_right_shift_width, int LE_right_shift_width, int GT_scale, int LE_scale,
    int threshold_x_quantized_len, const int *threshold_x_quantized, const int *right_shift_array) {
  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tg_fixed_leakyrelu_kernel:\n"
                                        "  layer_id %d\n"
                                        "  input_gddr: %lx, output_gaddr: %lx\n"
                                        "  input (%d, %d, %d, %d)\n"
                                        "  GT_scale:%d, LE_scale:%d\n"
                                        "  GT_right_shift_width:%d, LE_right_shift_width:%d\n",
                                        layer_id, input_gaddr, output_gaddr, input_n, input_c,
                                        input_h, input_w, GT_scale, LE_scale, GT_right_shift_width,
                                        LE_right_shift_width););

  for (int i = 0; i < threshold_x_quantized_len; i++) {
    LLVM_DEBUG(llvm::errs() << "threshold_x_quantized/right_shift_array[" << i << "]:" << threshold_x_quantized[i]
            << "/" << right_shift_array[i];);
  }

  // Split input based on local memory
  uint32_t total_eu = NPU_NUM * EU_NUM;
  uint32_t lane_size = LOCAL_MEM_SIZE;
  uint32_t max_N = (1 << 12) - 1;  // 1880v2: 12 bit
  uint32_t max_W = (1 << 12) - 1;  // 1880v2: 12 bit
  uint32_t count = input_n * input_c * input_h * input_w;
  uint32_t tiled_N = count / total_eu / 3;  // 3 blobs
  tiled_N = (tiled_N > max_N) ? max_N : tiled_N;

  // local tensor shape(tiled_N, npu_num, 1, eu_num)
  cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(tiled_N, NPU_NUM, 1, EU_NUM);
  cvk_tl_stride_t tl_stride =
      ctx.tl_default_stride(tl_shape, CVK_FMT_I8, /*eu_align=*/1);

  // Find max tiled_N
  uint32_t required_size = 0;
  do {
    tl_shape.n = tiled_N;
    tl_stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    required_size = 3 * tl_shape.n * tl_stride.n;  // 3 blobs

    if (required_size <= lane_size) {
      break;
    }

  } while (--tiled_N);

  LLVM_DEBUG(
      llvm::errs() << llvm::format("  tiled_bottom shape (%d, %d, %d, %d), stride (%d, %d, %d, %d)\n"
                                "  required_size %d kB/lane\n",
                                tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w, tl_stride.n,
                                tl_stride.c, tl_stride.h, tl_stride.w, required_size / 1024););

  ASSERT(tiled_N);
  if (!tiled_N) {
    return;
  }

  // Tiled local memory layout:
  //   tiled bottom/result
  //   tiled relu
  //   tiled neg

  // Tiled bottom
  required_size /= 3;  // for 3 blobs
  cvk_tl_t tl_tiled_bottom;
  tl_tiled_bottom.start_address = 0;
  tl_tiled_bottom.fmt = CVK_FMT_I8;
  tl_tiled_bottom.shape = tl_shape;
  tl_tiled_bottom.stride = tl_stride;

  // Tiled relu
  cvk_tl_t tl_tiled_relu = tl_tiled_bottom;
  tl_tiled_relu.start_address = tl_tiled_bottom.start_address + required_size;

  // Tiled neg
  cvk_tl_t tl_tiled_neg = tl_tiled_bottom;
  tl_tiled_neg.start_address = tl_tiled_relu.start_address + required_size;

  // In unit of tiled_N * npu_num * eu_num
  uint32_t global_input_offset = 0;
  for (uint32_t i = 0; i < (count / total_eu / tiled_N); i++) {
    // Load as a chunk of contiguous memory in global memory, not use global shape/stride
    // Local memory use tensor shape to maximize eu utilization.

    LLVM_DEBUG(llvm::errs() << llvm::format("  [%d] tdma load: gaddr 0x%x+0x%x\n", i, input_gaddr,
                                          global_input_offset););

    ctx.tdma_load(&tl_tiled_bottom, input_gaddr + global_input_offset);

    tl_leaky_relu(ctx, layer_id, tl_tiled_bottom, tl_tiled_relu, tl_tiled_neg, GT_right_shift_width,
                  LE_right_shift_width, GT_scale, LE_scale);

    // Store bottom as a chunk of contiguous memory, not use global shape/stride
    ctx.tdma_store(&tl_tiled_bottom, output_gaddr + global_input_offset);

    // Next input offset
    global_input_offset += tiled_N * total_eu;

  }  // for (uint32_t i = 0; i < (count/total_eu/tiled_N); i++)

  // Remaining count, in unit of npu_num * eu_num
  if (global_input_offset < count) {
    uint32_t tiled_W = (count - global_input_offset) / NPU_NUM;
    tiled_N = 1;
    do {
      tl_shape.n = tiled_N;
      tl_shape.w = tiled_W;
      tl_stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
      required_size = 3 * tl_shape.n * tl_stride.n;  // 3 blobs

      if (required_size <= lane_size && (tiled_W <=  max_W)) {
        break;
      } else {
        tiled_W/=2;
        tiled_N*=2;
      }
    } while (true);// Magic number for 2^12 -1 - 32

    if((count - global_input_offset) % NPU_NUM != 0) {
      ASSERT(0 && "Remaining size should align npu_num, or die");
    }


    // Update shape, stride
    tl_shape.n = tiled_N;
    tl_shape.w = tiled_W;
    tl_stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    required_size = tl_shape.n * tl_stride.n;

    LLVM_DEBUG(
        llvm::errs() << llvm::format("  tiled_bottom shape (%d, %d, %d, %d), stride (%d, %d, %d, %d)\n"
                                  "  required_size %d kB/lane\n",
                                  tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w, tl_stride.n,
                                  tl_stride.c, tl_stride.h, tl_stride.w, required_size / 1024););

    // Tiled bottom
    tl_tiled_bottom.shape = tl_shape;
    tl_tiled_bottom.stride = tl_stride;

    // Tiled bottom with precise stride
    cvk_tl_t tl_tiled_bottom_precise_stride = tl_tiled_bottom;
    tl_tiled_bottom_precise_stride.stride = {
        (uint32_t)(tl_shape.h * tl_shape.w * sizeof(uint8_t)),
        (uint32_t)(tl_shape.h * tl_shape.w * sizeof(uint8_t)),
        (uint32_t)(tl_shape.w * sizeof(uint8_t)),
        (uint32_t)(sizeof(uint8_t))};

    LLVM_DEBUG(
        llvm::errs() << llvm::format("  tiled_bottom_precise shape (%d, %d, %d, %d), stride (%d, %d, %d, %d)\n"
                              "  required_size %d kB/lane\n",
                              tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w, tl_tiled_bottom_precise_stride.stride.n,
                              tl_tiled_bottom_precise_stride.stride.c, tl_tiled_bottom_precise_stride.stride.h, tl_tiled_bottom_precise_stride.stride.w, required_size / 1024););

    // Tiled relu
    tl_tiled_relu = tl_tiled_bottom;
    tl_tiled_relu.start_address = tl_tiled_bottom.start_address + required_size;

    // Tiled neg
    tl_tiled_neg = tl_tiled_bottom;
    tl_tiled_neg.start_address = tl_tiled_relu.start_address + required_size;

    // Load as a chunk of contiguous memory in global memory, not use global shape/stride
    // Local memory use tensor shape to maximize eu utilization.
    LLVM_DEBUG(llvm::errs() << llvm::format("  tdma load: gaddr 0x%x+0x%x\n", input_gaddr,
                                          global_input_offset););

    ctx.tdma_load(&tl_tiled_bottom, input_gaddr + global_input_offset);

    tl_leaky_relu(ctx, layer_id, tl_tiled_bottom, tl_tiled_relu, tl_tiled_neg, GT_right_shift_width,
                  LE_right_shift_width, GT_scale, LE_scale);

    // Store bottom as a chunk of contiguous memory, not use global shape/stride
    ctx.tdma_store(&tl_tiled_bottom, output_gaddr + global_input_offset);

    global_input_offset += tl_tiled_bottom_precise_stride.shape.n * tl_tiled_bottom_precise_stride.stride.n * NPU_NUM;
  }

  // Remaining count, in unit of eu_num
  if (global_input_offset != count) {
    LLVM_DEBUG(llvm::errs() << llvm::format(
        "global_input_offset != count (%d != %d)/n",
        global_input_offset, count););
    ASSERT(0);
  }
}
#else
void cvi_backend_tg_fixed_leakyrelu_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                            uint32_t inst_id, const uint32_t *depends, uint32_t depends_len,
                                            uint64_t input_gaddr, uint64_t output_gaddr, int input_n,
                                            int input_c, int input_h, int input_w,
                                            int GT_right_shift_width, int LE_right_shift_width,
                                            int GT_scale, int LE_scale) {
  LLVM_DEBUG(llvm::errs() << "cvi_backend_tg_fixed_leakyrelu_kernel"
                        << llvm::format(": "
                                        "input_gddr: %lx, output_gaddr:%lx\n"
                                        "GT_scale:%d, LE_scale:%d\n"
                                        "GT_right_shift_width:%d, LE_right_shift_width:%d\n"
                                        "input_n:%d input_c:%d input_h:%d input_w:%d\n",
                                        input_gaddr, output_gaddr, GT_scale, LE_scale,
                                        GT_right_shift_width, LE_right_shift_width, input_n,
                                        input_c, input_h, input_w));
  input_n = input_n;
  int count = input_n * input_c * input_h * input_w;
  int slice_num = get_slice_num_element_wise(ctx, 3, count + 1);
  gaddr_t slice_bottom_gaddr = input_gaddr;
  gaddr_t slice_top_gaddr = output_gaddr;

  for (int slice_idx = 0; slice_idx < slice_num; slice_idx++) {
    int count_sec = count / slice_num + (slice_idx < count % slice_num);

    // load with matrix format
    // set shape
    cvk_ml_shape_t input_shape = ctx.ml_shape_t1(count_sec, CVK_FMT_I8);
    cvk_ml_t *tl_bottom =
        ctx.lmem_alloc_matrix(input_shape, CVK_FMT_I8, 1);                                 // EU-aligned
    cvk_ml_t *tl_relu = ctx.lmem_alloc_matrix(input_shape, CVK_FMT_I8, 1);  // EU-aligned
    cvk_ml_t *tl_neg = ctx.lmem_alloc_matrix(input_shape, CVK_FMT_I8, 1);   // EU-aligned

    // Global memory from reshaped local memory
    cvk_mg_t ts_bottom;
    ts_bottom.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(slice_bottom_gaddr);
    ts_bottom.start_address = slice_bottom_gaddr;
    ts_bottom.shape = {static_cast<uint32_t>(input_shape.n), static_cast<uint32_t>(input_shape.col)};
    ts_bottom.stride = {static_cast<uint32_t>(input_shape.col)};

    LLVM_DEBUG(llvm::errs() << llvm::format("    [%d] tdma_g2l_tensor_copy: bottom\n"
                                          "      shape(n=%d, c=%d, w=%d, col=%d)\n",
                                          slice_idx, tl_bottom->shape.n, tl_bottom->shape.c,
                                          tl_bottom->shape.w, tl_bottom->shape.col));

    cvk_tdma_g2l_matrix_copy_param_t p1 = {0};
    p1.src = &ts_bottom;
    p1.dst = tl_bottom;
    ctx.tdma_g2l_matrix_copy(&p1);

    // Convert to tensor format
    cvk_tl_t bottom;
    bottom.start_address = tl_bottom->start_address;
    bottom.fmt = tl_bottom->fmt;
    bottom.shape = {tl_bottom->shape.n, tl_bottom->shape.c, 1, tl_bottom->shape.w};
    bottom.stride = {tl_bottom->stride.n, tl_bottom->stride.c, tl_bottom->stride.h, 1};

    cvk_tl_t relu;
    relu.start_address = tl_relu->start_address;
    relu.fmt = tl_relu->fmt;
    relu.shape = {tl_relu->shape.n, tl_relu->shape.c, 1, tl_relu->shape.w};
    relu.stride = {tl_relu->stride.n, tl_relu->stride.c, tl_relu->stride.h, 1};

    cvk_tl_t neg;
    neg.start_address = tl_neg->start_address;
    neg.fmt = tl_neg->fmt;
    neg.shape = {tl_neg->shape.n, tl_neg->shape.c, 1, tl_neg->shape.w};
    neg.stride = {tl_neg->stride.n, tl_neg->stride.c, tl_neg->stride.h, 1};

    // 0. relu = relu(bottom)
    cvk_tiu_max_param_t p13 = {0};
    p13.max = &relu;
    p13.a = &bottom;
    p13.b_is_const = 1;
    p13.b_const.is_signed = 1;
    p13.b_const.val = 0;
    ctx.tiu_max(&p13);

    // 1. relu = (relu * GT_scale) >> GT_right_shift_width
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &relu;
    p.a = &relu;
    p.b_const.val = GT_scale;
    p.b_const.is_signed = true;
    p.b_is_const = 1;
    p.rshift_bits = GT_right_shift_width;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    // 2. neg = neg(0, botom)
    cvk_tiu_min_param_t p7 = {0};
    p7.min = &neg;
    p7.a = &bottom;
    p7.b_is_const = 1;
    p7.b_const.val = 0;
    p7.b_const.is_signed = 1;
    ctx.tiu_min(&p7);

    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope) >> LE_right_shift_width
    p.res_high = nullptr;
    p.res_low = &neg;
    p.a = &neg;
    p.b_const.val = LE_scale;
    p.b_const.is_signed = true;
    p.b_is_const = 1;
    p.rshift_bits = LE_right_shift_width;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    // 4. bottom = or relu, neg
    cvk_tiu_or_int8_param_t p9 = {0};
    p9.res = &bottom;
    p9.a = &relu;
    p9.b = &neg;
    ctx.tiu_or_int8(&p9);

    // Store with matrix format
    // move result to global
    // Global memory shape == local memory shape
    // Gobal memory stride from local memory shape
    cvk_mg_t ts_top;
    ts_top.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(slice_top_gaddr);
    ts_top.start_address = slice_top_gaddr;
    ts_top.shape = {tl_bottom->shape.n, tl_bottom->shape.col};
    ts_top.stride = {tl_bottom->shape.col};

    LLVM_DEBUG(llvm::errs() << llvm::format("    [%d] gdma_store: bottom\n"
                                          "      shape(n=%d, c=%d, w=%d, col=%d)\n",
                                          slice_idx, tl_bottom->shape.n, tl_bottom->shape.c,
                                          tl_bottom->shape.w, tl_bottom->shape.col));

    cvk_tdma_l2g_matrix_copy_param_t p10 = {0};
    p10.src = tl_bottom;
    p10.dst = &ts_top;
    ctx.tdma_l2g_matrix_copy(&p10);

    // free
    ctx.lmem_free_matrix(tl_neg);
    ctx.lmem_free_matrix(tl_relu);
    ctx.lmem_free_matrix(tl_bottom);

    slice_bottom_gaddr += count_sec * sizeof(uint8_t);
    slice_top_gaddr += count_sec * sizeof(uint8_t);
  }
}
#endif

//}  // namespace bmnet
