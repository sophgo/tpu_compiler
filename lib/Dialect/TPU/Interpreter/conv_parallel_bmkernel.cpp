/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */

#include "conv_parallel_bmkernel.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
//#include <targets/plat-bm188x/bmkernel/bmkernel_api.h>
//#include <targets/plat-bm188x/parallel_conv_188x.hpp>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_conv"
#define DEBUG_SPLIT "bmnet_bm1880v2_bmkernel_conv_split"

#define DEBUG_BMNET(x) LLVM_DEBUG(x)

//namespace bmnet {
// TODO(wwcai): to remove
#if 1
#define CHIP_VERSION (ctx.hw.chip_version)
#define NODECHIP_SHIFT (ctx.hw.nodechip_shift)
#define NPU_SHIFT (ctx.hw.npu_shift)
#define EU_SHIFT (ctx.hw.eu_shift)
#define LOCAL_MEM_ADDRWIDTH (ctx.hw.local_mem_shift)
#define LOCAL_MEM_BANKS (ctx.hw.local_mem_banks)
#define GLOBAL_MEM_SIZE (ctx.hw.global_mem_size)
#define CHIP_IS_BM1680 (CHIP_VERSION == BM_CHIP_BM1680)
#define CHIP_IS_BM1682 (CHIP_VERSION == BM_CHIP_BM1682)
#define NODECHIP_NUM (1 << NODECHIP_SHIFT)
#define NODECHIP_MASK (NODECHIP_NUM - 1)
#define NPU_NUM (1 << NPU_SHIFT)
#define NPU_MASK (NPU_NUM - 1)
#define EU_NUM (1 << EU_SHIFT)
#define EU_MASK (EU_NUM - 1)
#define LOCAL_MEM_SIZE (1 << LOCAL_MEM_ADDRWIDTH)
#endif
#define ASSERT(x) assert(x)

#define RELU   1
#define PRELU  2

#define SPLIT_FAILED 0xFFFF

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Not split ic since it needs 32b ofmap for partial sum.
int BM1880v2ConvFixedParallelv2::split(const BM1880v2BackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "BM1880v2ConvFixedParallelv2::split =>\n"
                  "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                  "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                  "  stride (%d, %d), dilation (%d, %d)\n",
                  groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw, pad_top,
                  pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h, dilation_w));

  slices.n = 1;
  slices.oc = ceiling_func_shift(oc, NPU_SHIFT);  // lane parallelism
  slices.ic = 1;
  slices.h = (ih + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)
  slices.w = (iw + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)

  int oc_step = (oc >= ctx.hw.npu_num) ? ctx.hw.npu_num : oc;  // use all lanes

  // We may need to put EU-alignment info in one place
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i8 = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i16 = ctx.shape_t4(2, oc_step, 1, 1);

  u32 coeff_oc_step_size = 0;
  if (do_bias) {
    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  // prelu needs extra tl_slope compared to leaky relu.
  if (do_activation && activation_method == PRELU) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);
  }

  if (do_bn) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);

    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  if (do_scale) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);
  }

  if (do_scale_bias) {
    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  // Add weight size
  coeff_oc_step_size += ctx.lmem_tensor_to_size(ctx.shape_t4(ic, oc_step, kh, kw),
                                                /*eu_align=*/0);

  //
  // Slices may not be a good way to find size
  // We may try to increase or decrease width in aligned with 4, 8, 16 ...
  // or specific height/width (8, 8), (16, 16) ...
  //
  // Split ow
  for (slices.w = 1; slices.w <= ow; ++slices.w) {
    int ow_step = ceiling_func(ow, slices.w);
    int iw_step = math_min((ow_step - 1) * stride_w + kw_extent, iw);

    // Split oh
    for (slices.h = 1; slices.h <= oh; ++slices.h) {
      // split n
      for (slices.n = 1; slices.n <= n; ++slices.n) {
        int n_step = ceiling_func(n, slices.n);

        int oh_step = ceiling_func(oh, slices.h);
        int ih_step = math_min((oh_step - 1) * stride_h + kh_extent, ih);

        u32 total_needed = 0;

        u32 ofmap_size = ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, oc_step, oh_step, ow_step),
                                                 /*eu_align=*/1);
        total_needed += ofmap_size;

        u32 ifmap_size = ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, ic, ih_step, iw_step),
                                                 /*eu_align=*/1);
        total_needed += ifmap_size;

        total_needed += coeff_oc_step_size;

        // Double buffers so that TDMA load and store can run during TIU executes.
        total_needed *= 2;

        // Both prelu and leaky relu need tl_neg, tl_relu.
        // tl_relu, tl_neg are not from tmda and not final output.
        // One copy is enough.
        if (do_activation && ((activation_method == PRELU) ||
                              (activation_method == RELU && activation_arg[0] != 0.0f))) {
          total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
        }

        if (total_needed < LOCAL_MEM_SIZE) {
          DEBUG_BMNET(
              llvm::errs() << llvm::format(
                  "  Slices(n=%d, oc=%d, ic=%d, h=%d, w=%d), n_step %d, oh_step %d, ih_step %d"
                  ", coeff_oc_step_size %d, total_needed %d\n",
                  slices.n, slices.oc, slices.ic, slices.h, slices.w, n_step, oh_step, ih_step,
                  coeff_oc_step_size, total_needed));
          DEBUG_BMNET(llvm::errs() << "<= BM1880v2ConvFixedParallelv2::split succeed" << "/n");
          return total_needed;
        }

      }  // for (slices.n = 1; slices.n < n; ++slices.n)

    }  // for (slices.h = 1; slices.h <= oh; ++slices.h)

  }  // for (slices.w = 1; slices.w <= ow; ++slices.ow)

  llvm::errs() << "BM1880v2ConvFixedParallelv2::split fail";
  DEBUG_BMNET(llvm::errs() << "<= BM1880v2ConvFixedParallelv2::split fail" << "/n");

  return SPLIT_FAILED;
}

//
// This function implemnets weight reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
// TIU/TDMA command execution flow:
//   DMA G2L,  cmd_id 1, wait_id_tpu 0
//   DMA G2L,  cmd_id 2, wait_id_tpu 0
//   DMA G2L,  cmd_id 3, wait_id_tpu 0, LD0
//   TIU conv, cmd_id 1, cmd_id_gdma 3, TIU0, wait LD0
//   DMA G2L,  cmd_id 4, wait_id_tpu 0, LD1, no wait
//   TIU conv, cmd_id 2, cmd_id_gdma 4, TIU1, wait LD1
//   DMA L2G,  cmd_id 5, wait_id_tpu 1, SD0, wait TIU1
//   DMA G2L,  cmd_id 6, wait_id_tpu 0, LD2, no wait
//   TIU conv, cmd_id 3, cmd_id_gdma 6, TIU2, wait LD2
//   DMA L2G,  cmd_id 7, wait_id_tpu 2, SD1, wait TIU2
//   DMA G2L,  cmd_id 8, wait_id_tpu 0, LD3, no wait
//   TIU conv, cmd_id 4, cmd_id_gdma 8, TIU3, wait LD3
//
//   TDMA      TIU
//   LD0
//   LD1       TIU0
//   SD0/LD2   TIU1
//   SD1/LD3   TIU2
//
void ConvReuseWeight(const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t ga_ifmap,
                     gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean,
                     gaddr_t ga_bn_variance, gaddr_t ga_scale, gaddr_t ga_scale_bias, int input_n,
                     int input_c, int input_h, int input_w, int groups, int output_c, u16 kh,
                     u16 kw, u16 dilation_h, u16 dilation_w, u8 pad_top, u8 pad_bottom, u8 pad_left,
                     u8 pad_right, u8 stride_h, u8 stride_w, int do_bias, int do_bn, int do_scale,
                     int do_scale_bias, int do_activation, float bn_scale, float bn_eps,
                     int activation_method, float activation_arg[], gaddr_t activation_ga_slope,
                     bool activation_channel_shared, int activation_gt_scale,
                     int activation_gt_rshift, int activation_le_scale, int activation_le_rshift,
                     int right_shift_width, int bn_right_shift_width, int scale_right_shift_width,
                     SLICES &slices) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int ic_step = ceiling_func(ic, slices.ic);
  int oh_step = ceiling_func(oh, slices.h);
  int ow_step = ceiling_func(ow, slices.w);
  int ih_step = input_h;
  int iw_step = input_w;
  int oc_step = oc;

  // Always use all lanes.
  // Not divided by slices.oc.
  // E.g. mtcnn_det2_cic oc = 48, slices.oc = 2
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > ctx.hw.npu_num);
    oc_step = ctx.hw.npu_num;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "ConvReuseWeight =>\n"
                  "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                  "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                  "  stride (%d, %d), dilation (%d, %d)\n"
                  "  Slices (n=%d, oc=%d, ic=%d, h=%d, w=%d)\n",
                  groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw, pad_top,
                  pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h, dilation_w,
                  slices.n, slices.oc, slices.ic, slices.h, slices.w));

  bool fused_conv_relu =
      (!do_scale && !do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bool fused_conv_bn_relu =
      (!do_scale && do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bmk1880v2_tensor_lmem_shape_t oc_shape_ = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(n_step, ic_step, ih_step, input_w);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(n_step, oc_step, oh_step, ow);

  bmk1880v2_tensor_lmem_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_bn_mean[2] = {nullptr, nullptr},
                          *tl_bn_variance[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_scale[2] = {nullptr, nullptr}, *tl_scale_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_slope[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_ifmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_ofmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_neg = nullptr, *tl_relu = nullptr;

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  bmk1880v2_tensor_tgmem_stride_t ofmap_gstride = {static_cast<u32>(output_c) * oh * ow,
                                                   static_cast<u32>(oh) * ow, static_cast<u32>(ow)};
  bmk1880v2_tensor_tgmem_stride_t ifmap_gstride = {static_cast<u32>(input_c) * input_h * input_w,
                                                   static_cast<u32>(input_h) * input_w,
                                                   static_cast<u32>(input_w)};
  bmk1880v2_tensor_tgmem_stride_t bias_gstride = {static_cast<u32>(output_c), 1, 1};
  bmk1880v2_tensor_tgmem_stride_t weight_gstride = {
      static_cast<u32>(oc) * kh * kw * ic, static_cast<u32>(kh) * kw * ic, static_cast<u32>(ic)};

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_weight[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_ifmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ifmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] && tl_ofmap[1]);

  bmk1880v2_tensor_lmem_shape_t coeff_shape_i8 = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i16 = ctx.shape_t4(2, oc_step, 1, 1);
  if (do_bias) {
    // 16 bit
    tl_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_aling=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
  }

  // Both prelu and leaky relu needs tl_neg, tl_relu.
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    tl_neg = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                   /*eu_align=*/1);
    tl_relu = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                    /*eu_align=*/1);
    ASSERT(tl_neg && tl_relu);
  }

  // prelu needs extra tl_slope
  if (do_activation && activation_method == PRELU) {
    // weight of depthwise conv is aligned
    tl_slope[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_slope[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    ASSERT(tl_slope[0] && tl_slope[1]);
  }

  if (do_bn) {
    // weight of depthwise conv is aligned
    tl_bn_variance[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_bn_variance[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);

    // 16 bit
    tl_bn_mean[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bn_mean[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_bn_variance[0] && tl_bn_variance[1] && tl_bn_mean[0] && tl_bn_mean[1]);
  }

  if (do_scale) {
    // weight of depthwise conv is aligned
    tl_scale[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    tl_scale[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    ASSERT(tl_scale[0] && tl_scale[1]);
  }

  if (do_scale_bias) {
    // 16 bit
    tl_scale_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_scale_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_scale_bias[0] && tl_scale_bias[1]);
  }

  // split groups
  for (int ig = 0; ig < groups; ++ig) {
    int first = 1;
    int flip = 0;
    int coeff_flip = 0;
    gaddr_t ga_ofmap_cur[2] = {0};

    ctx.parallel_disable();

    // split oc
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = math_min(oc - oc_pos, oc_step);

      u64 coeff_offset = ig * oc + oc_pos;

      // Actual shape for tdma, tiu
      coeff_shape_i8 = ctx.shape_t4(1, cur_oc, 1, 1);
      coeff_shape_i16 = ctx.shape_t4(2, cur_oc, 1, 1);

      if (do_bias) {
        // 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bias[coeff_flip]->shape = coeff_shape_i16;
        tl_bias[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bias[coeff_flip]->shape, /*eu_aign=*/0);

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "  [ig=%d][oc_pos=%d] tdma_load_stride:\n"
                        "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                        "%d, %d), stride (%d, %d, %d)\n",
                        ig, oc_pos, ga_bias + coeff_offset, tl_bias[coeff_flip]->start_address,
                        tl_bias[coeff_flip]->shape.n, tl_bias[coeff_flip]->shape.c,
                        tl_bias[coeff_flip]->shape.h, tl_bias[coeff_flip]->shape.w, bias_gstride.n,
                        bias_gstride.c, bias_gstride.h));
        ctx.tdma_load_stride(tl_bias[coeff_flip], ga_bias + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      if (do_activation && activation_method == PRELU) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_slope[coeff_flip]->shape = coeff_shape_i8;
        tl_slope[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_slope[coeff_flip]->shape, /*eu_align=*/1);

        ctx.tdma_load(tl_slope[coeff_flip], activation_ga_slope + coeff_offset, CTRL_WEIGHT);
      }

      if (do_bn) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bn_variance[coeff_flip]->shape = coeff_shape_i8;
        tl_bn_variance[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bn_variance[coeff_flip]->shape, /*eu_align=*/1);
        ctx.tdma_load(tl_bn_variance[coeff_flip], ga_bn_variance + coeff_offset, CTRL_WEIGHT);

        // 16 bit
        tl_bn_mean[coeff_flip]->shape = coeff_shape_i16;
        tl_bn_mean[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bn_mean[coeff_flip]->shape, /*eu_align=*/0);
        ctx.tdma_load_stride(tl_bn_mean[coeff_flip], ga_bn_mean + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      if (do_scale) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale[coeff_flip]->shape = coeff_shape_i8;
        tl_scale[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_scale[coeff_flip]->shape, /*eu_align=*/1);
        ctx.tdma_load(tl_scale[coeff_flip], ga_scale + coeff_offset, CTRL_WEIGHT);
      }

      if (do_scale_bias) {
        // 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale_bias[coeff_flip]->shape = coeff_shape_i16;
        tl_scale_bias[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_scale_bias[coeff_flip]->shape, /*eu_align=*/0);
        ctx.tdma_load_stride(tl_scale_bias[coeff_flip], ga_scale_bias + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      // Weight shape for load != shape for tiu
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_weight[coeff_flip]->shape = ctx.shape_t4(ic, cur_oc, kh, kw);
      tl_weight[coeff_flip]->stride =
          ctx.tensor_lmem_default_stride(tl_weight[coeff_flip]->shape, /*eu_aign*/ 0);

      u64 weight_offset = ga_weight + ig * oc * ic * kh * kw + oc_pos * ic * kh * kw;
      {
        // Same local address, different shape, stride
        bmk1880v2_tensor_lmem_t tl_tmp;
        tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
        tl_tmp.fmt = FMT_I8;
        tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic);
        tl_tmp.stride = ctx.tensor_lmem_default_stride(tl_tmp.shape, /*eu_align=*/0);

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "  [ig=%d][oc_pos=%d] tdma_load_stride:\n"
                        "    tl_weight gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                        "%d, %d), stride (%d, %d, %d)\n",
                        ig, oc_pos, weight_offset, tl_tmp.start_address, tl_tmp.shape.n,
                        tl_tmp.shape.c, tl_tmp.shape.h, tl_tmp.shape.w, tl_tmp.stride.n,
                        tl_tmp.stride.c, tl_tmp.stride.h, tl_tmp.stride.w));
        ctx.tdma_load_stride(&tl_tmp, weight_offset, weight_gstride, CTRL_WEIGHT);
      }

      bmk1880v2_tensor_lmem_shape_t ifmap_shape[2] = {0};
      bmk1880v2_tensor_lmem_shape_t ofmap_shape[2] = {0};
      gaddr_t ga_ifmap_cur[2] = {0};

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = math_min(input_n - n_pos, n_step);

        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = math_min(oh - oh_pos, oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = math_max(oh_top * stride_h - pad_top, 0);
          int ih_bot = math_min((oh_bot - 1) * stride_h + kh_ext - pad_top, input_h);
          int cur_ih = ih_bot - ih_top;

          int ph_top = 0;
          if (ih_top == 0) {
            ph_top = pad_top - oh_top * stride_h;
          }

          int ph_bot = 0;
          if (ih_bot == input_h) {
            ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top - input_h;
          }

          // split w
          for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step) {
            int cur_ow = math_min(ow - ow_pos, ow_step);

            int ow_left = ow_pos;
            int ow_right = ow_left + cur_ow;
            int iw_left = math_max(ow_left * stride_w - pad_left, 0);
            int iw_right = math_min((ow_right - 1) * stride_w + kw_ext - pad_left, input_w);
            int cur_iw = iw_right - iw_left;

            int pw_left = 0;
            if (iw_left == 0) {
              pw_left = pad_left - ow_left * stride_w;
            }

            int pw_right = 0;
            if (iw_right == input_w) {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
            }

            DEBUG_BMNET(
                llvm::errs() << llvm::format("  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d]"
                                          " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                                          ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                                          ig, oc_pos, n_pos, oh_pos, ow_pos, cur_oh, cur_ih, ih_top,
                                          ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ofmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[flip]->stride =
                ctx.tensor_lmem_default_stride(tl_ofmap[flip]->shape, /*eu_aign=*/1);

            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, ic, cur_ih, cur_iw);
            tl_ifmap[flip]->stride =
                ctx.tensor_lmem_default_stride(tl_ifmap[flip]->shape, /*eu_align=*/1);

            u64 ifmap_offset = ga_ifmap + ig * ic * input_h * input_w +
                               n_pos * input_c * input_h * input_w + ih_top * input_w + iw_left;

            DEBUG_BMNET(
                llvm::errs() << llvm::format(
                    "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] tdma_load_stride:\n"
                    "    tl_ifmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                    "%d, %d), stride (%d, %d, %d)\n",
                    ig, oc_pos, n_pos, oh_pos, ow_pos, ifmap_offset, tl_ifmap[flip]->start_address,
                    tl_ifmap[flip]->shape.n, tl_ifmap[flip]->shape.c, tl_ifmap[flip]->shape.h,
                    tl_ifmap[flip]->shape.w, tl_ifmap[flip]->stride.n, tl_ifmap[flip]->stride.c,
                    tl_ifmap[flip]->stride.h, tl_ifmap[flip]->stride.w));

            ctx.tdma_load_stride(tl_ifmap[flip], ifmap_offset, ifmap_gstride, CTRL_NEURON);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              bmk1880v2_tiu_convolution_param_t param;
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = param.ins_last_h = 0;
              param.ins_w = param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pad_left;
              param.pad_right = pad_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.rshift_bits = right_shift_width;
              param.enable_double_conv = 0;
              param.ps32_mode = 0;
              param.w_is_const = 0;
              param.layer_id = layer_id;

              DEBUG_BMNET(llvm::errs() << llvm::format(
                              "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] conv:\n"
                              "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                              ig, oc_pos, n_pos, oh_pos, ow_pos, param.ifmap->start_address,
                              param.ifmap->shape.n, param.ifmap->shape.c, param.ifmap->shape.h,
                              param.ifmap->shape.w, param.weight->start_address,
                              param.weight->shape.n, param.weight->shape.c, param.weight->shape.h,
                              param.weight->shape.w, param.ofmap->start_address,
                              param.ofmap->shape.n, param.ofmap->shape.c, param.ofmap->shape.h,
                              param.ofmap->shape.w));

              ctx.tiu_convolution(&param);
            }

            bmk1880v2_tiu_depthwise_convolution_param_t param;
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
            param.relu_enable = 0;
            param.layer_id = layer_id;
            if (do_bn) {
              // out(n,c,h,w) = in(n,c,h,w) * variance(1,c,1,1) + mean(1,c,1,1)
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_bn_variance[coeff_flip];
              param.bias = tl_bn_mean[coeff_flip];
              param.rshift_bits = bn_right_shift_width;
              param.relu_enable = fused_conv_bn_relu;
              ctx.tiu_depthwise_convolution(&param);
            }
            if (do_scale && do_scale_bias) {
              // computing x * scale + bias
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = tl_scale_bias[coeff_flip];
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale) {
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = nullptr;
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale_bias) {
              ASSERT(0);  // TODO(zakk)
            }

            if (do_activation) {
              switch (activation_method) {
                case RELU:
                  if (activation_arg[0] == 0.0f) {
                    // relu
                    if (!fused_conv_relu && !fused_conv_bn_relu) {
                      // should not come here !
                      ASSERT(0);

                      bmk1880v2_tiu_element_wise_max_param_t p13;
                      p13.max = tl_ofmap[flip];
                      p13.a = tl_ofmap[flip];
                      p13.b_is_const = 1;
                      p13.b_is_signed = 1;
                      p13.b_val = 0;
                      p13.layer_id = layer_id;
                      ctx.tiu_element_wise_max(&p13);
                    }
                  } else {
                    // leaky relu

                    // bmk does not keep eu-align info, user need to update stride if shape changed
                    tl_relu->shape = tl_ofmap[flip]->shape;
                    tl_relu->stride =
                        ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                    tl_neg->shape = tl_ofmap[flip]->shape;
                    tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                    bmk1880v2_tiu_element_wise_max_param_t p1;
                    p1.max = tl_relu;
                    p1.a = tl_ofmap[flip];
                    p1.b_is_const = 1;
                    p1.b_is_signed = 1;
                    p1.b_val = 0;
                    p1.layer_id = layer_id;
                    ctx.tiu_element_wise_max(&p1);

                    bmk1880v2_tiu_element_wise_mul_param_t p2;
                    p2.res_high = nullptr;
                    p2.res_low = tl_relu;
                    p2.a = tl_relu;
                    p2.b_val = activation_gt_scale;
                    p2.b_is_signed = true;
                    p2.b_is_const = 1;
                    p2.rshift_bits = activation_gt_rshift;
                    p2.layer_id = layer_id;
                    p2.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p2);

                    bmk1880v2_tiu_element_wise_min_param_t p3;
                    p3.min = tl_neg;
                    p3.a = tl_ofmap[flip];
                    p3.b_is_const = 1;
                    p3.b_val = 0;
                    p3.b_is_signed = 1;
                    p3.layer_id = layer_id;
                    ctx.tiu_element_wise_min(&p3);

                    bmk1880v2_tiu_element_wise_mul_param_t p4;
                    p4.res_high = nullptr;
                    p4.res_low = tl_neg;
                    p4.a = tl_neg;
                    p4.b_val = activation_le_scale;
                    p4.b_is_signed = true;
                    p4.b_is_const = 1;
                    p4.rshift_bits = activation_le_rshift;
                    p4.layer_id = layer_id;
                    p4.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p4);

                    bmk1880v2_tiu_element_wise_or_int8_param_t p5;
                    p5.res = tl_ofmap[flip];
                    p5.a = tl_relu;
                    p5.b = tl_neg;
                    p5.layer_id = layer_id;
                    ctx.tiu_element_wise_or_int8(&p5);
                  }
                  break;
                case PRELU: {
                  ASSERT(!activation_channel_shared);

                  // bmk does not keep eu-align info, user need to update stride if shape changed
                  tl_relu->shape = tl_ofmap[flip]->shape;
                  tl_relu->stride = ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                  tl_neg->shape = tl_ofmap[flip]->shape;
                  tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                  // 0. relu = relu(tl_ofmap)
                  // 1. relu = (relu * gt_scale) >> gt_rshift
                  // 2. neg = neg(0, botom)
                  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
                  // 4. tl_ofmap = or relu, neg
                  bmk1880v2_tiu_element_wise_max_param_t p2;
                  p2.max = tl_relu;
                  p2.a = tl_ofmap[flip];
                  p2.b_is_const = 1;
                  p2.b_is_signed = 1;
                  p2.b_val = 0;
                  p2.layer_id = layer_id;
                  ctx.tiu_element_wise_max(&p2);

                  bmk1880v2_tiu_element_wise_mul_param_t p3;
                  p3.res_high = nullptr;
                  p3.res_low = tl_relu;
                  p3.a = tl_relu;
                  p3.b_val = activation_gt_scale;
                  p3.b_is_signed = true;
                  p3.b_is_const = 1;
                  p3.rshift_bits = activation_gt_rshift;
                  p3.layer_id = layer_id;
                  p3.relu_enable = 0;
                  ctx.tiu_element_wise_mul(&p3);

                  bmk1880v2_tiu_element_wise_min_param_t p4;
                  p4.min = tl_neg;
                  p4.a = tl_ofmap[flip];
                  p4.b_is_const = 1;
                  p4.b_val = 0;
                  p4.b_is_signed = 1;
                  p4.layer_id = layer_id;
                  ctx.tiu_element_wise_min(&p4);

                  bmk1880v2_tiu_depthwise_convolution_param_t p5;
                  p5.ins_h = 0;
                  p5.ins_last_h = 0;
                  p5.ins_w = 0;
                  p5.ins_last_w = 0;
                  p5.pad_top = 0;
                  p5.pad_bottom = 0;
                  p5.pad_left = 0;
                  p5.pad_right = 0;
                  p5.stride_h = 1;
                  p5.stride_w = 1;
                  p5.dilation_h = 1;
                  p5.dilation_w = 1;
                  p5.ofmap = tl_neg;
                  p5.ifmap = tl_neg;
                  p5.weight = tl_slope[coeff_flip];
                  p5.bias = nullptr;
                  p5.rshift_bits = activation_le_rshift;
                  p5.relu_enable = 0;
                  p5.layer_id = layer_id;
                  ctx.tiu_depthwise_convolution(&p5);

                  bmk1880v2_tiu_element_wise_or_int8_param_t p6;
                  p6.res = tl_ofmap[flip];
                  p6.a = tl_relu;
                  p6.b = tl_neg;
                  p6.layer_id = layer_id;
                  ctx.tiu_element_wise_or_int8(&p6);
                } break;
                default:
                  ASSERT(0);
              }  // switch (activation_method)
            }    // if (do_activation)

            ga_ofmap_cur[flip] = ga_ofmap + ig * oc * oh * ow + n_pos * output_c * oh * ow +
                                 oc_pos * oh * ow + oh_top * ow + ow_left;

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int flip_back = 1 - flip;

              // Store back to global memory
              DEBUG_BMNET(
                  llvm::errs() << llvm::format(
                      "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] tdma_store_stride:\n"
                      "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                      "%d, %d), stride (%d, %d, %d)\n",
                      ig, oc_pos, n_pos, oh_pos, ow_pos, ga_ofmap_cur[flip_back],
                      tl_ofmap[flip_back]->start_address, tl_ofmap[flip_back]->shape.n,
                      tl_ofmap[flip_back]->shape.c, tl_ofmap[flip_back]->shape.h,
                      tl_ofmap[flip_back]->shape.w, tl_ofmap[flip_back]->stride.n,
                      tl_ofmap[flip_back]->stride.c, tl_ofmap[flip_back]->stride.h,
                      tl_ofmap[flip_back]->stride.w));

              ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_gstride,
                                    CTRL_NEURON);
            }

            flip = 1 - flip;

          }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

        }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      }  // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    int flip_back = 1 - flip;

    // Store back to global memory
    DEBUG_BMNET(
        llvm::errs() << llvm::format("  [ig=%d] tdma_store_stride:\n"
                                  "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                                  "%d, %d), stride (%d, %d, %d)\n",
                                  ig, ga_ofmap_cur[flip_back], tl_ofmap[flip_back]->start_address,
                                  tl_ofmap[flip_back]->shape.n, tl_ofmap[flip_back]->shape.c,
                                  tl_ofmap[flip_back]->shape.h, tl_ofmap[flip_back]->shape.w,
                                  tl_ofmap[flip_back]->stride.n, tl_ofmap[flip_back]->stride.c,
                                  tl_ofmap[flip_back]->stride.h, tl_ofmap[flip_back]->stride.w));

    ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_gstride, CTRL_NEURON);

  }  // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias[1]);
    ctx.lmem_free_tensor(tl_scale_bias[0]);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale[1]);
    ctx.lmem_free_tensor(tl_scale[0]);
  }
  if (do_bn) {
    ctx.lmem_free_tensor(tl_bn_mean[1]);
    ctx.lmem_free_tensor(tl_bn_mean[0]);
    ctx.lmem_free_tensor(tl_bn_variance[1]);
    ctx.lmem_free_tensor(tl_bn_variance[0]);
  }
  if (do_activation && activation_method == PRELU) {
    ctx.lmem_free_tensor(tl_slope[1]);
    ctx.lmem_free_tensor(tl_slope[0]);
  }
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    ctx.lmem_free_tensor(tl_relu);
    ctx.lmem_free_tensor(tl_neg);
  }
  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias[1]);
    ctx.lmem_free_tensor(tl_bias[0]);
  }
  ctx.lmem_free_tensor(tl_ofmap[1]);
  ctx.lmem_free_tensor(tl_ofmap[0]);
  ctx.lmem_free_tensor(tl_ifmap[1]);
  ctx.lmem_free_tensor(tl_ifmap[0]);
  ctx.lmem_free_tensor(tl_weight[1]);
  ctx.lmem_free_tensor(tl_weight[0]);

  DEBUG_BMNET(llvm::errs() << "<=ConvReuseWeight" << "/n");
}

//
// This function implemnets activation(ifmap) reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
void ConvReuseActivation(const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t ga_ifmap,
                         gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean,
                         gaddr_t ga_bn_variance, gaddr_t ga_scale, gaddr_t ga_scale_bias,
                         int input_n, int input_c, int input_h, int input_w, int groups,
                         int output_c, u16 kh, u16 kw, u16 dilation_h, u16 dilation_w, u8 pad_top,
                         u8 pad_bottom, u8 pad_left, u8 pad_right, u8 stride_h, u8 stride_w,
                         int do_bias, int do_bn, int do_scale, int do_scale_bias, int do_activation,
                         float bn_scale, float bn_eps, int activation_method,
                         float activation_arg[], gaddr_t activation_ga_slope,
                         bool activation_channel_shared, int activation_gt_scale,
                         int activation_gt_rshift, int activation_le_scale,
                         int activation_le_rshift, int right_shift_width, int bn_right_shift_width,
                         int scale_right_shift_width, SLICES &slices) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int ic_step = ceiling_func(ic, slices.ic);
  int oh_step = ceiling_func(oh, slices.h);
  int ow_step = ceiling_func(ow, slices.w);
  int ih_step = input_h;
  int iw_step = input_w;
  int oc_step = oc;

  // Always use all lanes.
  // Not divided by slices.oc.
  // E.g. mtcnn_det2_cic oc = 48, slices.oc = 2
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > ctx.hw.npu_num);
    oc_step = ctx.hw.npu_num;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "ConvReuseActivation =>\n"
                  "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                  "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                  "  stride (%d, %d), dilation (%d, %d)\n"
                  "  Slices (n=%d, oc=%d, ic=%d, h=%d, w=%d)\n",
                  groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw, pad_top,
                  pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h, dilation_w,
                  slices.n, slices.oc, slices.ic, slices.h, slices.w));

  bool fused_conv_relu =
      (!do_scale && !do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bool fused_conv_bn_relu =
      (!do_scale && do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bmk1880v2_tensor_lmem_shape_t oc_shape_ = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(n_step, ic_step, ih_step, input_w);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(n_step, oc_step, oh_step, ow);

  bmk1880v2_tensor_lmem_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_bn_mean[2] = {nullptr, nullptr},
                          *tl_bn_variance[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_scale[2] = {nullptr, nullptr}, *tl_scale_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_slope[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_ifmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_ofmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_neg = nullptr, *tl_relu = nullptr;

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  bmk1880v2_tensor_tgmem_stride_t ofmap_gstride = {static_cast<u32>(output_c) * oh * ow,
                                                   static_cast<u32>(oh) * ow, static_cast<u32>(ow)};
  bmk1880v2_tensor_tgmem_stride_t ifmap_gstride = {static_cast<u32>(input_c) * input_h * input_w,
                                                   static_cast<u32>(input_h) * input_w,
                                                   static_cast<u32>(input_w)};
  bmk1880v2_tensor_tgmem_stride_t bias_gstride = {static_cast<u32>(output_c), 1, 1};
  bmk1880v2_tensor_tgmem_stride_t weight_gstride = {
      static_cast<u32>(oc) * kh * kw * ic, static_cast<u32>(kh) * kw * ic, static_cast<u32>(ic)};

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_weight[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_ifmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ifmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] && tl_ofmap[1]);

  bmk1880v2_tensor_lmem_shape_t coeff_shape_i8 = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i16 = ctx.shape_t4(2, oc_step, 1, 1);
  if (do_bias) {
    // 16 bit
    tl_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_aling=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
  }

  // Both prelu and leaky relu needs tl_neg, tl_relu.
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    tl_neg = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                   /*eu_align=*/1);
    tl_relu = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                    /*eu_align=*/1);
    ASSERT(tl_neg && tl_relu);
  }

  // prelu needs extra tl_slope
  if (do_activation && activation_method == PRELU) {
    // weight of depthwise conv is aligned
    tl_slope[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_slope[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    ASSERT(tl_slope[0] && tl_slope[1]);
  }

  if (do_bn) {
    // weight of depthwise conv is aligned
    tl_bn_variance[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_bn_variance[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);

    // 16 bit
    tl_bn_mean[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bn_mean[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_bn_variance[0] && tl_bn_variance[1] && tl_bn_mean[0] && tl_bn_mean[1]);
  }

  if (do_scale) {
    // weight of depthwise conv is aligned
    tl_scale[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    tl_scale[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    ASSERT(tl_scale[0] && tl_scale[1]);
  }

  if (do_scale_bias) {
    // 16 bit
    tl_scale_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_scale_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_scale_bias[0] && tl_scale_bias[1]);
  }

  // split groups
  for (int ig = 0; ig < groups; ++ig) {
    int first = 1;
    int flip = 0;
    int coeff_flip = 0;
    gaddr_t ga_ofmap_cur[2] = {0};

    ctx.parallel_disable();

    bmk1880v2_tensor_lmem_shape_t ifmap_shape[2] = {0};
    bmk1880v2_tensor_lmem_shape_t ofmap_shape[2] = {0};
    gaddr_t ga_ifmap_cur[2] = {0};

    // split n
    for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
      int cur_n = math_min(input_n - n_pos, n_step);

      // split h
      for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
        int cur_oh = math_min(oh - oh_pos, oh_step);

        int oh_top = oh_pos;
        int oh_bot = oh_top + cur_oh;
        int ih_top = math_max(oh_top * stride_h - pad_top, 0);
        int ih_bot = math_min((oh_bot - 1) * stride_h + kh_ext - pad_top, input_h);
        int cur_ih = ih_bot - ih_top;

        int ph_top = 0;
        if (ih_top == 0) {
          ph_top = pad_top - oh_top * stride_h;
        }

        int ph_bot = 0;
        if (ih_bot == input_h) {
          ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top - input_h;
        }

        // split w
        for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step) {
          int cur_ow = math_min(ow - ow_pos, ow_step);

          int ow_left = ow_pos;
          int ow_right = ow_left + cur_ow;
          int iw_left = math_max(ow_left * stride_w - pad_left, 0);
          int iw_right = math_min((ow_right - 1) * stride_w + kw_ext - pad_left, input_w);
          int cur_iw = iw_right - iw_left;

          int pw_left = 0;
          if (iw_left == 0) {
            pw_left = pad_left - ow_left * stride_w;
          }

          int pw_right = 0;
          if (iw_right == input_w) {
            pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
          }

          // bmk does not keep eu-align info, user need to update stride if shape changed
          tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, ic, cur_ih, cur_iw);
          tl_ifmap[flip]->stride =
              ctx.tensor_lmem_default_stride(tl_ifmap[flip]->shape, /*eu_align=*/1);

          u64 ifmap_offset = ga_ifmap + ig * ic * input_h * input_w +
                             n_pos * input_c * input_h * input_w + ih_top * input_w + iw_left;

          DEBUG_BMNET(llvm::errs() << llvm::format(
                          "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] tdma_load_stride:\n"
                          "    tl_ifmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                          "%d, %d), stride (%d, %d, %d)\n",
                          ig, n_pos, oh_pos, ow_pos, ifmap_offset, tl_ifmap[flip]->start_address,
                          tl_ifmap[flip]->shape.n, tl_ifmap[flip]->shape.c, tl_ifmap[flip]->shape.h,
                          tl_ifmap[flip]->shape.w, tl_ifmap[flip]->stride.n,
                          tl_ifmap[flip]->stride.c, tl_ifmap[flip]->stride.h,
                          tl_ifmap[flip]->stride.w));

          ctx.tdma_load_stride(tl_ifmap[flip], ifmap_offset, ifmap_gstride, CTRL_NEURON);

          // split oc
          for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
            int cur_oc = math_min(oc - oc_pos, oc_step);

            u64 coeff_offset = ig * oc + oc_pos;

            // Actual shape for tdma, tiu
            coeff_shape_i8 = ctx.shape_t4(1, cur_oc, 1, 1);
            coeff_shape_i16 = ctx.shape_t4(2, cur_oc, 1, 1);

            if (do_bias) {
              // 16 bit
              // bmk does not keep eu-align info, user need to update stride if shape changed
              tl_bias[coeff_flip]->shape = coeff_shape_i16;
              tl_bias[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_bias[coeff_flip]->shape, /*eu_aign=*/0);

              DEBUG_BMNET(
                  llvm::errs() << llvm::format(
                      "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] tdma_load_stride:\n"
                      "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                      "%d, %d), stride (%d, %d, %d)\n",
                      ig, n_pos, oh_pos, ow_pos, oc_pos, ga_bias + coeff_offset,
                      tl_bias[coeff_flip]->start_address, tl_bias[coeff_flip]->shape.n,
                      tl_bias[coeff_flip]->shape.c, tl_bias[coeff_flip]->shape.h,
                      tl_bias[coeff_flip]->shape.w, bias_gstride.n, bias_gstride.c,
                      bias_gstride.h));
              ctx.tdma_load_stride(tl_bias[coeff_flip], ga_bias + coeff_offset, bias_gstride,
                                   CTRL_WEIGHT);
            }

            if (do_activation && activation_method == PRELU) {
              // weight of depthwise conv is aligned
              // bmk does not keep eu-align info, user need to update stride if shape changed
              tl_slope[coeff_flip]->shape = coeff_shape_i8;
              tl_slope[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_slope[coeff_flip]->shape, /*eu_align=*/1);

              ctx.tdma_load(tl_slope[coeff_flip], activation_ga_slope + coeff_offset, CTRL_WEIGHT);
            }

            if (do_bn) {
              // weight of depthwise conv is aligned
              // bmk does not keep eu-align info, user need to update stride if shape changed
              tl_bn_variance[coeff_flip]->shape = coeff_shape_i8;
              tl_bn_variance[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_bn_variance[coeff_flip]->shape, /*eu_align=*/1);
              ctx.tdma_load(tl_bn_variance[coeff_flip], ga_bn_variance + coeff_offset, CTRL_WEIGHT);

              // 16 bit
              tl_bn_mean[coeff_flip]->shape = coeff_shape_i16;
              tl_bn_mean[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_bn_mean[coeff_flip]->shape, /*eu_align=*/0);
              ctx.tdma_load_stride(tl_bn_mean[coeff_flip], ga_bn_mean + coeff_offset, bias_gstride,
                                   CTRL_WEIGHT);
            }

            if (do_scale) {
              // weight of depthwise conv is aligned
              // bmk does not keep eu-align info, user need to update stride if shape changed
              tl_scale[coeff_flip]->shape = coeff_shape_i8;
              tl_scale[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_scale[coeff_flip]->shape, /*eu_align=*/1);
              ctx.tdma_load(tl_scale[coeff_flip], ga_scale + coeff_offset, CTRL_WEIGHT);
            }

            if (do_scale_bias) {
              // 16 bit
              // bmk does not keep eu-align info, user need to update stride if shape changed
              tl_scale_bias[coeff_flip]->shape = coeff_shape_i16;
              tl_scale_bias[coeff_flip]->stride =
                  ctx.tensor_lmem_default_stride(tl_scale_bias[coeff_flip]->shape, /*eu_align=*/0);
              ctx.tdma_load_stride(tl_scale_bias[coeff_flip], ga_scale_bias + coeff_offset,
                                   bias_gstride, CTRL_WEIGHT);
            }

            // Weight shape for load != shape for tiu
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_weight[coeff_flip]->shape = ctx.shape_t4(ic, cur_oc, kh, kw);
            tl_weight[coeff_flip]->stride =
                ctx.tensor_lmem_default_stride(tl_weight[coeff_flip]->shape, /*eu_aign*/ 0);

            u64 weight_offset = ga_weight + ig * oc * ic * kh * kw + oc_pos * ic * kh * kw;
            {
              // Same local address, different shape, stride
              bmk1880v2_tensor_lmem_t tl_tmp;
              tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
              tl_tmp.fmt = FMT_I8;
              tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic);
              tl_tmp.stride = ctx.tensor_lmem_default_stride(tl_tmp.shape, /*eu_align=*/0);

              DEBUG_BMNET(
                  llvm::errs() << llvm::format(
                      "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] tdma_load_stride:\n"
                      "    tl_weight gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                      "%d, %d), stride (%d, %d, %d)\n",
                      ig, n_pos, oh_pos, ow_pos, oc_pos, weight_offset, tl_tmp.start_address,
                      tl_tmp.shape.n, tl_tmp.shape.c, tl_tmp.shape.h, tl_tmp.shape.w,
                      tl_tmp.stride.n, tl_tmp.stride.c, tl_tmp.stride.h, tl_tmp.stride.w));

              ctx.tdma_load_stride(&tl_tmp, weight_offset, weight_gstride, CTRL_WEIGHT);
            }

            DEBUG_BMNET(
                llvm::errs() << llvm::format("  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d]"
                                          " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                                          ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                                          ig, n_pos, oh_pos, ow_pos, oc_pos, cur_oh, cur_ih, ih_top,
                                          ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ofmap[coeff_flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[coeff_flip]->stride =
                ctx.tensor_lmem_default_stride(tl_ofmap[coeff_flip]->shape, /*eu_aign=*/1);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              bmk1880v2_tiu_convolution_param_t param;
              param.ofmap = tl_ofmap[coeff_flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = param.ins_last_h = 0;
              param.ins_w = param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pad_left;
              param.pad_right = pad_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.rshift_bits = right_shift_width;
              param.enable_double_conv = 0;
              param.ps32_mode = 0;
              param.w_is_const = 0;
              param.layer_id = layer_id;

              DEBUG_BMNET(llvm::errs() << llvm::format(
                              "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] conv:\n"
                              "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                              ig, n_pos, oh_pos, ow_pos, oc_pos, param.ifmap->start_address,
                              param.ifmap->shape.n, param.ifmap->shape.c, param.ifmap->shape.h,
                              param.ifmap->shape.w, param.weight->start_address,
                              param.weight->shape.n, param.weight->shape.c, param.weight->shape.h,
                              param.weight->shape.w, param.ofmap->start_address,
                              param.ofmap->shape.n, param.ofmap->shape.c, param.ofmap->shape.h,
                              param.ofmap->shape.w));

              ctx.tiu_convolution(&param);
            }

            bmk1880v2_tiu_depthwise_convolution_param_t param;
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
            param.relu_enable = 0;
            param.layer_id = layer_id;
            if (do_bn) {
              // out(n,c,h,w) = in(n,c,h,w) * variance(1,c,1,1) + mean(1,c,1,1)
              param.ofmap = tl_ofmap[coeff_flip];
              param.ifmap = tl_ofmap[coeff_flip];
              param.weight = tl_bn_variance[coeff_flip];
              param.bias = tl_bn_mean[coeff_flip];
              param.rshift_bits = bn_right_shift_width;
              param.relu_enable = fused_conv_bn_relu;
              ctx.tiu_depthwise_convolution(&param);
            }
            if (do_scale && do_scale_bias) {
              // computing x * scale + bias
              param.ofmap = tl_ofmap[coeff_flip];
              param.ifmap = tl_ofmap[coeff_flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = tl_scale_bias[coeff_flip];
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale) {
              param.ofmap = tl_ofmap[coeff_flip];
              param.ifmap = tl_ofmap[coeff_flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = nullptr;
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale_bias) {
              ASSERT(0);  // TODO(zakk)
            }

            if (do_activation) {
              switch (activation_method) {
                case RELU:
                  if (activation_arg[0] == 0.0f) {
                    // relu
                    if (!fused_conv_relu && !fused_conv_bn_relu) {
                      // should not come here !
                      ASSERT(0);

                      bmk1880v2_tiu_element_wise_max_param_t p13;
                      p13.max = tl_ofmap[coeff_flip];
                      p13.a = tl_ofmap[coeff_flip];
                      p13.b_is_const = 1;
                      p13.b_is_signed = 1;
                      p13.b_val = 0;
                      p13.layer_id = layer_id;
                      ctx.tiu_element_wise_max(&p13);
                    }
                  } else {
                    // leaky relu

                    // bmk does not keep eu-align info, user need to update stride if shape changed
                    tl_relu->shape = tl_ofmap[coeff_flip]->shape;
                    tl_relu->stride =
                        ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                    tl_neg->shape = tl_ofmap[coeff_flip]->shape;
                    tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                    bmk1880v2_tiu_element_wise_max_param_t p1;
                    p1.max = tl_relu;
                    p1.a = tl_ofmap[coeff_flip];
                    p1.b_is_const = 1;
                    p1.b_is_signed = 1;
                    p1.b_val = 0;
                    p1.layer_id = layer_id;
                    ctx.tiu_element_wise_max(&p1);

                    bmk1880v2_tiu_element_wise_mul_param_t p2;
                    p2.res_high = nullptr;
                    p2.res_low = tl_relu;
                    p2.a = tl_relu;
                    p2.b_val = activation_gt_scale;
                    p2.b_is_signed = true;
                    p2.b_is_const = 1;
                    p2.rshift_bits = activation_gt_rshift;
                    p2.layer_id = layer_id;
                    p2.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p2);

                    bmk1880v2_tiu_element_wise_min_param_t p3;
                    p3.min = tl_neg;
                    p3.a = tl_ofmap[coeff_flip];
                    p3.b_is_const = 1;
                    p3.b_val = 0;
                    p3.b_is_signed = 1;
                    p3.layer_id = layer_id;
                    ctx.tiu_element_wise_min(&p3);

                    bmk1880v2_tiu_element_wise_mul_param_t p4;
                    p4.res_high = nullptr;
                    p4.res_low = tl_neg;
                    p4.a = tl_neg;
                    p4.b_val = activation_le_scale;
                    p4.b_is_signed = true;
                    p4.b_is_const = 1;
                    p4.rshift_bits = activation_le_rshift;
                    p4.layer_id = layer_id;
                    p4.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p4);

                    bmk1880v2_tiu_element_wise_or_int8_param_t p5;
                    p5.res = tl_ofmap[coeff_flip];
                    p5.a = tl_relu;
                    p5.b = tl_neg;
                    p5.layer_id = layer_id;
                    ctx.tiu_element_wise_or_int8(&p5);
                  }
                  break;
                case PRELU: {
                  ASSERT(!activation_channel_shared);

                  // bmk does not keep eu-align info, user need to update stride if shape changed
                  tl_relu->shape = tl_ofmap[coeff_flip]->shape;
                  tl_relu->stride = ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                  tl_neg->shape = tl_ofmap[coeff_flip]->shape;
                  tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                  // 0. relu = relu(tl_ofmap)
                  // 1. relu = (relu * gt_scale) >> gt_rshift
                  // 2. neg = neg(0, botom)
                  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
                  // 4. tl_ofmap = or relu, neg
                  bmk1880v2_tiu_element_wise_max_param_t p2;
                  p2.max = tl_relu;
                  p2.a = tl_ofmap[coeff_flip];
                  p2.b_is_const = 1;
                  p2.b_is_signed = 1;
                  p2.b_val = 0;
                  p2.layer_id = layer_id;
                  ctx.tiu_element_wise_max(&p2);

                  bmk1880v2_tiu_element_wise_mul_param_t p3;
                  p3.res_high = nullptr;
                  p3.res_low = tl_relu;
                  p3.a = tl_relu;
                  p3.b_val = activation_gt_scale;
                  p3.b_is_signed = true;
                  p3.b_is_const = 1;
                  p3.rshift_bits = activation_gt_rshift;
                  p3.layer_id = layer_id;
                  p3.relu_enable = 0;
                  ctx.tiu_element_wise_mul(&p3);

                  bmk1880v2_tiu_element_wise_min_param_t p4;
                  p4.min = tl_neg;
                  p4.a = tl_ofmap[coeff_flip];
                  p4.b_is_const = 1;
                  p4.b_val = 0;
                  p4.b_is_signed = 1;
                  p4.layer_id = layer_id;
                  ctx.tiu_element_wise_min(&p4);

                  bmk1880v2_tiu_depthwise_convolution_param_t p5;
                  p5.ins_h = 0;
                  p5.ins_last_h = 0;
                  p5.ins_w = 0;
                  p5.ins_last_w = 0;
                  p5.pad_top = 0;
                  p5.pad_bottom = 0;
                  p5.pad_left = 0;
                  p5.pad_right = 0;
                  p5.stride_h = 1;
                  p5.stride_w = 1;
                  p5.dilation_h = 1;
                  p5.dilation_w = 1;
                  p5.ofmap = tl_neg;
                  p5.ifmap = tl_neg;
                  p5.weight = tl_slope[coeff_flip];
                  p5.bias = nullptr;
                  p5.rshift_bits = activation_le_rshift;
                  p5.relu_enable = 0;
                  p5.layer_id = layer_id;
                  ctx.tiu_depthwise_convolution(&p5);

                  bmk1880v2_tiu_element_wise_or_int8_param_t p6;
                  p6.res = tl_ofmap[coeff_flip];
                  p6.a = tl_relu;
                  p6.b = tl_neg;
                  p6.layer_id = layer_id;
                  ctx.tiu_element_wise_or_int8(&p6);
                } break;
                default:
                  ASSERT(0);
              }  // switch (activation_method)
            }    // if (do_activation)

            ga_ofmap_cur[coeff_flip] = ga_ofmap + ig * oc * oh * ow + n_pos * output_c * oh * ow +
                                       oc_pos * oh * ow + oh_top * ow + ow_left;

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int coeff_flip_back = 1 - coeff_flip;

              // Store back to global memory
              DEBUG_BMNET(
                  llvm::errs() << llvm::format(
                      "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] tdma_store_stride:\n"
                      "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                      "%d, %d), stride (%d, %d, %d)\n",
                      ig, oc_pos, n_pos, oh_pos, ow_pos, oc_pos, ga_ofmap_cur[coeff_flip_back],
                      tl_ofmap[coeff_flip_back]->start_address, tl_ofmap[coeff_flip_back]->shape.n,
                      tl_ofmap[coeff_flip_back]->shape.c, tl_ofmap[coeff_flip_back]->shape.h,
                      tl_ofmap[coeff_flip_back]->shape.w, tl_ofmap[coeff_flip_back]->stride.n,
                      tl_ofmap[coeff_flip_back]->stride.c, tl_ofmap[coeff_flip_back]->stride.h,
                      tl_ofmap[coeff_flip_back]->stride.w));

              ctx.tdma_store_stride(tl_ofmap[coeff_flip_back], ga_ofmap_cur[coeff_flip_back],
                                    ofmap_gstride, CTRL_NEURON);
            }

            coeff_flip = 1 - coeff_flip;

          }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

          flip = 1 - flip;

        }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

      }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

    }  // for (int n_i = 0; n_i < n; ni += n_step)

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    int coeff_flip_back = 1 - coeff_flip;

    // Store back to global memory
    DEBUG_BMNET(llvm::errs() << llvm::format(
                    "  [ig=%d] tdma_store_stride:\n"
                    "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                    "%d, %d), stride (%d, %d, %d)\n",
                    ig, ga_ofmap_cur[coeff_flip_back], tl_ofmap[coeff_flip_back]->start_address,
                    tl_ofmap[coeff_flip_back]->shape.n, tl_ofmap[coeff_flip_back]->shape.c,
                    tl_ofmap[coeff_flip_back]->shape.h, tl_ofmap[coeff_flip_back]->shape.w,
                    tl_ofmap[coeff_flip_back]->stride.n, tl_ofmap[coeff_flip_back]->stride.c,
                    tl_ofmap[coeff_flip_back]->stride.h, tl_ofmap[coeff_flip_back]->stride.w));

    ctx.tdma_store_stride(tl_ofmap[coeff_flip_back], ga_ofmap_cur[coeff_flip_back], ofmap_gstride,
                          CTRL_NEURON);

  }  // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias[1]);
    ctx.lmem_free_tensor(tl_scale_bias[0]);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale[1]);
    ctx.lmem_free_tensor(tl_scale[0]);
  }
  if (do_bn) {
    ctx.lmem_free_tensor(tl_bn_mean[1]);
    ctx.lmem_free_tensor(tl_bn_mean[0]);
    ctx.lmem_free_tensor(tl_bn_variance[1]);
    ctx.lmem_free_tensor(tl_bn_variance[0]);
  }
  if (do_activation && activation_method == PRELU) {
    ctx.lmem_free_tensor(tl_slope[1]);
    ctx.lmem_free_tensor(tl_slope[0]);
  }
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    ctx.lmem_free_tensor(tl_relu);
    ctx.lmem_free_tensor(tl_neg);
  }
  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias[1]);
    ctx.lmem_free_tensor(tl_bias[0]);
  }
  ctx.lmem_free_tensor(tl_ofmap[1]);
  ctx.lmem_free_tensor(tl_ofmap[0]);
  ctx.lmem_free_tensor(tl_ifmap[1]);
  ctx.lmem_free_tensor(tl_ifmap[0]);
  ctx.lmem_free_tensor(tl_weight[1]);
  ctx.lmem_free_tensor(tl_weight[0]);

  DEBUG_BMNET(llvm::errs() << "<= ConvReuseActivation" << "/n");
}

void BM1880v2ConvFixedParallelv2::do_conv(const BM1880v2BackendContext &ctx) {
  bool isReuseActivation = false;

  // 1x1 convolution uses the whole input feature map
  // Old parallel conv uses all kernels and get better result for 1x1 kernel.
  // Full weight will be implemented later.
  if (this->kh == 1 && this->kw == 1) {
    isReuseActivation = true;
  }

  if (isReuseActivation) {
    ConvReuseActivation(
        ctx, this->layer_id, this->ga_ifmap, this->ga_ofmap, this->ga_weight, this->ga_bias,
        this->ga_bn_mean, this->ga_bn_variance, this->ga_scale, this->ga_scale_bias, this->input_n,
        this->input_c, this->input_h, this->input_w, this->groups, this->output_c, this->kh,
        this->kw, this->dilation_h, this->dilation_w, this->pad_top, this->pad_bottom,
        this->pad_left, this->pad_right, this->stride_h, this->stride_w, this->do_bias, this->do_bn,
        this->do_scale, this->do_scale_bias, this->do_activation, this->bn_scale, this->bn_eps,
        this->activation_method, this->activation_arg, this->activation_ga_slope,
        this->activation_channel_shared, this->activation_gt_scale, this->activation_gt_rshift,
        this->activation_le_scale, this->activation_le_rshift, this->right_shift_width,
        this->bn_right_shift_width, this->scale_right_shift_width, this->slices);
  } else {
    ConvReuseWeight(
        ctx, this->layer_id, this->ga_ifmap, this->ga_ofmap, this->ga_weight, this->ga_bias,
        this->ga_bn_mean, this->ga_bn_variance, this->ga_scale, this->ga_scale_bias, this->input_n,
        this->input_c, this->input_h, this->input_w, this->groups, this->output_c, this->kh,
        this->kw, this->dilation_h, this->dilation_w, this->pad_top, this->pad_bottom,
        this->pad_left, this->pad_right, this->stride_h, this->stride_w, this->do_bias, this->do_bn,
        this->do_scale, this->do_scale_bias, this->do_activation, this->bn_scale, this->bn_eps,
        this->activation_method, this->activation_arg, this->activation_ga_slope,
        this->activation_channel_shared, this->activation_gt_scale, this->activation_gt_rshift,
        this->activation_le_scale, this->activation_le_rshift, this->right_shift_width,
        this->bn_right_shift_width, this->scale_right_shift_width, this->slices);
  }
}

static BM1880v2ConvFixed *find_best_conv_method(const BM1880v2BackendContext &ctx,
                                                ConvFixed_ARGS &args) {
  BM1880v2ConvFixed *conv_parallelv2 = new BM1880v2ConvFixedParallelv2(args);
  int slices_num_parallelv2 = conv_parallelv2->split(ctx);

  ASSERT(slices_num_parallelv2 != SPLIT_FAILED);
  return conv_parallelv2;
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Borrowed from BM1880v2ConvFixedParallelv2::split
static int BM1880v2DepthwiseConvSplit(const BM1880v2BackendContext &ctx, int input_n, int input_c,
                                      int input_h, int input_w, u16 kh, u16 kw, u16 dilation_h,
                                      u16 dilation_w, u8 pad_top, u8 pad_bottom, u8 pad_left,
                                      u8 pad_right, u8 stride_h, u8 stride_w, int do_bias,
                                      int do_bn, int do_scale, int do_scale_bias, int do_activation,
                                      int activation_method, float activation_arg[],
                                      SLICES &slices) {
  int ic = input_c;
  int oc = input_c;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "BM1880v2ConvFixedParallelv2::split =>\n"
                  "  ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                  "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                  "  stride (%d, %d), dilation (%d, %d)\n",
                  input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw, pad_top,
                  pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h, dilation_w));

  slices.n = 1;
  slices.oc = ceiling_func_shift(oc, NPU_SHIFT);  // lane parallelism
  slices.ic = ic;
  slices.h = (ih + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)
  slices.w = (iw + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)

  int oc_step = (oc >= ctx.hw.npu_num) ? ctx.hw.npu_num : oc;  // use all lanes
  int ic_step = 1;

  // We may need to put EU-alignment info in one place
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i8 = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i16 = ctx.shape_t4(2, oc_step, 1, 1);

  u32 coeff_oc_step_size = 0;
  if (do_bias) {
    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  // prelu needs extra tl_slope compared to leaky relu.
  if (do_activation && activation_method == PRELU) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);
  }

  if (do_bn) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);

    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  if (do_scale) {
    // weight of depthwise conv is aligned
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i8, /*eu_align=*/1);
  }

  if (do_scale_bias) {
    // 16 bit
    coeff_oc_step_size += ctx.lmem_tensor_to_size(coeff_shape_i16, /*eu_align=*/0);
  }

  // Add weight size
  coeff_oc_step_size += ctx.lmem_tensor_to_size(ctx.shape_t4(ic_step, oc_step, kh, kw),
                                                /*eu_align=*/0);

  //
  // Slices may not be a good way to find size
  // We may try to increase or decrease width in aligned with 4, 8, 16 ...
  // or specific height/width (8, 8), (16, 16) ...
  //
  // Split ow
  for (slices.w = 1; slices.w <= ow; ++slices.w) {
    int ow_step = ceiling_func(ow, slices.w);
    int iw_step = math_min((ow_step - 1) * stride_w + kw_extent, iw);

    // Split oh
    for (slices.h = 1; slices.h <= oh; ++slices.h) {
      // split n
      for (slices.n = 1; slices.n <= n; ++slices.n) {
        int n_step = ceiling_func(n, slices.n);

        int oh_step = ceiling_func(oh, slices.h);
        int ih_step = math_min((oh_step - 1) * stride_h + kh_extent, ih);

        u32 total_needed = 0;

        u32 ofmap_size = ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, oc_step, oh_step, ow_step),
                                                 /*eu_align=*/1);
        total_needed += ofmap_size;

        u32 ifmap_size = ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, ic_step, ih_step, iw_step),
                                                 /*eu_align=*/1);
        total_needed += ifmap_size;

        total_needed += coeff_oc_step_size;

        // Double buffers so that TDMA load and store can run during TIU executes.
        total_needed *= 2;

        // Both prelu and leaky relu need tl_neg, tl_relu.
        // tl_relu, tl_neg are not from tmda and not final output.
        // One copy is enough.
        if (do_activation && ((activation_method == PRELU) ||
                              (activation_method == RELU && activation_arg[0] != 0.0f))) {
          total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
        }

        if (total_needed < LOCAL_MEM_SIZE) {
          DEBUG_BMNET(
              llvm::errs() << llvm::format(
                  "  Slices(n=%d, oc=%d, ic=%d, h=%d, w=%d), n_step %d, oh_step %d, ih_step %d"
                  ", coeff_oc_step_size %d, total_needed %d\n",
                  slices.n, slices.oc, slices.ic, slices.h, slices.w, n_step, oh_step, ih_step,
                  coeff_oc_step_size, total_needed));
          DEBUG_BMNET(llvm::errs() << "<= BM1880v2DepthwiseConvSplit succeed" << "/n");
          return total_needed;
        }

      }  // for (slices.n = 1; slices.n < n; ++slices.n)

    }  // for (slices.h = 1; slices.h <= oh; ++slices.h)

  }  // for (slices.w = 1; slices.w <= ow; ++slices.ow)

  llvm::errs() << "BM1880v2DepthwiseConvSplit fail";
  DEBUG_BMNET(llvm::errs() << "<= BM1880v2DepthwiseConvSplit fail" << "/n");

  return SPLIT_FAILED;
}

// Borrowed from BM1880v2ConvFixedParallelv2::do_conv
static void BM1880v2DepthwiseConv(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean, gaddr_t ga_bn_variance,
    gaddr_t ga_scale, gaddr_t ga_scale_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, u16 kh, u16 kw, u16 dilation_h, u16 dilation_w, u8 pad_top,
    u8 pad_bottom, u8 pad_left, u8 pad_right, u8 stride_h, u8 stride_w, int result_add, int do_bias,
    int do_bn, int do_scale, int do_scale_bias, int do_activation, float bn_scale, float bn_eps,
    int activation_method, float activation_arg[], gaddr_t activation_ga_slope,
    bool activation_channel_shared, int activation_gt_scale, int activation_gt_rshift,
    int activation_le_scale, int activation_le_rshift, int right_shift_width,
    int bn_right_shift_width, int scale_right_shift_width) {
  SLICES slices = {.n = 1, .oc = 1, .ic = 1, .h = 1, .w = 1};

  BM1880v2DepthwiseConvSplit(ctx, input_n, input_c, input_h, input_w, kh, kw, dilation_h,
                             dilation_w, pad_top, pad_bottom, pad_left, pad_right, stride_h,
                             stride_w, do_bias, do_bn, do_scale, do_scale_bias, do_activation,
                             activation_method, activation_arg, slices);

  int oc = output_c;
  int ic = 1;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int ic_step = n_step;
  int oh_step = ceiling_func(oh, slices.h);
  int ow_step = ceiling_func(ow, slices.w);
  int ih_step = input_h;
  int iw_step = input_w;
  int oc_step = oc;

  // Always use all lanes.
  // Not divided by slices.oc.
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > ctx.hw.npu_num);
    oc_step = ctx.hw.npu_num;
    ic_step = oc_step;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  bool fused_conv_relu =
      (!do_scale && !do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bool fused_conv_bn_relu =
      (!do_scale && do_bn &&
       (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f)))
          ? true
          : false;

  bmk1880v2_tensor_lmem_shape_t oc_shape_ = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(n_step, ic_step, ih_step, input_w);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(n_step, oc_step, oh_step, ow);

  bmk1880v2_tensor_lmem_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_bn_mean[2] = {nullptr, nullptr},
                          *tl_bn_variance[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_scale[2] = {nullptr, nullptr}, *tl_scale_bias[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_slope[2] = {nullptr, nullptr};
  bmk1880v2_tensor_lmem_t *tl_ifmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_ofmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_neg = nullptr, *tl_relu = nullptr;

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  bmk1880v2_tensor_tgmem_stride_t ofmap_gstride = {static_cast<u32>(output_c) * oh * ow,
                                                   static_cast<u32>(oh) * ow, static_cast<u32>(ow)};
  bmk1880v2_tensor_tgmem_stride_t ifmap_gstride = {static_cast<u32>(input_c) * input_h * input_w,
                                                   static_cast<u32>(input_h) * input_w,
                                                   static_cast<u32>(input_w)};
  bmk1880v2_tensor_tgmem_stride_t bias_gstride = {static_cast<u32>(output_c), 1, 1};
  bmk1880v2_tensor_tgmem_stride_t weight_gstride = {
      static_cast<u32>(oc) * kh * kw * ic, static_cast<u32>(kh) * kw * ic, static_cast<u32>(ic)};

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(ic_step, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_weight[1] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(ic_step, oc_step, kh, kw), FMT_I8, /*eu_align=*/0);
  tl_ifmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic_step, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ifmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic_step, ih_step, iw_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  tl_ofmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                      /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] && tl_ofmap[1]);

  bmk1880v2_tensor_lmem_shape_t coeff_shape_i8 = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t coeff_shape_i16 = ctx.shape_t4(2, oc_step, 1, 1);
  if (do_bias) {
    // 16 bit
    tl_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_aling=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
  }

  // Both prelu and leaky relu needs tl_neg, tl_relu.
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    tl_neg = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                   /*eu_align=*/1);
    tl_relu = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), FMT_I8,
                                    /*eu_align=*/1);
    ASSERT(tl_neg && tl_relu);
  }

  // prelu needs extra tl_slope
  if (do_activation && activation_method == PRELU) {
    // weight of depthwise conv is aligned
    tl_slope[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_slope[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    ASSERT(tl_slope[0] && tl_slope[1]);
  }

  if (do_bn) {
    // weight of depthwise conv is aligned
    tl_bn_variance[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);
    tl_bn_variance[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_align=*/1);

    // 16 bit
    tl_bn_mean[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_bn_mean[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_bn_variance[0] && tl_bn_variance[1] && tl_bn_mean[0] && tl_bn_mean[1]);
  }

  if (do_scale) {
    // weight of depthwise conv is aligned
    tl_scale[0] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    tl_scale[1] = ctx.lmem_alloc_tensor(coeff_shape_i8, FMT_I8, /*eu_algn=*/1);
    ASSERT(tl_scale[0] && tl_scale[1]);
  }

  if (do_scale_bias) {
    // 16 bit
    tl_scale_bias[0] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    tl_scale_bias[1] = ctx.lmem_alloc_tensor(coeff_shape_i16, FMT_I8, /*eu_align=*/0);
    ASSERT(tl_scale_bias[0] && tl_scale_bias[1]);
  }

  // split groups
  for (int ig = 0; ig < 1; ++ig) {
    int first = 1;
    int flip = 0;
    int coeff_flip = 0;
    gaddr_t ga_ofmap_cur[2] = {0};

    ctx.parallel_disable();

    // split oc
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = math_min(oc - oc_pos, oc_step);

      u64 coeff_offset = ig * oc + oc_pos;

      // Actual shape for tdma, tiu
      coeff_shape_i8 = ctx.shape_t4(1, cur_oc, 1, 1);
      coeff_shape_i16 = ctx.shape_t4(2, cur_oc, 1, 1);

      if (do_bias) {
        // 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bias[coeff_flip]->shape = coeff_shape_i16;
        tl_bias[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bias[coeff_flip]->shape, /*eu_aign=*/0);

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "  [ig=%d][oc_pos=%d] tdma_load_stride:\n"
                        "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                        "%d, %d), stride (%d, %d, %d)\n",
                        ig, oc_pos, ga_bias + coeff_offset, tl_bias[coeff_flip]->start_address,
                        tl_bias[coeff_flip]->shape.n, tl_bias[coeff_flip]->shape.c,
                        tl_bias[coeff_flip]->shape.h, tl_bias[coeff_flip]->shape.w, bias_gstride.n,
                        bias_gstride.c, bias_gstride.h));
        ctx.tdma_load_stride(tl_bias[coeff_flip], ga_bias + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      if (do_activation && activation_method == PRELU) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_slope[coeff_flip]->shape = coeff_shape_i8;
        tl_slope[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_slope[coeff_flip]->shape, /*eu_align=*/1);

        ctx.tdma_load(tl_slope[coeff_flip], activation_ga_slope + coeff_offset, CTRL_WEIGHT);
      }

      if (do_bn) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bn_variance[coeff_flip]->shape = coeff_shape_i8;
        tl_bn_variance[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bn_variance[coeff_flip]->shape, /*eu_align=*/1);
        ctx.tdma_load(tl_bn_variance[coeff_flip], ga_bn_variance + coeff_offset, CTRL_WEIGHT);

        // 16 bit
        tl_bn_mean[coeff_flip]->shape = coeff_shape_i16;
        tl_bn_mean[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_bn_mean[coeff_flip]->shape, /*eu_align=*/0);
        ctx.tdma_load_stride(tl_bn_mean[coeff_flip], ga_bn_mean + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      if (do_scale) {
        // weight of depthwise conv is aligned
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale[coeff_flip]->shape = coeff_shape_i8;
        tl_scale[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_scale[coeff_flip]->shape, /*eu_align=*/1);
        ctx.tdma_load(tl_scale[coeff_flip], ga_scale + coeff_offset, CTRL_WEIGHT);
      }

      if (do_scale_bias) {
        // 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale_bias[coeff_flip]->shape = coeff_shape_i16;
        tl_scale_bias[coeff_flip]->stride =
            ctx.tensor_lmem_default_stride(tl_scale_bias[coeff_flip]->shape, /*eu_align=*/0);
        ctx.tdma_load_stride(tl_scale_bias[coeff_flip], ga_scale_bias + coeff_offset, bias_gstride,
                             CTRL_WEIGHT);
      }

      // Weight shape for load != shape for tiu
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_weight[coeff_flip]->shape = ctx.shape_t4(1, cur_oc, kh, kw);
      tl_weight[coeff_flip]->stride =
          ctx.tensor_lmem_default_stride(tl_weight[coeff_flip]->shape, /*eu_aign*/ 1);

      u64 weight_offset = ga_weight + oc_pos * kh * kw;
      {
        // Same local address, different shape, stride
        bmk1880v2_tensor_lmem_t tl_tmp;
        tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
        tl_tmp.fmt = FMT_I8;
        tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic_step);
        tl_tmp.stride = ctx.tensor_lmem_default_stride(tl_tmp.shape, /*eu_align=*/0);

        ctx.tdma_load_stride(&tl_tmp, weight_offset, weight_gstride, CTRL_WEIGHT);
      }

      bmk1880v2_tensor_lmem_shape_t ifmap_shape[2] = {0};
      bmk1880v2_tensor_lmem_shape_t ofmap_shape[2] = {0};
      gaddr_t ga_ifmap_cur[2] = {0};

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = math_min(input_n - n_pos, n_step);

        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = math_min(oh - oh_pos, oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = math_max(oh_top * stride_h - pad_top, 0);
          int ih_bot = math_min((oh_bot - 1) * stride_h + kh_ext - pad_top, input_h);
          int cur_ih = ih_bot - ih_top;

          int ph_top = 0;
          if (ih_top == 0) {
            ph_top = pad_top - oh_top * stride_h;
          }

          int ph_bot = 0;
          if (ih_bot == input_h) {
            ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top - input_h;
          }

          // split w
          for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step) {
            int cur_ow = math_min(ow - ow_pos, ow_step);

            int ow_left = ow_pos;
            int ow_right = ow_left + cur_ow;
            int iw_left = math_max(ow_left * stride_w - pad_left, 0);
            int iw_right = math_min((ow_right - 1) * stride_w + kw_ext - pad_left, input_w);
            int cur_iw = iw_right - iw_left;

            int pw_left = 0;
            if (iw_left == 0) {
              pw_left = pad_left - ow_left * stride_w;
            }

            int pw_right = 0;
            if (iw_right == input_w) {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
            }

            DEBUG_BMNET(
                llvm::errs() << llvm::format("  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d]"
                                          " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                                          ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                                          ig, oc_pos, n_pos, oh_pos, ow_pos, cur_oh, cur_ih, ih_top,
                                          ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ofmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[flip]->stride =
                ctx.tensor_lmem_default_stride(tl_ofmap[flip]->shape, /*eu_aign=*/1);

            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_ih, cur_iw);
            tl_ifmap[flip]->stride =
                ctx.tensor_lmem_default_stride(tl_ifmap[flip]->shape, /*eu_align=*/1);

            u64 ifmap_offset = ga_ifmap + n_pos * input_c * input_h * input_w +
                               oc_pos * input_h * input_w + ih_top * input_w + iw_left;
            ctx.tdma_load_stride(tl_ifmap[flip], ifmap_offset, ifmap_gstride, CTRL_NEURON);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              bmk1880v2_tiu_depthwise_convolution_param_t param;
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = param.ins_last_h = 0;
              param.ins_w = param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pad_left;
              param.pad_right = pad_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.rshift_bits = right_shift_width;
              param.layer_id = layer_id;

              DEBUG_BMNET(llvm::errs() << llvm::format(
                              "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] conv:\n"
                              "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                              "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                              ig, n_pos, oh_pos, ow_pos, param.ifmap->start_address,
                              param.ifmap->shape.n, param.ifmap->shape.c, param.ifmap->shape.h,
                              param.ifmap->shape.w, param.weight->start_address,
                              param.weight->shape.n, param.weight->shape.c, param.weight->shape.h,
                              param.weight->shape.w, param.ofmap->start_address,
                              param.ofmap->shape.n, param.ofmap->shape.c, param.ofmap->shape.h,
                              param.ofmap->shape.w));

              ctx.tiu_depthwise_convolution(&param);
            }

            bmk1880v2_tiu_depthwise_convolution_param_t param;
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
            param.relu_enable = 0;
            param.layer_id = layer_id;
            if (do_bn) {
              // out(n,c,h,w) = in(n,c,h,w) * variance(1,c,1,1) + mean(1,c,1,1)
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_bn_variance[coeff_flip];
              param.bias = tl_bn_mean[coeff_flip];
              param.rshift_bits = bn_right_shift_width;
              param.relu_enable = fused_conv_bn_relu;
              ctx.tiu_depthwise_convolution(&param);
            }
            if (do_scale && do_scale_bias) {
              // computing x * scale + bias
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = tl_scale_bias[coeff_flip];
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale) {
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ofmap[flip];
              param.weight = tl_scale[coeff_flip];
              param.bias = nullptr;
              param.rshift_bits = scale_right_shift_width;
              ctx.tiu_depthwise_convolution(&param);
            } else if (do_scale_bias) {
              ASSERT(0);  // TODO(zakk)
            }

            if (do_activation) {
              switch (activation_method) {
                case RELU:
                  if (activation_arg[0] == 0.0f) {
                    // relu
                    if (!fused_conv_relu && !fused_conv_bn_relu) {
                      // should not come here !
                      ASSERT(0);

                      bmk1880v2_tiu_element_wise_max_param_t p13;
                      p13.max = tl_ofmap[flip];
                      p13.a = tl_ofmap[flip];
                      p13.b_is_const = 1;
                      p13.b_is_signed = 1;
                      p13.b_val = 0;
                      p13.layer_id = layer_id;
                      ctx.tiu_element_wise_max(&p13);
                    }
                  } else {
                    // leaky relu

                    // bmk does not keep eu-align info, user need to update stride if shape changed
                    tl_relu->shape = tl_ofmap[flip]->shape;
                    tl_relu->stride =
                        ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                    tl_neg->shape = tl_ofmap[flip]->shape;
                    tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                    bmk1880v2_tiu_element_wise_max_param_t p1;
                    p1.max = tl_relu;
                    p1.a = tl_ofmap[flip];
                    p1.b_is_const = 1;
                    p1.b_is_signed = 1;
                    p1.b_val = 0;
                    p1.layer_id = layer_id;
                    ctx.tiu_element_wise_max(&p1);

                    bmk1880v2_tiu_element_wise_mul_param_t p2;
                    p2.res_high = nullptr;
                    p2.res_low = tl_relu;
                    p2.a = tl_relu;
                    p2.b_val = activation_gt_scale;
                    p2.b_is_signed = true;
                    p2.b_is_const = 1;
                    p2.rshift_bits = activation_gt_rshift;
                    p2.layer_id = layer_id;
                    p2.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p2);

                    bmk1880v2_tiu_element_wise_min_param_t p3;
                    p3.min = tl_neg;
                    p3.a = tl_ofmap[flip];
                    p3.b_is_const = 1;
                    p3.b_val = 0;
                    p3.b_is_signed = 1;
                    p3.layer_id = layer_id;
                    ctx.tiu_element_wise_min(&p3);

                    bmk1880v2_tiu_element_wise_mul_param_t p4;
                    p4.res_high = nullptr;
                    p4.res_low = tl_neg;
                    p4.a = tl_neg;
                    p4.b_val = activation_le_scale;
                    p4.b_is_signed = true;
                    p4.b_is_const = 1;
                    p4.rshift_bits = activation_le_rshift;
                    p4.layer_id = layer_id;
                    p4.relu_enable = 0;
                    ctx.tiu_element_wise_mul(&p4);

                    bmk1880v2_tiu_element_wise_or_int8_param_t p5;
                    p5.res = tl_ofmap[flip];
                    p5.a = tl_relu;
                    p5.b = tl_neg;
                    p5.layer_id = layer_id;
                    ctx.tiu_element_wise_or_int8(&p5);
                  }
                  break;
                case PRELU: {
                  ASSERT(!activation_channel_shared);

                  // bmk does not keep eu-align info, user need to update stride if shape changed
                  tl_relu->shape = tl_ofmap[flip]->shape;
                  tl_relu->stride = ctx.tensor_lmem_default_stride(tl_relu->shape, /*eu_align=*/1);

                  tl_neg->shape = tl_ofmap[flip]->shape;
                  tl_neg->stride = ctx.tensor_lmem_default_stride(tl_neg->shape, /*eu_align=*/1);

                  // 0. relu = relu(tl_ofmap)
                  // 1. relu = (relu * gt_scale) >> gt_rshift
                  // 2. neg = neg(0, botom)
                  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
                  // 4. tl_ofmap = or relu, neg
                  bmk1880v2_tiu_element_wise_max_param_t p2;
                  p2.max = tl_relu;
                  p2.a = tl_ofmap[flip];
                  p2.b_is_const = 1;
                  p2.b_is_signed = 1;
                  p2.b_val = 0;
                  p2.layer_id = layer_id;
                  ctx.tiu_element_wise_max(&p2);

                  bmk1880v2_tiu_element_wise_mul_param_t p3;
                  p3.res_high = nullptr;
                  p3.res_low = tl_relu;
                  p3.a = tl_relu;
                  p3.b_val = activation_gt_scale;
                  p3.b_is_signed = true;
                  p3.b_is_const = 1;
                  p3.rshift_bits = activation_gt_rshift;
                  p3.layer_id = layer_id;
                  p3.relu_enable = 0;
                  ctx.tiu_element_wise_mul(&p3);

                  bmk1880v2_tiu_element_wise_min_param_t p4;
                  p4.min = tl_neg;
                  p4.a = tl_ofmap[flip];
                  p4.b_is_const = 1;
                  p4.b_val = 0;
                  p4.b_is_signed = 1;
                  p4.layer_id = layer_id;
                  ctx.tiu_element_wise_min(&p4);

                  bmk1880v2_tiu_depthwise_convolution_param_t p5;
                  p5.ins_h = 0;
                  p5.ins_last_h = 0;
                  p5.ins_w = 0;
                  p5.ins_last_w = 0;
                  p5.pad_top = 0;
                  p5.pad_bottom = 0;
                  p5.pad_left = 0;
                  p5.pad_right = 0;
                  p5.stride_h = 1;
                  p5.stride_w = 1;
                  p5.dilation_h = 1;
                  p5.dilation_w = 1;
                  p5.ofmap = tl_neg;
                  p5.ifmap = tl_neg;
                  p5.weight = tl_slope[coeff_flip];
                  p5.bias = nullptr;
                  p5.rshift_bits = activation_le_rshift;
                  p5.relu_enable = 0;
                  p5.layer_id = layer_id;
                  ctx.tiu_depthwise_convolution(&p5);

                  bmk1880v2_tiu_element_wise_or_int8_param_t p6;
                  p6.res = tl_ofmap[flip];
                  p6.a = tl_relu;
                  p6.b = tl_neg;
                  p6.layer_id = layer_id;
                  ctx.tiu_element_wise_or_int8(&p6);
                } break;
                default:
                  ASSERT(0);
              }  // switch (activation_method)
            }    // if (do_activation)

            ga_ofmap_cur[flip] =
                ga_ofmap + n_pos * output_c * oh * ow + oc_pos * oh * ow + oh_top * ow + ow_left;

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int flip_back = 1 - flip;

              // Store back to global memory
              ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_gstride,
                                    CTRL_NEURON);
            }

            flip = 1 - flip;

          }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

        }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      }  // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    int flip_back = 1 - flip;

    // Store back to global memory
    ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_gstride, CTRL_NEURON);

  }  // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias[1]);
    ctx.lmem_free_tensor(tl_scale_bias[0]);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale[1]);
    ctx.lmem_free_tensor(tl_scale[0]);
  }
  if (do_bn) {
    ctx.lmem_free_tensor(tl_bn_mean[1]);
    ctx.lmem_free_tensor(tl_bn_mean[0]);
    ctx.lmem_free_tensor(tl_bn_variance[1]);
    ctx.lmem_free_tensor(tl_bn_variance[0]);
  }
  if (do_activation && activation_method == PRELU) {
    ctx.lmem_free_tensor(tl_slope[1]);
    ctx.lmem_free_tensor(tl_slope[0]);
  }
  if (do_activation && ((activation_method == PRELU) ||
                        (activation_method == RELU && activation_arg[0] != 0.0f))) {
    ctx.lmem_free_tensor(tl_relu);
    ctx.lmem_free_tensor(tl_neg);
  }
  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias[1]);
    ctx.lmem_free_tensor(tl_bias[0]);
  }
  ctx.lmem_free_tensor(tl_ofmap[1]);
  ctx.lmem_free_tensor(tl_ofmap[0]);
  ctx.lmem_free_tensor(tl_ifmap[1]);
  ctx.lmem_free_tensor(tl_ifmap[0]);
  ctx.lmem_free_tensor(tl_weight[1]);
  ctx.lmem_free_tensor(tl_weight[0]);
}

void bmnet_conv_parallel_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx, u32 stream_id, u32 inst_id, u32 layer_id, const u32 *depends,
    u32 depends_len, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    gaddr_t ga_bn_mean, gaddr_t ga_bn_variance, gaddr_t ga_scale, gaddr_t ga_scale_bias,
    int input_n, int input_c, int input_h, int input_w, int groups, int output_c, u16 kh, u16 kw,
    u16 dilation_h, u16 dilation_w, u8 pad_top, u8 pad_bottom, u8 pad_left, u8 pad_right,
    u8 stride_h, u8 stride_w, int result_add, int do_bias, int do_bn, int do_scale,
    int do_scale_bias, int do_activation, float bn_scale, float bn_eps, int activation_method,
    float activation_arg[], gaddr_t activation_ga_slope, bool activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift,
    int activation_le_scale,  // slope, TODO
    int activation_le_rshift, int right_shift_width, int bn_right_shift_width,
    int scale_right_shift_width, bool use_winograd, int threshold_x_quantized_len,
    const int *threshold_x_quantized, const int *right_shift_array) {
  int nsec = ceiling_func_shift(input_n, NODECHIP_SHIFT);

  // this message is too long for llvm::format, so seperate it
  DEBUG_BMNET(llvm::errs() << llvm::format("bmnet_conv_parallel_fixed_forward_bmkernel:\n"
                                        "    layer_id %d\n"
                                        "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                                        "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                                        "    kernel = (%d, %d), dilation = (%d, %d)\n"
                                        "    pad = (%d, %d,%d, %d), stride = (%d, %d)\n",
                                        layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, nsec,
                                        input_c, input_h, input_w, groups, output_c, kh, kw,
                                        dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
                                        pad_right, stride_h, stride_w));
  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "    result_add = %d, do_bias = %d, do_bn = %d, do_scale = %d\n"
                  "    do_scale_bias = %d, do_activation = %d, activation_method %d\n",
                  result_add, do_bias, do_bn, do_scale, do_scale_bias, do_activation,
                  activation_method));
  // for (int i = 0; i < threshold_x_quantized_len; i++) {
  //  VLOG(3) << "threshold_x_quantized/right_shift_array[" << i << "]:" << threshold_x_quantized[i]
  //          << "/" << right_shift_array[i];
  //}

  if (input_c == groups && output_c == groups && 1 != groups) {
    return BM1880v2DepthwiseConv(
        ctx, layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, ga_bn_mean, ga_bn_variance, ga_scale,
        ga_scale_bias, input_n, input_c, input_h, input_w, groups, output_c, kh, kw, dilation_h,
        dilation_w, pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, result_add,
        do_bias, do_bn, do_scale, do_scale_bias, do_activation, bn_scale, bn_eps, activation_method,
        activation_arg, activation_ga_slope, activation_channel_shared, activation_gt_scale,
        activation_gt_rshift, activation_le_scale, activation_le_rshift, right_shift_width,
        bn_right_shift_width, scale_right_shift_width);
  }

  ConvFixed_ARGS args = {
      .ga_ifmap = ga_ifmap,
      .ga_ofmap = ga_ofmap,
      .ga_weight = ga_weight,
      .ga_bias = ga_bias,
      .ga_bn_mean = ga_bn_mean,
      .ga_bn_variance = ga_bn_variance,
      .ga_scale = ga_scale,
      .ga_scale_bias = ga_scale_bias,
      .input_n = nsec,
      .input_c = input_c,
      .input_h = input_h,
      .input_w = input_w,
      .groups = groups,
      .output_c = output_c,
      .kh = kh,
      .kw = kw,
      .dilation_h = dilation_h,
      .dilation_w = dilation_w,
      .pad_top = pad_top,
      .pad_bottom = pad_bottom,
      .pad_left = pad_left,
      .pad_right = pad_right,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .result_add = static_cast<bool>(result_add),
      .do_bias = static_cast<bool>(do_bias),
      .do_bn = static_cast<bool>(do_bn),
      .do_scale = static_cast<bool>(do_scale),
      .do_scale_bias = static_cast<bool>(do_scale_bias),
      .do_activation = static_cast<bool>(do_activation),
      .bn_scale = bn_scale,
      .bn_eps = bn_eps,
      .activation_method = activation_method,
      .activation_arg = activation_arg,
      .activation_ga_slope = activation_ga_slope,
      .activation_channel_shared = activation_channel_shared,
      .activation_gt_scale = activation_gt_scale,
      .activation_gt_rshift = activation_gt_rshift,
      .activation_le_scale = activation_le_scale,
      .activation_le_rshift = activation_le_rshift,
      .right_shift_width = right_shift_width,
      .bn_right_shift_width = bn_right_shift_width,
      .scale_right_shift_width = scale_right_shift_width,
      .layer_id = layer_id,
  };

  BM1880v2ConvFixed *conv = find_best_conv_method(ctx, args);
  conv->do_conv(ctx);
  delete conv;
}
