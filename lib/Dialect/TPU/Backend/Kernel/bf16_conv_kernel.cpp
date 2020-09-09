/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_conv_kernel.cpp
 * Description:
 */
#include "bf16_conv_kernel.h"
#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_convbf16"
#define DEBUG_SPLIT "bmnet_bm1880v2_bmkernel_convbf16_split"

#define ASSERT(x) assert(x)

#define RELU 0
#define PRELU 1

#define SPLIT_FAILED 0xFFFF

// namespace bmnet {
int BM1880v2ConvBF16::_split_oc(const CviBackendContext &ctx, int ic_step) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  int oc_step = oc;
  cvk_tl_t *o = NULL, *_ifmap = NULL, *_weight = NULL, *_bias = NULL;
  bool o_alloc = false, w_alloc = false;

  // try to not slice ic
  // we slice oc as 1st
  // NOTICE: we assume slice to n = 1
  // TODO: try to slice h / w
  int n_step = 1;
  _ifmap = conv_ifmap_tensor(ctx, n_step, ic, input_h, input_w, ic_step, fmt);
  if (!_ifmap) {
    return 0;
  }

  if (do_bias) {
    _bias = conv_bias_tensor(ctx, oc, fmt);
  }

  // find proper oc_step/ic_step could alloc success
  do {
    _weight = conv_weight_tensor(ctx, kh, kw, ic_step, oc_step, oc, ic, fmt);

    w_alloc = _weight != NULL;
    if (w_alloc) {
      o = conv_ofmap_tensor(ctx, n_step, oh, ow, oc, oc_step, fmt);
      o_alloc = o != NULL;
      if (o_alloc) {
        ctx.lmem_free_tensor(o);
      }
      ctx.lmem_free_tensor(_weight);
    }
  } while (!(o_alloc && w_alloc) && --oc_step);

  if (do_bias) {
    ctx.lmem_free_tensor(_bias);
  }

  ctx.lmem_free_tensor(_ifmap);

  if (oc_step < 0) {
    oc_step = SPLIT_FAILED;
  }

  return oc_step;
}

int BM1880v2ConvBF16::_split_ic(const CviBackendContext &ctx, int ic_step, int oc_step) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  cvk_tl_t *o = NULL, *_ifmap = NULL, *_weight = NULL, *_bias = NULL;
  bool o_alloc = false, w_alloc = false;

  // try to not slice ic
  // we slice oc as 1st
  // NOTICE: we assume slice to n = 1
  // TODO: try to slice h / w
  int n_step = 1;
  _ifmap = conv_ifmap_tensor(ctx, n_step, ic, input_h, input_w, ic_step, fmt);
  if (!_ifmap) {
    return 0;
  }

  if (do_bias) {
    _bias = conv_bias_tensor(ctx, oc, fmt);
  }

  // find proper oc_step/ic_step could alloc success
  do {
    _weight = conv_weight_tensor(ctx, kh, kw, ic_step, oc_step, oc, ic, fmt);

    w_alloc = _weight != NULL;
    if (w_alloc) {
      o = conv_ofmap_tensor(ctx, n_step, oh, ow, oc, oc_step, fmt);
      o_alloc = o != NULL;
      if (o_alloc) {
        ctx.lmem_free_tensor(o);
      }
      ctx.lmem_free_tensor(_weight);
    }
  } while (0);

  if (do_bias) {
    ctx.lmem_free_tensor(_bias);
  }

  ctx.lmem_free_tensor(_ifmap);

  return w_alloc && o_alloc;
}

/**
 * FIXME: split ih/iw
 */
int BM1880v2ConvBF16::split_ic(const CviBackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;

  cvk_fmt_t fmt = CVK_FMT_BF16;
  int ic_step = ic;
  int oc_step = -1;
  // Always use all lanes.
  // Not divided by slices.oc.
  // E.g. mtcnn_det2_cic oc = 48, slices.oc = 2
  // It is better to store step.
  oc_step = oc > NPU_NUM ? NPU_NUM : oc;

  for (; ic_step >= 0; ic_step--) {
    // oc_step = _split_oc(ctx, ic_step);
    // if (oc_step > 0)
    if (_split_ic(ctx, ic_step, oc_step)) {
      break;
    }
  }

  // ic_step = 40;

  // assert(ic_step != ic && "no need to slice ic");

  // TODO: review slice fail case
  assert(oc_step > 0 && ic_step > 0 && "not found proper slice ic/oc");

  // round down to even for enable 'double conv'
  ic_step = (ic_step / 2) * 2;
  LLVM_DEBUG(llvm::errs() << llvm::format("ic_step %d oc_step %d\n", ic_step, oc_step));

  int n_step = 1;
  slices.ic_step = ic_step;
  slices.oc_step = oc_step;
  slices.oh_step = oh;
  slices.ow_step = ow;
  slices.ih_step = ih;
  slices.iw_step = iw;

  uint32_t coeff_oc_step_size = 0;
  uint32_t total_needed = 0;

  if (do_bias) {
    // 2x 16bit
    coeff_oc_step_size +=
        ctx.lmem_tensor_to_size(ctx.shape_t4(2, oc_step, 1, 1), fmt, /*eu_align=*/0);
  }

  // Add weight size
  coeff_oc_step_size +=
      ctx.lmem_tensor_to_size(ctx.shape_t4(ic, oc_step, kh, kw), fmt, /*eu_align=*/0);

  uint32_t ofmap_size =
      ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, oc_step, oh, ow), fmt, /*eu_align=*/1);

  total_needed += ofmap_size;

  uint32_t ifmap_size =
      ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, ic_step, ih, iw), fmt, /*eu_align=*/1);
  total_needed += ifmap_size;

  total_needed += coeff_oc_step_size;

  return total_needed;
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Not split ic since it needs 32b ofmap for partial sum.
// \return total need
int BM1880v2ConvBF16::_split(const CviBackendContext &ctx, int duplicate_weights) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  // Depthwise
  bool isDepthWise =
      (input_c == groups && output_c == groups && 1 != groups) ? true : false;
  if (isDepthWise) {
    ic = input_c;
    oc = output_c;
  }

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "BM1880v2ConvBF16::split =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n",
                 groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw,
                 pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h,
                 dilation_w));

  slices.n = 1;
  slices.oc = ceiling_func(oc, NPU_NUM); // lane parallelism
  // slices.ic = isDepthWise ? ic : 1;
  slices.ic = 1;
  slices.h = (ih + (4095 - 32 - 1)) / (4095 - 32); // 12bit, max 4095-32(lanes)
  slices.w = (iw + (4095 - 32 - 1)) / (4095 - 32); // 12bit, max 4095-32(lanes)

  int ic_step = isDepthWise ? 1 : ic;
  int num_oc_step = 1;

  //
  // Slices may not be a good way to find size
  // We may try to increase or decrease width in aligned with 4, 8, 16 ...
  // or specific height/width (8, 8), (16, 16) ...
  //
  // Split ow
  if (is_split_ic()) {
    LLVM_DEBUG(llvm::errs() << "<= slice ic(" << ic << ")\n";);
    return split_ic(ctx);
  }

  if (is_split_oc()) {
    LLVM_DEBUG(llvm::errs() << "<= slice oc\n";);
    num_oc_step = (oc + NPU_NUM - 1) / NPU_NUM;
  }

  // TODO: suppot slice kernel
  // 'iw / slices.w >= kw_extent' means we CANT slice kernel
  for (slices.w = 1; slices.w <= ow && iw / slices.w >= kw_extent; ++slices.w) {
    int ow_step = ceiling_func(ow, slices.w);
    int iw_step = std::min((ow_step - 1) * stride_w + kw_extent, (int)iw);

    if ((slices.w == 1) && (stride_w > 1)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, (int)iw);
      slices.iw_step = iw_step;
    }

    // Split oh
    // TODO: support slice kernel
    for (slices.h = 1; slices.h <= oh && ih / slices.h >= kh_extent; ++slices.h) {
      // Split oc
      // TODO: config not split it
      for (int slice_oc = 0; slice_oc < num_oc_step; ++slice_oc) {
        // Downward, align lanes
        //   E.g. oc = 48, oc_step: 48, 32
        int oc_step = std::min((num_oc_step - slice_oc) * NPU_NUM, (int)oc);
        if (num_oc_step == 1) {
          // FIXME: not check every loop
          oc_step = oc;
          slices.oc = 1;
        }

        uint32_t coeff_oc_step_size = 0;

        if (do_bias) {
          // 2x 16bit
          coeff_oc_step_size += ctx.lmem_tensor_to_size(ctx.shape_t4(2, oc_step, 1, 1),
                                                        CVK_FMT_BF16, /*eu_align=*/0);
        }

        // TODO: handle prelu

        // Add weight size
        if (isDepthWise) {
          coeff_oc_step_size += ctx.lmem_tensor_to_size(ctx.shape_t4(1, oc_step, kh, kw),
                                                        CVK_FMT_BF16, /*eu_align=*/0);
        } else {
          coeff_oc_step_size += ctx.lmem_tensor_to_size(
              ctx.shape_t4(ic_step, oc_step, kh, kw), CVK_FMT_BF16, /*eu_align=*/0);
        }
        // split n
        for (slices.n = 1; slices.n <= n; ++slices.n) {
          int n_step = ceiling_func(n, slices.n);

          int oh_step = ceiling_func(oh, slices.h);
          int ih_step = std::min((oh_step - 1) * stride_h + kh_extent, (int)ih);

          uint32_t total_needed = 0;

          uint32_t ofmap_size =
              ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, oc_step, oh_step, ow_step),
                                      CVK_FMT_BF16, /*eu_align=*/1);

          total_needed += ofmap_size;

          uint32_t ifmap_size = 0;
          if (isDepthWise) {
            ifmap_size =
                ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, oc_step, ih_step, iw_step),
                                        CVK_FMT_BF16, /*eu_align=*/1);
          } else {
            ifmap_size =
                ctx.lmem_tensor_to_size(ctx.shape_t4(n_step, ic_step, ih_step, iw_step),
                                        CVK_FMT_BF16, /*eu_align=*/1);
          }

          total_needed += ifmap_size;

          total_needed += coeff_oc_step_size;

          // Double buffers so that TDMA load and store can run during TIU executes.
          total_needed *= duplicate_weights;

          // TODO: handle prelu, leaky relu
          // Both prelu and leaky relu need tl_neg, tl_relu.
          // tl_relu, tl_neg are not from tmda and not final output.
          // One copy is enough.
          // if (do_activation && ((activation_method == PRELU) ||
          //                       (activation_method == RELU && activation_arg &&
          //                       activation_arg[0] != 0.0f))) {
          //   total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
          // }

          if (total_needed < (uint32_t)LOCAL_MEM_SIZE) {
            slices.ic_step = ic_step;
            slices.oc_step = oc_step;
            slices.oh_step = oh_step;
            slices.ow_step = ow_step;
            slices.ih_step = ih_step;
            slices.iw_step = iw_step;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  Slices(n=%d, oc=%d, ic=%d, h=%d, w=%d), n_step %d, oh_step "
                           "%d, ih_step %d"
                           ", coeff_oc_step_size %d, total_needed %d\n",
                           slices.n, slices.oc, slices.ic, slices.h, slices.w, n_step,
                           oh_step, ih_step, coeff_oc_step_size, total_needed););
            LLVM_DEBUG(llvm::errs() << "<= BM1880v2ConvFixedParallelv2::split succeed"
                                    << "\n");
            return total_needed;
          }

        } // for (slices.n = 1; slices.n < n; ++slices.n)

      } // for (int slice_oc = 0; slice_oc < num_oc_step; ++slice_oc)

    } // for (slices.h = 1; slices.h <= oh; ++slices.h)

  } // for (slices.w = 1; slices.w <= ow; ++slices.ow)

  LLVM_DEBUG(llvm::errs() << "<= BM1880v2ConvBF16::split fail"
                          << "\n");

  return SPLIT_FAILED;
}

/**
 */
int BM1880v2ConvBF16::split(const CviBackendContext &ctx) {

  // dw means duplicate_weights
  int dw = 2;
  uint32_t total_needed = 0;
  policy = POLICY_NO_SPLIT;
  if (SPLIT_FAILED != (total_needed = _split(ctx, dw))) {
    LLVM_DEBUG(llvm::errs() << "policy: POLICY_NO_SPLIT success\n");
    return total_needed;
  }

  policy = POLICY_SPLIT_OC;
  if (SPLIT_FAILED != (total_needed = _split(ctx, dw))) {
    LLVM_DEBUG(llvm::errs() << "policy: POLICY_SPLIT_OC success\n");
    return total_needed;
  }

  dw = 1;
  policy = POLICY_NO_REUSE_WEIGHT | POLICY_SPLIT_OC;
  if (SPLIT_FAILED != (total_needed = _split(ctx, dw))) {
    LLVM_DEBUG(llvm::errs() <<
                   "policy: POLICY_NO_REUSE_WEIGHT + POLICY_SPLIT_OC success\n");
    return total_needed;
  }

  dw = 1;
  policy = POLICY_SPLIT_IC;
  if (SPLIT_FAILED != (total_needed = _split(ctx, dw))) {
    LLVM_DEBUG(llvm::errs() << "policy: POLICY_SPLIT_IC success\n");
    return total_needed;
  }

  return SPLIT_FAILED;
}

void BM1880v2ConvBF16::ConvReuseActivation(const CviBackendContext &ctx) {
  uint32_t ic = input_c / groups;
  uint32_t oc = output_c / groups;
  uint32_t kh_ext = dilation_h * (kh - 1) + 1;
  uint32_t kw_ext = dilation_w * (kw - 1) + 1;
  uint32_t oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  uint32_t ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  uint32_t n_step = ceiling_func(input_n, slices.n);
  uint32_t oh_step = ceiling_func(oh, slices.h);
  uint32_t ow_step = ceiling_func(ow, slices.w);
  uint32_t ih_step = input_h;
  uint32_t iw_step = input_w;
  uint32_t oc_step = oc;

  // Always use all lanes.
  // Not divided by slices.oc.
  // E.g. mtcnn_det2_cic oc = 48, slices.oc = 2
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > (uint32_t)NPU_NUM);
    oc_step = NPU_NUM;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "ConvReuseActivation =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n"
                 "  Slices (n=%d, oc=%d, ic=%d, h=%d, w=%d)\n",
                 groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw,
                 pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h,
                 dilation_w, slices.n, slices.oc, slices.ic, slices.h, slices.w));

  bool fused_conv_relu = (!do_scale && !do_bn &&
                          (do_activation && activation_method == RELU &&
                           (!activation_arg || (activation_arg[0] == 0.0f))))
                             ? true
                             : false;

  cvk_tl_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  cvk_tl_t *tl_ifmap[2] = {nullptr};
  cvk_tl_t *tl_ofmap[2] = {nullptr};

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  cvk_tg_stride_t ofmap_gstride = {static_cast<uint32_t>(output_c) * oh * ow,
                                   static_cast<uint32_t>(oh) * ow,
                                   static_cast<uint32_t>(ow)};
  cvk_tg_stride_t ifmap_gstride = {static_cast<uint32_t>(input_c) * input_h * input_w,
                                   static_cast<uint32_t>(input_h) * input_w,
                                   static_cast<uint32_t>(input_w)};
  cvk_tg_stride_t bias_gstride = {static_cast<uint32_t>(output_c), 1, 1};
  cvk_tg_stride_t weight_gstride = {static_cast<uint32_t>(oc) * kh * kw * ic,
                                    static_cast<uint32_t>(kh) * kw * ic,
                                    static_cast<uint32_t>(ic)};

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), CVK_FMT_BF16,
                                       /*eu_align=*/0);
  tl_weight[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), CVK_FMT_BF16,
                                       /*eu_align=*/0);
  tl_ifmap[0] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ifmap[1] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ofmap[0] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ofmap[1] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] &&
         tl_ofmap[1]);

  if (do_bias) {
    // 2 x 16bit
    tl_bias[0] = ctx.lmem_alloc_tensor({2, oc, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    tl_bias[1] = ctx.lmem_alloc_tensor({2, oc, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
  }

  // TODO: prelu, leaky relu

  // split groups
  for (int ig = 0; ig < groups; ++ig) {
    int first = 1;
    int flip = 0;
    int coeff_flip = 0;
    gaddr_t ga_ofmap_cur[2] = {0};

    ctx.parallel_disable();

    // split n
    for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
      int cur_n = std::min(input_n - n_pos, (int)n_step);

      // split h
      for (int oh_pos = 0; oh_pos < (int)oh; oh_pos += oh_step) {
        int cur_oh = (int)std::min(oh - oh_pos, oh_step);

        int oh_top = oh_pos;
        int oh_bot = oh_top + cur_oh;
        int ih_top = std::max(oh_top * stride_h - pad_top, 0);
        int ih_bot = std::min((int)((oh_bot - 1) * stride_h + kh_ext - pad_top), input_h);
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
        for (int ow_pos = 0; ow_pos < (int)ow; ow_pos += ow_step) {
          int cur_ow = (int)std::min(ow - ow_pos, ow_step);

          int ow_left = ow_pos;
          int ow_right = ow_left + cur_ow;
          int iw_left = std::max(ow_left * stride_w - pad_left, 0);
          int iw_right = (int)std::min((int)((ow_right - 1) * stride_w + kw_ext - pad_left), input_w);
          int cur_iw = iw_right - iw_left;

          // bmk does not keep eu-align info, user need to update stride if shape changed
          tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, ic, cur_ih, cur_iw);
          tl_ifmap[flip]->stride =
              ctx.tl_default_stride(tl_ifmap[flip]->shape, CVK_FMT_BF16, /*eu_align=*/1);

          uint64_t ifmap_offset =
              (ig * ic * input_h * input_w + n_pos * input_c * input_h * input_w +
               ih_top * input_w + iw_left) * sizeof(uint16_t);

          LLVM_DEBUG(
              llvm::errs() << llvm::format(
                  "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] tdma_load_stride_bf16:\n"
                  "    tl_ifmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                  "%d, %d), gstride (%d, %d, %d)\n",
                  ig, n_pos, oh_pos, ow_pos, ifmap_offset, tl_ifmap[flip]->start_address,
                  tl_ifmap[flip]->shape.n, tl_ifmap[flip]->shape.c,
                  tl_ifmap[flip]->shape.h, tl_ifmap[flip]->shape.w, ifmap_gstride.n,
                  ifmap_gstride.c, ifmap_gstride.h));

          ctx.tdma_load_stride_bf16(tl_ifmap[flip], ga_ifmap + ifmap_offset,
                                    ifmap_gstride);

          // split oc
          for (int oc_pos = 0; oc_pos < (int)oc; oc_pos += oc_step) {
            int cur_oc = (int)std::min(oc - oc_pos, oc_step);

            uint64_t coeff_offset = (ig * oc + oc_pos) * sizeof(uint16_t);

            if (do_bias) {
              // 16 bit
              // bmk does not keep eu-align info, user need to update stride if shape
              // changed
              tl_bias[coeff_flip]->shape = {2, (uint32_t)cur_oc, 1, 1};
              tl_bias[coeff_flip]->stride = ctx.tl_default_stride(
                  tl_bias[coeff_flip]->shape, CVK_FMT_BF16, /*eu_align=*/0);

              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] "
                             "tdma_load_stride_bf16:\n"
                             "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                             "%d, %d), gstride (%d, %d, %d)\n",
                             ig, n_pos, oh_pos, ow_pos, oc_pos, ga_bias + coeff_offset,
                             tl_bias[coeff_flip]->start_address,
                             tl_bias[coeff_flip]->shape.n, tl_bias[coeff_flip]->shape.c,
                             tl_bias[coeff_flip]->shape.h, tl_bias[coeff_flip]->shape.w,
                             bias_gstride.n, bias_gstride.c, bias_gstride.h));
              ctx.tdma_load_stride_bf16(tl_bias[coeff_flip], ga_bias + coeff_offset,
                                        bias_gstride);
            }

            // Weight shape for load != shape for tiu
            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_weight[coeff_flip]->shape = ctx.shape_t4(ic, cur_oc, kh, kw);
            tl_weight[coeff_flip]->stride = ctx.tl_default_stride(
                tl_weight[coeff_flip]->shape, CVK_FMT_BF16, /*eu_align*/ 0);

            uint64_t weight_offset =
                (ig * oc * ic * kh * kw + oc_pos * ic * kh * kw) * sizeof(uint16_t);
            {
              // Same local address, different shape, stride
              cvk_tl_t tl_tmp;
              tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
              tl_tmp.fmt = CVK_FMT_BF16;
              tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic);
              tl_tmp.stride =
                  ctx.tl_default_stride(tl_tmp.shape, CVK_FMT_BF16, /*eu_align=*/0);

              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] "
                             "tdma_load_stride_bf16:\n"
                             "    tl_weight gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                             "%d, %d), gstride (%d, %d, %d)\n",
                             ig, n_pos, oh_pos, ow_pos, oc_pos, weight_offset,
                             tl_tmp.start_address, tl_tmp.shape.n, tl_tmp.shape.c,
                             tl_tmp.shape.h, tl_tmp.shape.w, weight_gstride.n,
                             weight_gstride.c, weight_gstride.h));

              ctx.tdma_load_stride_bf16(&tl_tmp, ga_weight + weight_offset,
                                        weight_gstride);
            }

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d]"
                           " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                           ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                           ig, n_pos, oh_pos, ow_pos, oc_pos, cur_oh, cur_ih, ih_top,
                           ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_ofmap[coeff_flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[coeff_flip]->stride = ctx.tl_default_stride(
                tl_ofmap[coeff_flip]->shape, CVK_FMT_BF16, /*eu_align=*/1);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              cvk_tiu_pt_convolution_param_t param = {0};
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
              param.ps32_mode = 0;
              param.w_is_const = 0;
              param.layer_id = layer_id;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
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

              ctx.tiu_pt_convolution(&param);
            }

            ga_ofmap_cur[coeff_flip] =
                ga_ofmap + (ig * oc * oh * ow + n_pos * output_c * oh * ow +
                            oc_pos * oh * ow + oh_top * ow + ow_left) *
                               sizeof(uint16_t);

            if (first) {
              // postpone first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int coeff_flip_back = 1 - coeff_flip;

              // Store back to global memory
              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d][oc_pos=%d] "
                             "tdma_store_stride_bf16:\n"
                             "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                             "%d, %d), gstride (%d, %d, %d)\n",
                             ig, oc_pos, n_pos, oh_pos, ow_pos, oc_pos,
                             ga_ofmap_cur[coeff_flip_back],
                             tl_ofmap[coeff_flip_back]->start_address,
                             tl_ofmap[coeff_flip_back]->shape.n,
                             tl_ofmap[coeff_flip_back]->shape.c,
                             tl_ofmap[coeff_flip_back]->shape.h,
                             tl_ofmap[coeff_flip_back]->shape.w, ofmap_gstride.n,
                             ofmap_gstride.c, ofmap_gstride.h));

              ctx.tdma_store_stride_bf16(tl_ofmap[coeff_flip_back],
                                         ga_ofmap_cur[coeff_flip_back], ofmap_gstride);
            }

            coeff_flip = 1 - coeff_flip;

          } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

          flip = 1 - flip;

        } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

      } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

    } // for (int n_i = 0; n_i < n; ni += n_step)

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    int coeff_flip_back = 1 - coeff_flip;

    // Store back to global memory
    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "  [ig=%d] tdma_store_stride_bf16:\n"
                   "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                   "%d, %d), gstride (%d, %d, %d)\n",
                   ig, ga_ofmap_cur[coeff_flip_back],
                   tl_ofmap[coeff_flip_back]->start_address,
                   tl_ofmap[coeff_flip_back]->shape.n, tl_ofmap[coeff_flip_back]->shape.c,
                   tl_ofmap[coeff_flip_back]->shape.h, tl_ofmap[coeff_flip_back]->shape.w,
                   ofmap_gstride.n, ofmap_gstride.c, ofmap_gstride.h));

    ctx.tdma_store_stride_bf16(tl_ofmap[coeff_flip_back], ga_ofmap_cur[coeff_flip_back],
                               ofmap_gstride);

  } // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
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

  LLVM_DEBUG(llvm::errs() << "<= ConvReuseActivation"
                          << "\n");
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
void BM1880v2ConvBF16::ConvReuseWeight(const CviBackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int oh_step = slices.oh_step;
  int ow_step = slices.ow_step;
  int ih_step = slices.ih_step;
  int iw_step = slices.iw_step;
  int oc_step = slices.oc_step;

  // Always use all lanes.
  // Not divided by slices.oc.
  // E.g. mtcnn_det2_cic oc = 48, slices.oc = 2
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > NPU_NUM);
    oc_step = NPU_NUM;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "ConvReuseWeight =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n"
                 "  Slices (n=%d, oc=%d, ic=%d, h=%d, w=%d)\n",
                 groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw,
                 pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h,
                 dilation_w, slices.n, slices.oc, slices.ic, slices.h, slices.w));

  bool fused_conv_relu = (!do_scale && !do_bn &&
                          (do_activation && activation_method == RELU &&
                           (!activation_arg || (activation_arg[0] == 0.0f))))
                             ? true
                             : false;

  cvk_tl_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  cvk_tl_t *tl_ifmap[2] = {nullptr};
  cvk_tl_t *tl_ofmap[2] = {nullptr};

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  // cvk_tg_stride_t ofmap_gstride = {static_cast<uint32_t>(output_c) * oh * ow,
  //                                                  static_cast<uint32_t>(oh) * ow,
  //                                                  static_cast<uint32_t>(ow)};
  // cvk_tg_stride_t ifmap_gstride = {static_cast<uint32_t>(input_c) * input_h * input_w,
  //                                                  static_cast<uint32_t>(input_h) *
  //                                                  input_w,
  //                                                  static_cast<uint32_t>(input_w)};
  // cvk_tg_stride_t bias_gstride = {static_cast<uint32_t>(output_c), 1, 1};
  // cvk_tg_stride_t weight_gstride = {
  //     static_cast<uint32_t>(oc) * kh * kw * ic, static_cast<uint32_t>(kh) * kw * ic,
  //     static_cast<uint32_t>(ic)};
  cvk_tg_stride_t ofmap_gstride =
      ctx.tg_default_stride({1, (uint32_t)output_c, (uint32_t)oh, (uint32_t)ow}, CVK_FMT_BF16);
  cvk_tg_stride_t ifmap_gstride =
      ctx.tg_default_stride({1, (uint32_t)input_c, (uint32_t)input_h, (uint32_t)input_w}, CVK_FMT_BF16);
  cvk_tg_stride_t bias_gstride = ctx.tg_default_stride({1, (uint32_t)output_c, 1, 1}, CVK_FMT_BF16);
  cvk_tg_stride_t weight_gstride =
      ctx.tg_default_stride({1, (uint32_t)oc, (uint32_t)(kh * kw), (uint32_t)ic}, CVK_FMT_BF16);

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), CVK_FMT_BF16,
                                       /*eu_align=*/0);
  if (is_reuse_weight()) {
    tl_weight[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(ic, oc_step, kh, kw), CVK_FMT_BF16,
                                         /*eu_align=*/0);
  } else {
    // tl_weight[1] = tl_weight[0];
  }

  tl_ifmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step),
                                      CVK_FMT_BF16, /*eu_align=*/1);

  if (is_reuse_weight()) {
    tl_ifmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, ic, ih_step, iw_step),
                                        CVK_FMT_BF16, /*eu_align=*/1);
  } else {
    // tl_ifmap[1] = tl_ifmap[0];
  }

  tl_ofmap[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step),
                                      CVK_FMT_BF16, /*eu_align=*/1);

  if (is_reuse_weight()) {
    tl_ofmap[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step),
                                        CVK_FMT_BF16, /*eu_align=*/1);
  } else {
    // tl_ofmap[1] = tl_ofmap[0];
  }

  ASSERT(tl_weight[0] && tl_ifmap[0] && tl_ofmap[0]);

  if (is_reuse_weight()) {
    ASSERT(tl_weight[1] && tl_ifmap[1] && tl_ofmap[1]);
  }

  if (do_bias) {
    // 16 bit
    tl_bias[0] = ctx.lmem_alloc_tensor({2, (uint32_t)oc_step, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    if (is_reuse_weight()) {
      tl_bias[1] =
          ctx.lmem_alloc_tensor({2, (uint32_t)oc_step, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    } else {
      // tl_bias[1] = tl_bias[0];
    }
    ASSERT(tl_bias[0]);
    if (is_reuse_weight()) {
      ASSERT(tl_bias[1]);
    }
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
      int cur_oc = std::min(oc - oc_pos, (int)oc_step);

      uint64_t coeff_offset = (ig * oc + oc_pos) * sizeof(uint16_t);

      if (do_bias) {
        // 2x 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bias[coeff_flip]->shape = {2, (uint32_t)cur_oc, 1, 1};
        tl_bias[coeff_flip]->stride = ctx.tl_default_stride(tl_bias[coeff_flip]->shape,
                                                            CVK_FMT_BF16, /*eu_align=*/0);

        LLVM_DEBUG(llvm::errs() << llvm::format(
                       "  [ig=%d][oc_pos=%d] tdma_load_stride_bf16:\n"
                       "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                       "%d, %d), gstride (%d, %d, %d)\n",
                       ig, oc_pos, ga_bias + coeff_offset,
                       tl_bias[coeff_flip]->start_address, tl_bias[coeff_flip]->shape.n,
                       tl_bias[coeff_flip]->shape.c, tl_bias[coeff_flip]->shape.h,
                       tl_bias[coeff_flip]->shape.w, bias_gstride.n, bias_gstride.c,
                       bias_gstride.h));
        ctx.tdma_load_stride_bf16(tl_bias[coeff_flip], ga_bias + coeff_offset,
                                  bias_gstride);
      }

      // Weight shape for load != shape for tiu
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_weight[coeff_flip]->shape = ctx.shape_t4(ic, cur_oc, kh, kw);
      tl_weight[coeff_flip]->stride = ctx.tl_default_stride(tl_weight[coeff_flip]->shape,
                                                            CVK_FMT_BF16, /*eu_align*/ 0);

      uint64_t weight_offset =
          (ig * oc * ic * kh * kw + oc_pos * ic * kh * kw) * sizeof(uint16_t);
      {
        // Same local address, different shape, stride
        cvk_tl_t tl_tmp;
        tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
        tl_tmp.fmt = CVK_FMT_BF16;
        tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic);
        tl_tmp.stride = ctx.tl_default_stride(tl_tmp.shape, CVK_FMT_BF16, /*eu_align=*/0);

        LLVM_DEBUG(llvm::errs() << llvm::format(
                       "  [ig=%d][oc_pos=%d] tdma_load_stride_bf16:\n"
                       "    tl_weight gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                       "%d, %d), gstride (%d, %d, %d)\n",
                       ig, oc_pos, weight_offset, tl_tmp.start_address, tl_tmp.shape.n,
                       tl_tmp.shape.c, tl_tmp.shape.h, tl_tmp.shape.w, weight_gstride.n,
                       weight_gstride.c, weight_gstride.h));
        ctx.tdma_load_stride_bf16(&tl_tmp, ga_weight + weight_offset, weight_gstride);
      }

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, (int)n_step);

        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = std::min(oh - oh_pos, (int)oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = std::max(oh_top * stride_h - pad_top, 0);
          int ih_bot = std::min((oh_bot - 1) * stride_h + kh_ext - pad_top, (int)input_h);
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
            int cur_ow = std::min(ow - ow_pos, (int)ow_step);

            int ow_left = ow_pos;
            int ow_right = ow_left + cur_ow;
            int iw_left = std::max(ow_left * stride_w - pad_left, 0);
            int iw_right =
                std::min((ow_right - 1) * stride_w + kw_ext - pad_left, (int)input_w);
            int cur_iw = iw_right - iw_left;

            int pw_left = 0;
            if (iw_left == 0) {
              pw_left = pad_left - ow_left * stride_w;
            }

            int pw_right = 0;
            if (iw_right == input_w) {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
            }

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d]"
                           " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                           ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                           ig, oc_pos, n_pos, oh_pos, ow_pos, cur_oh, cur_ih, ih_top,
                           ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_ofmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[flip]->stride = ctx.tl_default_stride(tl_ofmap[flip]->shape,
                                                           CVK_FMT_BF16, /*eu_align=*/1);

            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, ic, cur_ih, cur_iw);
            tl_ifmap[flip]->stride = ctx.tl_default_stride(tl_ifmap[flip]->shape,
                                                           CVK_FMT_BF16, /*eu_align=*/1);

            uint64_t ifmap_offset =
                (ig * ic * input_h * input_w + n_pos * input_c * input_h * input_w +
                 ih_top * input_w + iw_left) *
                sizeof(uint16_t);

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] "
                           "tdma_load_stride_bf16:\n"
                           "    tl_ifmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                           "%d, %d), gstride (%d, %d, %d)\n",
                           ig, oc_pos, n_pos, oh_pos, ow_pos, ifmap_offset,
                           tl_ifmap[flip]->start_address, tl_ifmap[flip]->shape.n,
                           tl_ifmap[flip]->shape.c, tl_ifmap[flip]->shape.h,
                           tl_ifmap[flip]->shape.w, ifmap_gstride.n, ifmap_gstride.c,
                           ifmap_gstride.h));

            ctx.tdma_load_stride_bf16(tl_ifmap[flip], ga_ifmap + ifmap_offset,
                                      ifmap_gstride);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              cvk_tiu_pt_convolution_param_t param = {0};
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = param.ins_last_h = 0;
              param.ins_w = param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pw_left;
              param.pad_right = pw_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.ps32_mode = 0;
              param.w_is_const = 0;
              param.layer_id = layer_id;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
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

              ctx.tiu_pt_convolution(&param);
            }

            ga_ofmap_cur[flip] =
                ga_ofmap + (ig * oc * oh * ow + n_pos * output_c * oh * ow +
                            oc_pos * oh * ow + oh_top * ow + ow_left) *
                               sizeof(uint16_t);

            if (!is_reuse_weight()) {
              flip = 1;
              first = 0;
            }

            if (first) {
              // postpone first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int flip_back = 1 - flip;

              // Store back to global memory
              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] "
                             "tdma_store_stride_bf16:\n"
                             "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                             "%d, %d), gstride (%d, %d, %d)\n",
                             ig, oc_pos, n_pos, oh_pos, ow_pos, ga_ofmap_cur[flip_back],
                             tl_ofmap[flip_back]->start_address,
                             tl_ofmap[flip_back]->shape.n, tl_ofmap[flip_back]->shape.c,
                             tl_ofmap[flip_back]->shape.h, tl_ofmap[flip_back]->shape.w,
                             ofmap_gstride.n, ofmap_gstride.c, ofmap_gstride.h));

              ctx.tdma_store_stride_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                                         ofmap_gstride);
            }

            flip = 1 - flip;

          } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

        } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      } // for (int n_i = 0; n_i < n; ni += n_step)

      if (!is_reuse_weight()) {
        coeff_flip = 1;
      }

      coeff_flip = 1 - coeff_flip;

    } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    if (!is_reuse_weight()) {
      // TODO: no need to store last one cuz we store every loop
      flip = 1;
    } else {
      int flip_back = 1 - flip;

      // Store back to global memory
      LLVM_DEBUG(llvm::errs() << llvm::format(
                     "  [ig=%d] tdma_store_stride_bf16:\n"
                     "    tl_ofmap gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                     "%d, %d), gstride (%d, %d, %d)\n",
                     ig, ga_ofmap_cur[flip_back], tl_ofmap[flip_back]->start_address,
                     tl_ofmap[flip_back]->shape.n, tl_ofmap[flip_back]->shape.c,
                     tl_ofmap[flip_back]->shape.h, tl_ofmap[flip_back]->shape.w,
                     ofmap_gstride.n, ofmap_gstride.c, ofmap_gstride.h));

      ctx.tdma_store_stride_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                                 ofmap_gstride);
    }

  } // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
  if (do_bias) {
    if (is_reuse_weight())
      ctx.lmem_free_tensor(tl_bias[1]);

    ctx.lmem_free_tensor(tl_bias[0]);
  }
  if (is_reuse_weight())
    ctx.lmem_free_tensor(tl_ofmap[1]);

  ctx.lmem_free_tensor(tl_ofmap[0]);

  if (is_reuse_weight())
    ctx.lmem_free_tensor(tl_ifmap[1]);

  ctx.lmem_free_tensor(tl_ifmap[0]);

  if (is_reuse_weight())
    ctx.lmem_free_tensor(tl_weight[1]);

  ctx.lmem_free_tensor(tl_weight[0]);

  LLVM_DEBUG(llvm::errs() << "<=ConvReuseWeight"
                          << "\n");
}

cvk_tl_t *BM1880v2ConvBF16::conv_ifmap_tensor(const CviBackendContext &ctx, int input_n,
                                              int input_c, int input_h, int input_w,
                                              uint32_t ic_step, cvk_fmt_t fmt) {
  assert(fmt == CVK_FMT_BF16);

  cvk_tl_shape_t s;
  s.n = input_n;
  s.c = ic_step;
  s.h = input_h;
  s.w = input_w;
  cvk_tl_t *t = ctx.lmem_alloc_tensor(s, fmt, /*eu_align=*/1);
  if (t) {
    t->shape.c = input_c;
  }
  return t;
}

cvk_tl_t *BM1880v2ConvBF16::conv_weight_tensor(const CviBackendContext &ctx, int kh,
                                               int kw, uint32_t ic_step, uint32_t oc_step,
                                               int output_c, int input_c, cvk_fmt_t fmt) {
  assert(fmt == CVK_FMT_BF16);

  cvk_tl_shape_t s;
  s.n = ic_step;
  s.c = oc_step;
  s.h = kh;
  s.w = kw;
  cvk_tl_t *t = ctx.lmem_alloc_tensor(s, fmt, /*eu_align=*/0);
  if (t) {
    t->shape.c = output_c;
    t->shape.n = input_c;
  }
  return t;
}

cvk_tl_t *BM1880v2ConvBF16::conv_ofmap_tensor(const CviBackendContext &ctx, int input_n,
                                              int oh, int ow, int output_c,
                                              uint32_t oc_step, cvk_fmt_t fmt) {
  assert(fmt == CVK_FMT_BF16);

  cvk_tl_shape_t s;
  s.n = input_n;
  s.c = oc_step;
  s.h = oh;
  s.w = ow;
  // partail sum's temp buffer is 32bit, we reserve 2 times(2 * bf16_size = fp32) for is
  s.n = s.n * 2;
  cvk_tl_t *t = ctx.lmem_alloc_tensor(s, fmt, /*eu_align=*/1);

  if (t) {
    t->shape.c = output_c;
    t->shape.n = input_n;
  }

  return t;
}

cvk_tl_t *BM1880v2ConvBF16::conv_bias_tensor(const CviBackendContext &ctx, int output_c,
                                             cvk_fmt_t fmt) {
  assert(fmt == CVK_FMT_BF16);
  cvk_tl_shape_t s;
  s.n = 2;
  s.c = output_c;
  s.h = 1;
  s.w = 1;
  return ctx.lmem_alloc_tensor(s, fmt, /*eu_align=*/0);
}

int BM1880v2ConvBF16::conv_ps(const CviBackendContext &ctx,
                              cvk_tiu_pt_convolution_param_t *conv_param,
                              uint32_t ic_step, uint32_t oc_step, uint32_t n_step,
                              cvk_tg_shape_t weight_shape, gaddr_t input_gaddr,
                              gaddr_t weight_gaddr, cvk_fmt_t fmt) {
  // for ps32_md[1] = 1, relu_enable & rshift_bits need to set to 0
  // so we store those parameters to conv_tmp_para
  cvk_tiu_pt_convolution_param_t conv_tmp_param = {0};
  conv_tmp_param.relu_enable = conv_param->relu_enable;
  conv_tmp_param.rshift_bits = conv_param->rshift_bits;
  conv_tmp_param.bias = conv_param->bias;
  conv_tmp_param.ofmap = conv_param->ofmap;

  assert(conv_param->ifmap->shape.c >= ic_step);
  assert(fmt == CVK_FMT_BF16);
  // TODO: deal with multibatch at once
  assert(n_step == 1 && conv_param->ifmap->shape.n == 1 &&
         conv_param->ofmap->shape.n == 1);

  // TODO: cal by function
  int bytesize = fmt == CVK_FMT_BF16 ? 2 : 1;

  const cvk_tl_t *saved_tl_weight = conv_param->weight;
  const cvk_tl_t *saved_tl_ifmap = conv_param->ifmap;
  const cvk_tl_t *saved_tl_bias = conv_param->bias;
  const cvk_tl_t *saved_tl_ofmap = conv_param->ofmap;
  const uint32_t input_c = conv_param->ifmap->shape.c;

  cvk_tl_shape_t cur_tl_ifmap_shape = {n_step, ic_step, conv_param->ifmap->shape.h,
                                       conv_param->ifmap->shape.w};

  cvk_tg_shape_t cur_tg_ifmap_shape = {n_step, cur_tl_ifmap_shape.c, cur_tl_ifmap_shape.h,
                                       cur_tl_ifmap_shape.w};

  cvk_tg_shape_t s;
  s.n = conv_param->ifmap->shape.n;
  s.c = conv_param->ifmap->shape.c;
  s.h = conv_param->ifmap->shape.h;
  s.w = conv_param->ifmap->shape.w;
  cvk_tg_stride_t cur_tg_ifmap_stride = ctx.tg_default_stride(s, fmt);

  cvk_tg_t cur_tg_ifmap;
  cur_tg_ifmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(input_gaddr);
  cur_tg_ifmap.start_address = input_gaddr;
  cur_tg_ifmap.shape = cur_tg_ifmap_shape;
  cur_tg_ifmap.stride = cur_tg_ifmap_stride;
  cur_tg_ifmap.fmt = fmt;

  // copy structure for variable input channel / output channel
  cvk_tl_t cur_tl_ifmap;
  tl_copy(ctx, &cur_tl_ifmap, conv_param->ifmap, cur_tl_ifmap_shape.n,
          cur_tl_ifmap_shape.c, cur_tl_ifmap_shape.h, cur_tl_ifmap_shape.w,
          /*eu_align=*/1);

  cvk_tl_t cur_tl_ofmap;
  tl_copy(ctx, &cur_tl_ofmap, conv_param->ofmap, n_step, oc_step,
          conv_param->ofmap->shape.h, conv_param->ofmap->shape.w,
          /*eu_align=*/1);

  cvk_tl_t cur_tl_bias;
  if (do_bias) {
    tl_copy(ctx, &cur_tl_bias, conv_param->bias, conv_param->bias->shape.n, oc_step,
            conv_param->bias->shape.h, conv_param->bias->shape.w,
            /*eu_align=*/0);
  }

  cvk_tl_t cur_tl_weight;
  cur_tl_weight.start_address = conv_param->weight->start_address;
  cur_tl_weight.shape = conv_param->weight->shape;
  cur_tl_weight.shape.n = ic_step;
  cur_tl_weight.stride = {
      1, cur_tl_weight.shape.n * cur_tl_weight.shape.h * cur_tl_weight.shape.w * bytesize,
      cur_tl_weight.shape.n * cur_tl_weight.shape.w * bytesize,
      cur_tl_weight.shape.n * bytesize};
  cur_tl_weight.stride.n = bytesize;
  cur_tl_weight.fmt = conv_param->weight->fmt;

  for (uint32_t ci = 0; ci < input_c; ci += ic_step) {
    uint32_t _ic_step = std::min(input_c - ci, ic_step);

    // load ic_step weight
    {
      uint32_t ic = weight_shape.n;
      uint32_t oc = oc_step;
      uint32_t kh = weight_shape.h;
      uint32_t kw = weight_shape.w;

      cvk_tg_t cur_tdma_tg_weight;
      cur_tdma_tg_weight.base_reg_index =
          ctx.getTdmaBaseSelectIndexFromGaddr(weight_gaddr);
      cur_tdma_tg_weight.start_address = weight_gaddr + ci * bytesize;
      cur_tdma_tg_weight.fmt = fmt;
      cur_tdma_tg_weight.shape = {1, oc, kh * kw, ic};
      cur_tdma_tg_weight.stride = ctx.tg_default_stride(cur_tdma_tg_weight.shape, fmt);
      cur_tdma_tg_weight.shape = {1, oc, kh * kw, 1};
      cur_tdma_tg_weight.shape.w = _ic_step;

      cvk_tl_t cur_tdma_tl_weight;
      tl_copy(ctx, &cur_tdma_tl_weight, &cur_tl_weight, cur_tdma_tg_weight.shape.n,
              oc_step, cur_tdma_tg_weight.shape.h, cur_tdma_tg_weight.shape.w,
              /*eu_align=*/0);

      cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
      p1.src = &cur_tdma_tg_weight;
      p1.dst = &cur_tdma_tl_weight;
      ctx.tdma_g2l_bf16_tensor_copy(&p1);
    }

    // load _ic_step in ifmap
    {
      cvk_tdma_g2l_tensor_copy_param_t p2 = {0};
      cur_tg_ifmap.start_address = input_gaddr + ci * cur_tg_ifmap_stride.c;

      tl_reshape(ctx, &cur_tl_ifmap, cur_tl_ifmap.shape.n, _ic_step, cur_tl_ifmap.shape.h,
                 cur_tl_ifmap.shape.w, /*eu_align=*/1);

      cur_tg_ifmap.shape.c = _ic_step;

      p2.src = &cur_tg_ifmap;
      p2.dst = &cur_tl_ifmap;
      ctx.tdma_g2l_bf16_tensor_copy(&p2);
    }

    cur_tl_weight.shape.n = _ic_step;
    cur_tl_weight.shape.c = oc_step;
    // TODO: no need reshape again?
    tl_reshape(ctx, &cur_tl_ofmap, cur_tl_ofmap.shape.n, oc_step, cur_tl_ofmap.shape.h,
               cur_tl_ofmap.shape.w,
               /*eu_align=*/1);

    conv_param->ofmap = &cur_tl_ofmap;
    conv_param->ifmap = &cur_tl_ifmap;
    conv_param->weight = &cur_tl_weight;

    // for ps32_md[1] = 1, relu_enable & rshift_bits need to set to 0
    // so we store those parameters to conv_tmp_para
    if (ci == 0 && ci + ic_step == input_c) {
      // no need to partial sum
    } else if (ci + ic_step >= input_c) {
      conv_param->relu_enable = conv_tmp_param.relu_enable;
      conv_param->rshift_bits = conv_tmp_param.rshift_bits;
      conv_param->bias = conv_tmp_param.bias;
      conv_param->ps32_mode = 1;

      if (do_bias) {
        tl_reshape(ctx, &cur_tl_bias, cur_tl_bias.shape.n, oc_step, cur_tl_bias.shape.h,
                   cur_tl_bias.shape.w,
                   /*eu_align=*/0);
        conv_param->bias = &cur_tl_bias;
      }

      LLVM_DEBUG(llvm::errs() << llvm::format(
                     "start write result in ci[%u] step %u, ifshapec is %u\n", ci,
                     _ic_step, input_c););
    } else if (ci == 0) {
      conv_param->relu_enable = 0;
      conv_param->rshift_bits = 0;
      conv_param->bias = 0;
      conv_param->ps32_mode = 2;
    } else {
      conv_param->relu_enable = 0;
      conv_param->rshift_bits = 0;
      conv_param->bias = 0;
      conv_param->ps32_mode = 3;
    }

    ctx.tiu_pt_convolution(conv_param);
    conv_param->weight = saved_tl_weight;
    conv_param->ifmap = saved_tl_ifmap;
    conv_param->bias = saved_tl_bias;
    conv_param->ofmap = saved_tl_ofmap;
  }
  return 0;
}

void BM1880v2ConvBF16::tl_reshape(const CviBackendContext &ctx, cvk_tl_t *dst, int re_n,
                                  int re_c, int re_h, int re_w, uint8_t eu_align) {
  assert(dst);
  assert(re_n && re_c && re_h && re_w);

  dst->shape.n = re_n;
  dst->shape.c = re_c;
  dst->shape.h = re_h;
  dst->shape.w = re_w;
  dst->stride = ctx.tl_default_stride(dst->shape, dst->fmt, eu_align);
}

// TODO: move to kernel.h
// \re_n means we resize it
void BM1880v2ConvBF16::tl_copy(const CviBackendContext &ctx, cvk_tl_t *dst,
                               const cvk_tl_t *src, int re_n, int re_c, int re_h,
                               int re_w, uint8_t eu_align) {

  assert(src);
  dst->start_address = src->start_address;
  dst->fmt = src->fmt;

  tl_reshape(ctx, dst, re_n, re_c, re_h, re_w, eu_align);
}

void BM1880v2ConvBF16::ConvPs32(const CviBackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int ic_step = ceiling_func(ic, slices.ic);
  ic_step = slices.ic_step;
  int oc_step = slices.oc_step;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "ConvPs32 =>\n"
                 "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
                 "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
                 "  stride (%d, %d), dilation (%d, %d)\n"
                 "  Slices (n=%d, oc=%d, ic=%d, h=%d, w=%d)\n",
                 groups, input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw,
                 pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h,
                 dilation_w, slices.n, slices.oc, slices.ic, slices.h, slices.w));

  bool fused_conv_relu = (!do_scale && !do_bn &&
                          (do_activation && activation_method == RELU &&
                           (!activation_arg || (activation_arg[0] == 0.0f))))
                             ? true
                             : false;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  cvk_tl_t *tl_weight, *tl_bias = nullptr, *tl_ifmap, *tl_ofmap;
  tl_weight = conv_weight_tensor(ctx, kh, kw, ic_step, oc_step, oc, ic, fmt);
  tl_ifmap = conv_ifmap_tensor(ctx, n_step, ic, input_h, input_w, ic_step, fmt);
  tl_ofmap = conv_ofmap_tensor(ctx, n_step, oh, ow, oc, oc_step, fmt);

  ASSERT(tl_weight && tl_ifmap && tl_ofmap);

  if (do_bias) {
    tl_bias = conv_bias_tensor(ctx, oc, fmt);
    ASSERT(tl_bias);
  }

  // split groups
  // TODO: verify n/group case
  for (int ig = 0; ig < groups; ++ig) {
    // split oc
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = std::min(oc - oc_pos, (int)oc_step);

      uint64_t weight_offset =
          (ig * oc * ic * kh * kw + oc_pos * ic * kh * kw) * sizeof(uint16_t);
      uint64_t coeff_offset = (ig * oc + oc_pos) * sizeof(uint16_t);

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, (int)n_step);

        uint64_t ifmap_offset =
            (ig * ic * input_h * input_w + n_pos * ic * input_h * input_w) *
            sizeof(uint16_t);
        // split ic
        cvk_tg_shape_t weight_shape;
        weight_shape.n = ic;
        weight_shape.c = oc;
        weight_shape.h = kh;
        weight_shape.w = kw;

        if (do_bias) {
          tl_reshape(ctx, tl_bias, tl_bias->shape.n, cur_oc, tl_bias->shape.h,
                     tl_bias->shape.w, /*eu_align=*/0);
          ctx.tdma_load_bf16(tl_bias, ga_bias + coeff_offset);
        }

        cvk_tiu_pt_convolution_param_t param = {0};
        param.ofmap = tl_ofmap;
        param.ifmap = tl_ifmap;
        param.weight = tl_weight;
        param.bias = tl_bias;
        param.ins_h = param.ins_last_h = 0;
        param.ins_w = param.ins_last_w = 0;
        param.pad_top = pad_top;
        param.pad_bottom = pad_bottom;
        param.pad_left = pad_left;
        param.pad_right = pad_right;
        param.stride_h = stride_h;
        param.stride_w = stride_w;
        param.dilation_h = dilation_h;
        param.dilation_w = dilation_w;
        param.relu_enable = fused_conv_relu;
        param.ps32_mode = 0;
        param.w_is_const = 0;
        param.layer_id = layer_id;

        conv_ps(ctx, &param, ic_step, cur_oc, cur_n, weight_shape,
                ga_ifmap + ifmap_offset, ga_weight + weight_offset, fmt);

        uint64_t ga_ofmap_cur =
            ga_ofmap + (n_pos * output_c * oh * ow + oc_pos * oh * ow) * sizeof(uint16_t);

        cvk_tl_t cur_tl_ofmap;
        tl_copy(ctx, &cur_tl_ofmap, param.ofmap, n_step, cur_oc, param.ofmap->shape.h,
                param.ofmap->shape.w,
                /*eu_align=*/1);

        cvk_tg_stride_t tg_s = ctx.tg_default_stride({1, (uint32_t)cur_oc, (uint32_t)oh, (uint32_t)ow}, fmt);
        // TODO: use tdma_store
        ctx.tdma_store_stride_bf16(&cur_tl_ofmap, ga_ofmap_cur, tg_s);

      } // for (int n_i = 0; n_i < n; ni += n_step)

    } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

  } // for (int group_i = 0; group_i < groups; ++groups)

  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias);
  }

  ctx.lmem_free_tensor(tl_ofmap);
  ctx.lmem_free_tensor(tl_ifmap);
  ctx.lmem_free_tensor(tl_weight);

  LLVM_DEBUG(llvm::errs() << "<=ConvPs32"
                          << "\n");
}

void BM1880v2ConvBF16::DepthwiseConv(const CviBackendContext &ctx) {
  int oc = output_c;
  // int ic = 1;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int oh_step = ceiling_func(oh, slices.h);
  int ow_step = ceiling_func(ow, slices.w);
  int ih_step = input_h;
  int iw_step = input_w;
  int oc_step = oc;

  // Always use all lanes.
  // Not divided by slices.oc.
  // It is better to store step.
  if (slices.oc > 1) {
    ASSERT(oc > NPU_NUM);
    oc_step = NPU_NUM;
  }

  if (slices.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
  }
  if (slices.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
  }

  bool fused_conv_relu = (!do_scale && !do_bn &&
                          (do_activation && activation_method == RELU &&
                           (!activation_arg || (activation_arg[0] == 0.0f))))
                             ? true
                             : false;

  cvk_tl_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  cvk_tl_t *tl_ifmap[2] = {nullptr};
  cvk_tl_t *tl_ofmap[2] = {nullptr};

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  // cvk_tg_stride_t ofmap_gstride = {static_cast<uint32_t>(output_c) * oh * ow,
  //                                                  static_cast<uint32_t>(oh) * ow,
  //                                                  static_cast<uint32_t>(ow)};
  // cvk_tg_stride_t ifmap_gstride = {static_cast<uint32_t>(input_c) * input_h * input_w,
  //                                                  static_cast<uint32_t>(input_h) *
  //                                                  input_w,
  //                                                  static_cast<uint32_t>(input_w)};
  // cvk_tg_stride_t bias_gstride = {static_cast<uint32_t>(output_c), 1, 1};
  // cvk_tg_stride_t weight_gstride = {
  //     static_cast<uint32_t>(oc) * kh * kw * ic, static_cast<uint32_t>(kh) * kw * ic,
  //     static_cast<uint32_t>(ic)};
  cvk_tg_stride_t ofmap_gstride =
      ctx.tg_default_stride({1, (uint32_t)output_c, (uint32_t)oh, (uint32_t)ow}, CVK_FMT_BF16);
  cvk_tg_stride_t ifmap_gstride =
      ctx.tg_default_stride({1, (uint32_t)input_c, (uint32_t)input_h, (uint32_t)input_w}, CVK_FMT_BF16);
  cvk_tg_stride_t bias_gstride = ctx.tg_default_stride({1, (uint32_t)output_c, 1, 1}, CVK_FMT_BF16);

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(1, oc_step, kh, kw), CVK_FMT_BF16,
                                       /*eu_align=*/1);
  tl_weight[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(1, oc_step, kh, kw), CVK_FMT_BF16,
                                       /*eu_align=*/1);

  tl_ifmap[0] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, ih_step, iw_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ifmap[1] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, ih_step, iw_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ofmap[0] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  tl_ofmap[1] =
      ctx.lmem_alloc_tensor(ctx.shape_t4(n_step, oc_step, oh_step, ow_step), CVK_FMT_BF16,
                            /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] &&
         tl_ofmap[1]);

  if (do_bias) {
    // 16 bit
    tl_bias[0] = ctx.lmem_alloc_tensor({2, (uint32_t)oc_step, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    tl_bias[1] = ctx.lmem_alloc_tensor({2, (uint32_t)oc_step, 1, 1}, CVK_FMT_BF16, /*eu_align=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
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
      int cur_oc = std::min(oc - oc_pos, (int)oc_step);

      uint64_t coeff_offset = (ig * oc + oc_pos) * sizeof(uint16_t);

      if (do_bias) {
        // 2x 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bias[coeff_flip]->shape = {2, (uint32_t)cur_oc, 1, 1};
        tl_bias[coeff_flip]->stride = ctx.tl_default_stride(tl_bias[coeff_flip]->shape,
                                                            CVK_FMT_BF16, /*eu_align=*/0);

        LLVM_DEBUG(llvm::errs() << llvm::format(
                       "  [ig=%d][oc_pos=%d] tdma_load_stride_bf16:\n"
                       "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                       "%d, %d), gstride (%d, %d, %d)\n",
                       ig, oc_pos, ga_bias + coeff_offset,
                       tl_bias[coeff_flip]->start_address, tl_bias[coeff_flip]->shape.n,
                       tl_bias[coeff_flip]->shape.c, tl_bias[coeff_flip]->shape.h,
                       tl_bias[coeff_flip]->shape.w, bias_gstride.n, bias_gstride.c,
                       bias_gstride.h));
        ctx.tdma_load_stride_bf16(tl_bias[coeff_flip], ga_bias + coeff_offset,
                                  bias_gstride);
      }

      // Weight shape for load != shape for tiu
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_weight[coeff_flip]->shape = ctx.shape_t4(1, cur_oc, kh, kw);
      tl_weight[coeff_flip]->stride = ctx.tl_default_stride(tl_weight[coeff_flip]->shape,
                                                            CVK_FMT_BF16, /*eu_align*/ 1);

      uint64_t weight_offset = (oc_pos * kh * kw) * sizeof(uint16_t);
      {
        // Same local address, different shape, stride
        cvk_tl_t tl_tmp;
        tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
        tl_tmp.fmt = CVK_FMT_BF16;
        // depthwise no need to reshape
        // tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh * kw, ic_step);
        tl_tmp.shape = ctx.shape_t4(1, cur_oc, kh, kw);
        tl_tmp.stride = ctx.tl_default_stride(tl_tmp.shape, CVK_FMT_BF16, /*eu_align=*/1);

        // ctx.tdma_load_stride_bf16(&tl_tmp, ga_weight + weight_offset, weight_gstride);
        ctx.tdma_load_bf16(&tl_tmp, ga_weight + weight_offset);
      }

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, (int)n_step);

        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = std::min(oh - oh_pos, (int)oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = std::max(oh_top * stride_h - pad_top, 0);
          int ih_bot = std::min((oh_bot - 1) * stride_h + kh_ext - pad_top, (int)input_h);
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
            int cur_ow = std::min(ow - ow_pos, (int)ow_step);

            int ow_left = ow_pos;
            int ow_right = ow_left + cur_ow;
            int iw_left = std::max(ow_left * stride_w - pad_left, 0);
            int iw_right =
                std::min((ow_right - 1) * stride_w + kw_ext - pad_left, (int)input_w);
            int cur_iw = iw_right - iw_left;

            int pw_left = 0;
            if (iw_left == 0) {
              pw_left = pad_left - ow_left * stride_w;
            }

            int pw_right = 0;
            if (iw_right == input_w) {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left - input_w;
            }

            LLVM_DEBUG(llvm::errs() << llvm::format(
                           "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d]"
                           " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                           ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d\n",
                           ig, oc_pos, n_pos, oh_pos, ow_pos, cur_oh, cur_ih, ih_top,
                           ih_bot, cur_ow, cur_iw, iw_left, iw_right));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_ofmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[flip]->stride = ctx.tl_default_stride(tl_ofmap[flip]->shape,
                                                           CVK_FMT_BF16, /*eu_align=*/1);

            // bmk does not keep eu-align info, user need to update stride if shape
            // changed
            tl_ifmap[flip]->shape = ctx.shape_t4(cur_n, cur_oc, cur_ih, cur_iw);
            tl_ifmap[flip]->stride = ctx.tl_default_stride(tl_ifmap[flip]->shape,
                                                           CVK_FMT_BF16, /*eu_align=*/1);

            uint64_t ifmap_offset =
                (n_pos * input_c * input_h * input_w + oc_pos * input_h * input_w +
                 (ih_top * input_w + iw_left)) *
                sizeof(uint16_t);
            // ctx.tdma_load_bf16(tl_ifmap[flip], ga_ifmap + ifmap_offset);
            ctx.tdma_load_stride_bf16(tl_ifmap[flip], ga_ifmap + ifmap_offset,
                                      ifmap_gstride);

            ctx.parallel_disable();
            ctx.parallel_enable();

            {
              cvk_tiu_depthwise_pt_convolution_param_t param = {0};
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = param.ins_last_h = 0;
              param.ins_w = param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pw_left;
              param.pad_right = pw_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.rshift_bits = 0;
              param.layer_id = layer_id;

              LLVM_DEBUG(llvm::errs() << llvm::format(
                             "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] conv:\n"
                             "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                             "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                             "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                             ig, n_pos, oh_pos, ow_pos, param.ifmap->start_address,
                             param.ifmap->shape.n, param.ifmap->shape.c,
                             param.ifmap->shape.h, param.ifmap->shape.w,
                             param.weight->start_address, param.weight->shape.n,
                             param.weight->shape.c, param.weight->shape.h,
                             param.weight->shape.w, param.ofmap->start_address,
                             param.ofmap->shape.n, param.ofmap->shape.c,
                             param.ofmap->shape.h, param.ofmap->shape.w));

              ctx.tiu_pt_depthwise_convolution(&param);
            }

            ga_ofmap_cur[flip] = ga_ofmap + (n_pos * output_c * oh * ow +
                                             oc_pos * oh * ow + (oh_top * ow + ow_left)) *
                                                sizeof(uint16_t);

            if (first) {
              // postpone first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int flip_back = 1 - flip;

              // Store back to global memory
              // ctx.tdma_store_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back]);
              ctx.tdma_store_stride_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                                         ofmap_gstride);
            }

            flip = 1 - flip;

          } // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

        } // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      } // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    } // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    int flip_back = 1 - flip;

    // Store back to global memory
    ctx.tdma_store_stride_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                               ofmap_gstride);
    // ctx.tdma_store_bf16(tl_ofmap[flip_back], ga_ofmap_cur[flip_back]);

  } // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
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

void BM1880v2ConvBF16::do_conv(const CviBackendContext &ctx) {

  if (input_c == groups && output_c == groups && 1 != groups) {
    DepthwiseConv(ctx);
    return;
  }

  bool isReuseActivation = false;

#if 0 // FIXME: resnet50 scale2a_branch1 failed !!!
  // 1x1 convolution uses the whole input feature map
  // Old parallel conv uses all kernels and get better result for 1x1 kernel.
  // Full weight will be implemented later.
  if (this->kh == 1 && this->kw == 1) {
    isReuseActivation = true;
  }
#endif

  if (isReuseActivation) {
    ConvReuseActivation(ctx);
  } else if (is_split_ic()) {
    ConvPs32(ctx);
  } else {
    ConvReuseWeight(ctx);
  }
}

void bmnet_bf16_conv_forward_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean, gaddr_t ga_bn_variance,
    gaddr_t ga_scale, gaddr_t ga_scale_bias, int input_n, int input_c, int input_h,
    int input_w, int groups, int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t stride_h, uint8_t stride_w, int do_bias, int do_bn,
    int do_scale, int do_scale_bias, int do_activation, float bn_scale, float bn_eps,
    int activation_method, float activation_arg[], gaddr_t activation_ga_slope) {

#if 0
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "bmnet_bf16_conv_forward_kernel:\n"
      "    layer_id %d\n"
      "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
      "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
      "    kernel = (%d, %d), dilation = (%d, %d)\n"
      "    pad = (%d, %d,%d, %d), stride = (%d, %d)\n",
      layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n, input_c, input_h, input_w, groups, output_c,
      kh, kw, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w););
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "    do_bias = %d, do_bn = %d, do_scale = %d\n"
      "    do_scale_bias = %d, do_activation = %d, activation_method %d\n",
      do_bias, do_bn, do_scale, do_scale_bias, do_activation, activation_method););

  ConvBF16_ARGS args = {
      .ga_ifmap = ga_ifmap,
      .ga_ofmap = ga_ofmap,
      .ga_weight = ga_weight,
      .ga_bias = ga_bias,
      .ga_bn_mean = ga_bn_mean,
      .ga_bn_variance = ga_bn_variance,
      .ga_scale = ga_scale,
      .ga_scale_bias = ga_scale_bias,
      .input_n = input_n,
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
      .layer_id = layer_id,
  };

  // For tdma
  ctx.set_layer_id(layer_id);

  BM1880v2ConvBF16 *conv = new BM1880v2ConvBF16(args);
  conv->split(ctx);
  conv->do_conv(ctx);
  delete conv;

#else

  cvi_backend_tg_bf16_conv(
      ctx, layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, ga_bn_mean, ga_bn_variance,
      ga_scale, ga_scale_bias, input_n, input_c, input_h, input_w, groups, output_c, kh,
      kw, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, stride_h,
      stride_w, do_bias, do_bn, do_scale, do_scale_bias, do_activation, bn_scale, bn_eps,
      activation_method, activation_arg, activation_ga_slope);

#endif
}

//}  // namespace bmnet
