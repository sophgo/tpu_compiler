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

static bool _split_nh(const BM1880v2BackendContext &ctx, int n, int ic, int ih, int iw, int oc,
                      int oh, int ow, int kh, int kw, int dh, int dw, int pt, int pb, int pl,
                      int pr, int sh, int sw, int ifmap_blob_num, int ofmap_blob_num, u32 reserved,
                      int *n_slices, int *h_slices) {
  int ic_per_npu = ceiling_func_shift(ic, NPU_SHIFT);
  int oc_per_npu = ceiling_func_shift(oc, NPU_SHIFT);

  LLVM_DEBUG(
      llvm::errs() << llvm::format("_split_nh, reserved 0x%x, ifmap_blob_num %d, ofmap_blob_num %d:\n"
                                "    nchw (%d, %d, %d, %d), output chw (%d, %d, %d)\n"
                                "    k (%d, %d), d (%d, %d), p (%d, %d, %d, %d), s (%d, %d)\n",
                                reserved, ifmap_blob_num, ofmap_blob_num, n, ic, ih, iw, oc, oh, ow,
                                kh, kw, dh, dw, pt, pb, pl, pr, sh, sw));

  int ifmap_blob_size = ic_per_npu * ALIGN(ih * iw, EU_NUM) * INT8_SIZE;
  int ofmap_blob_size = oc_per_npu * ALIGN(oh * ow, EU_NUM) * INT8_SIZE;
  int total_size_per_n = ifmap_blob_num * ifmap_blob_size + ofmap_blob_num * ofmap_blob_size;

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "ifmap_blob_size 0x%x, ofmap_blob_size 0x%x, total_size_per_n 0x%x, reserved:0x%x\n",
          ifmap_blob_size, ofmap_blob_size, total_size_per_n, reserved));

  if (total_size_per_n * n + reserved <= LOCAL_MEM_SIZE) {
    // no need to slice
    *n_slices = 1;
    *h_slices = 1;
    LLVM_DEBUG(llvm::errs() << llvm::format("no need to split, total_size_per_n %d, n %d\n",
                                          total_size_per_n, n));
    return true;
  }

  if (total_size_per_n + reserved <= LOCAL_MEM_SIZE) {
    // localmem can host at least one n, no need to slice h
    *h_slices = 1;
    int slice_n = (LOCAL_MEM_SIZE - reserved) / total_size_per_n;
    ASSERT(slice_n >= 1);
    *n_slices = ceiling_func(n, slice_n);
    LLVM_DEBUG(llvm::errs() << llvm::format("split on n, total_size_per_n %d, n %d, n_slices = %d\n",
                                          total_size_per_n, n, *n_slices));
    return true;
  }

  *n_slices = n;

  // to slice h, start from a rough estimation, then do iteration
  int _h_slices = ceiling_func(total_size_per_n, LOCAL_MEM_SIZE - reserved);
  ASSERT(_h_slices > 1);
#if 0  // TODO(wwcai): why do this?
  // a quick hack for stride_h > 1
  if(sh > 1 && ow > 50 && _h_slices < oh) {
    _h_slices = oh;
  }
#endif
  LLVM_DEBUG(llvm::errs() << llvm::format("split on h, start from h_slices = %d\n", _h_slices));
  int kh_extent = dh * (kh - 1) + 1;
  while (_h_slices <= oh) {
    int cur_oh = ceiling_func(oh, _h_slices);
    // int cur_ih = (cur_oh - 1) * sh + kh_extent - ph * 2;
    int cur_ih = (cur_oh - 1) * sh + kh_extent;  // FIXME: no (- ph * 2)?
    int cur_ifmap_blob_size = ic_per_npu * ALIGN(cur_ih * iw, EU_NUM) * INT8_SIZE;
    int cur_ofmap_blob_size = oc_per_npu * ALIGN(cur_oh * ow, EU_NUM) * INT8_SIZE;
    int cur_total_size_per_n =
        ifmap_blob_num * cur_ifmap_blob_size + ofmap_blob_num * cur_ofmap_blob_size;
    LLVM_DEBUG(llvm::errs() << llvm::format("  cur_ih = %d, cur_ifmap_blob_size = %d\n", cur_ih,
                                          cur_ifmap_blob_size));
    LLVM_DEBUG(llvm::errs() << llvm::format("  cur_oh = %d, cur_ofmap_blob_size = %d\n", cur_oh,
                                          cur_ofmap_blob_size));
    LLVM_DEBUG(llvm::errs() << llvm::format("  cur_total_size_per_n = %d\n", cur_total_size_per_n));
    LLVM_DEBUG(llvm::errs() << llvm::format("  reserved = %d\n", reserved));
    if (cur_total_size_per_n + reserved <= LOCAL_MEM_SIZE) {
      *h_slices = _h_slices;
      LLVM_DEBUG(llvm::errs() << llvm::format("split on h OK, h_slices = %d\n", *h_slices));
      return true;
    }
    _h_slices++;
    LLVM_DEBUG(llvm::errs() << llvm::format("split on h, try h_slices = %d\n", _h_slices));
  }

  ASSERT(_h_slices > oh);
  *h_slices = _h_slices;
  LLVM_DEBUG(llvm::errs() << "split failed\n");
  return false;
}

static u32 __total_coeff_size(const BM1880v2BackendContext &ctx, int n, int ic, int ih, int iw,
                              int oc, int oh, int ow, int kh, int kw, bool do_bias, bool do_bn,
                              bool do_scale, bool do_scale_bias, bool do_prelu) {
  int oc_per_npu = ceiling_func_shift(oc, NPU_SHIFT);
  u32 weight_blob_count = ic * oc_per_npu * kh * kw;
  u32 weight_blob_size = weight_blob_count * INT8_SIZE;

  // bias, bn_mean, bn_variance, scale, scale_bias are all in size of oc
  u32 oc_blob_count = oc_per_npu;
  u32 oc_blob_size = oc_blob_count * INT8_SIZE;
  u32 oc_blob_aligned_size = oc_blob_count * EU_NUM * INT8_SIZE;

  u32 total_coeff_size = 0;
  if (do_scale) {
    total_coeff_size += oc_blob_aligned_size;
  }
  if (do_bn) {
    total_coeff_size += oc_blob_aligned_size;  // variance is aligned
    total_coeff_size += oc_blob_size * 2;      // mean is INT16
  }
  if (do_prelu) {
    total_coeff_size += oc_blob_size;  // TODO(wwcai):channel shared
  }
  total_coeff_size += weight_blob_size;
  if (do_bias) {
    total_coeff_size += oc_blob_size * 2;  // INT16
  }
  if (do_scale_bias) {
    total_coeff_size += oc_blob_size * 2;  // INT16
  }

  total_coeff_size = ALIGN(total_coeff_size, EU_NUM * INT8_SIZE);  // ifmap and ofmap need aligned

  LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "__total_coeff_size, weight_blob_size:0x%x, oc_blob_size:0x%x, total_coeff_size:0x%x\n",
          weight_blob_size, oc_blob_size, total_coeff_size));

  return total_coeff_size;
}

int BM1880v2ConvFixedParallel::split(const BM1880v2BackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  ASSERT(ic * groups == input_c);
  ASSERT(oc * groups == output_c);
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  u32 total_coeff_size =
      __total_coeff_size(ctx, n, ic, ih, iw, oc, oh, ow, kh, kw, do_bias, do_bn, do_scale,
                         do_scale_bias, do_activation && (activation_method == PRELU));

#define CONV_COEFF_SIZE_THRESHOLD (LOCAL_MEM_SIZE * 3 / 4)

  if (total_coeff_size > CONV_COEFF_SIZE_THRESHOLD) {
    // coeff too larege, giveup, has to try slice in c
    llvm::errs() << "[ConvFixedParallel::split], coeff too large: " << total_coeff_size
                 << ", not supported";
    return SPLIT_FAILED;
  }

  bool split_ret = _split_nh(ctx, n, ic, ih, iw, oc, oh, ow, kh, kw, dilation_h, dilation_w,
                             pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w,
                             2,  // in_blob_num
                             2,  // out_blob_num, origin://do_activation ?  4: 2,
                             total_coeff_size, &slices.n, &slices.h);
  if (split_ret != true) {
    // split failed, giveup, try slice in c
    llvm::errs() << "[ConvFixedParallel::split] split failed, try slice in c!";
    return SPLIT_FAILED;
  }

  llvm::errs() << "<TRY SPLIT> [ConvFixedParallel::split], <n:" << slices.n << ", oc:" << slices.oc
            << ", ic:" << slices.ic << ", h:" << slices.h << ">, total " << slices.n * slices.h
            << " slices";
  return (slices.n * slices.h);
}

void BM1880v2ConvFixedParallel::do_conv(const BM1880v2BackendContext &ctx) {
  llvm::errs() << "BM1880v2ConvFixedParallel::do_conv";

  int ic = input_c / groups;
  int oc = output_c / groups;
  ASSERT(ic * groups == input_c);
  ASSERT(oc * groups == output_c);
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "Parallel do_conv:\n"
                  "  in(%d, %d, %d, %d), weight(oc=%d, ic=%d, kh=%d, kw=%d), oh=%d, ow=%d\n",
                  input_n, input_c, input_h, input_w, oc, ic, kh, kw, oh, ow));

  int n_slices = slices.n;
  int h_slices = slices.h;

  bmk1880v2_tensor_lmem_t *tl_weight = nullptr, *tl_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_bn_mean = nullptr, *tl_bn_variance = nullptr;
  bmk1880v2_tensor_lmem_t *tl_scale = nullptr, *tl_scale_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_slope = nullptr;

  bmk1880v2_tensor_lmem_shape_t oc_shape = ctx.shape_t4(1, oc, 1, 1);

  if (do_scale) {
    tl_scale = ctx.lmem_alloc_tensor(oc_shape, FMT_I8, 1);  // weight is aligned in depthwise conv
  }
  if (do_bn) {
    // 16 bits mean
    llvm::errs() << "do_bn=" << do_bn;
    tl_bn_variance =
        ctx.lmem_alloc_tensor(oc_shape, FMT_I8, 1);  // weight is aligned in depthwise conv
    tl_bn_mean = ctx.lmem_alloc_tensor(ctx.shape_t4(2, oc, 1, 1), FMT_I8, 0);  // Not EU-aligned
  }
  if (do_activation && activation_method == PRELU) {
    tl_slope = ctx.lmem_alloc_tensor(oc_shape, FMT_I8, 0);  // Not EU-aligned
  }

  tl_weight = ctx.lmem_alloc_tensor(ctx.shape_t4(1, oc, kh * kw, ic), FMT_I8, 0);  // Not EU-aligned
  LLVM_DEBUG(llvm::errs() << llvm::format("tl_weight (%d %d %d %d) address 0x%x\n", ic, oc, kh, kw,
                                        tl_weight->start_address));

  if (do_bias) {
    tl_bias = ctx.lmem_alloc_tensor(ctx.shape_t4(2, oc, 1, 1), FMT_I8, 0);  // Not EU-aligned
  }

  if (do_scale_bias) {
    tl_scale_bias = ctx.lmem_alloc_tensor(ctx.shape_t4(2, oc, 1, 1), FMT_I8, 0);  // Not EU-aligned
  }

  //
  // because of the tl_free order issue, we need to alloc ifmap and ofmap now
  // and use tl_reshape to reshape when using them
  //
  bmk1880v2_tensor_lmem_t *tl_ifmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_ofmap[2] = {nullptr};
  bmk1880v2_tensor_lmem_t *tl_activation[2] = {nullptr};

  int slice_n = ceiling_func(n, n_slices);
  int slice_oh = oh, slice_ih = ih;
  if (h_slices > 1) {
    slice_oh = ceiling_func(oh, h_slices);
    // slice_ih = (slice_oh - 1) * stride_h + kh_extent - pad_top * 2;
    slice_ih = (slice_oh - 1) * stride_h + kh_extent;  // FIXME: no - pad_top * 2?
  }

  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(slice_n, ic, slice_ih, iw);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(slice_n, oc, slice_oh, ow);
  LLVM_DEBUG(llvm::errs() << llvm::format("alloc ifmap shape (%d,%d,%d,%d)\n", ifmap_shape_.n,
                                        ifmap_shape_.c, ifmap_shape_.h, ifmap_shape_.w));
  LLVM_DEBUG(llvm::errs() << llvm::format("alloc ofmap shape (%d,%d,%d,%d)\n", ofmap_shape_.n,
                                        ofmap_shape_.c, ofmap_shape_.h, ofmap_shape_.w));

  tl_ifmap[0] = ctx.lmem_alloc_tensor(ifmap_shape_, FMT_I8, 1);  // EU-aligned
  tl_ifmap[1] = ctx.lmem_alloc_tensor(ifmap_shape_, FMT_I8, 1);  // EU-aligned
  tl_ofmap[0] = ctx.lmem_alloc_tensor(ofmap_shape_, FMT_I8, 1);  // EU-aligned
  tl_ofmap[1] = ctx.lmem_alloc_tensor(ofmap_shape_, FMT_I8, 1);  // EU-aligned

  // Global memory stride from global memory shape
  bmk1880v2_tensor_tgmem_stride_t ofmap_gstride = {static_cast<u32>(output_c) * oh * ow,
                                                   static_cast<u32>(oh) * ow, static_cast<u32>(ow)};
  bmk1880v2_tensor_tgmem_stride_t ifmap_gstride = {static_cast<u32>(input_c) * input_h * input_w,
                                                   static_cast<u32>(input_h) * input_w,
                                                   static_cast<u32>(input_w)};

  bmk1880v2_tensor_tgmem_stride_t bias_stride;

  // depthwise conv, bias stride is different
  if (input_c == groups && output_c == groups && 1 != groups) {
    bias_stride = {static_cast<u32>(input_c), 1, 1};
  } else {
    bias_stride = {static_cast<u32>(output_c), 1, 1};
  }

  // do conv
  for (int ig = 0; ig < groups; ig++) {
    LLVM_DEBUG(llvm::errs() << llvm::format("ig %d / %d\n", ig, groups));
    ctx.parallel_disable();
    // load coeff for current group
    u32 weight_group_step = oc * ic * kh * kw * INT8_SIZE;
    u64 weight_offset = ga_weight + ig * weight_group_step;
    u32 oc_group_step = oc * INT8_SIZE;

    bmk1880v2_tensor_tgmem_stride_t stride = {static_cast<u32>(oc) * kh * kw * ic,
                                              static_cast<u32>(kh) * kw * ic, static_cast<u32>(ic)};
    ctx.tdma_load_stride(tl_weight, weight_offset, stride, CTRL_WEIGHT);

    LLVM_DEBUG(llvm::errs() << llvm::format("[%d] weight_offset=0x%x ga_weight=0x%x\n", ig,
                                          (unsigned)weight_offset, (unsigned)ga_weight));
    if (do_bias) {
      ctx.tdma_load_stride(tl_bias, ga_bias + ig * oc_group_step, bias_stride, CTRL_WEIGHT);
      LLVM_DEBUG(llvm::errs() << llvm::format("tl_bias (%d %d %d %d) address 0x%x\n", 2, oc, 1, 1,
                                            tl_bias->start_address));
    }

    if (do_bn) {
      ctx.tdma_load_stride(tl_bn_mean, ga_bn_mean + ig * oc_group_step, bias_stride, CTRL_WEIGHT);
      ctx.tdma_load(tl_bn_variance, ga_bn_variance + ig * oc_group_step, CTRL_WEIGHT);
    }

    if (do_scale) {
      ctx.tdma_load(tl_scale, ga_scale + ig * oc_group_step, CTRL_WEIGHT);
    }

    if (do_scale_bias) {
      ctx.tdma_load_stride(tl_scale_bias, ga_scale_bias + ig * oc_group_step, bias_stride,
                           CTRL_WEIGHT);
    }

    if (do_activation && activation_method == PRELU) {
      ctx.tdma_load(tl_slope, activation_ga_slope + ig * oc_group_step, CTRL_WEIGHT);
    }

    bmk1880v2_tensor_lmem_shape_t ifmap_shape[2] = {0};
    bmk1880v2_tensor_lmem_shape_t ofmap_shape[2] = {0};
    gaddr_t ga_ifmap_cur[2] = {0};
    gaddr_t ga_ofmap_cur[2] = {0};
    bmk1880v2_tensor_tgmem_stride_t ifmap_stride = {
        static_cast<u32>(input_c) * ih * iw, static_cast<u32>(ih) * iw, static_cast<u32>(iw)};

    bmk1880v2_tensor_tgmem_stride_t ofmap_stride = {
        static_cast<u32>(output_c) * oh * ow, static_cast<u32>(oh) * ow, static_cast<u32>(ow)};

    int first = 1;
    int flip = 0;

    int n_step = ceiling_func(n, n_slices);
    for (int n_pos = 0; n_pos < n; n_pos += n_step) {
      int cur_n = math_min(n - n_pos, n_step);
      LLVM_DEBUG(llvm::errs() << llvm::format("  n_pos %d / %d, cur_n %d\n", n_pos, n, cur_n));

      int oh_step = ceiling_func(oh, h_slices);
      ASSERT(oh_step);
      for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
        int cur_oh = math_min(oh - oh_pos, oh_step);
        LLVM_DEBUG(
            llvm::errs() << llvm::format("    oh_pos %d / %d, cur_oh %d\n", oh_pos, oh, cur_oh));
        int oh_top = oh_pos;
        int oh_bot = oh_pos + cur_oh;
        int ih_top = math_max(oh_top * stride_h - pad_top, 0);
        int ih_bot = math_min((oh_bot - 1) * stride_h + kh_extent - pad_top, ih);
        int cur_ih = ih_bot - ih_top;

        // padding
        int ph_top = 0;
        if (ih_top == 0) {
          ph_top = pad_top - oh_top * stride_h;
        }
        int ph_bot = 0;
        if (ih_bot == ih) {
          ph_bot = (oh_bot - 1) * stride_h + kh_extent - pad_top - ih;
        }
        LLVM_DEBUG(llvm::errs() << llvm::format("      cur_ih %d, ih_top %d, ph_top %d, ph_bot %d\n",
                                              cur_ih, ih_top, ph_top, ph_bot));

        ifmap_shape[flip] = ctx.shape_t4(cur_n, ic, cur_ih, iw);

        // reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_ifmap[flip]->shape = ifmap_shape[flip];
        tl_ifmap[flip]->stride = ctx.tensor_lmem_default_stride(tl_ifmap[flip]->shape, 1);

        ofmap_shape[flip] = ctx.shape_t4(cur_n, oc, cur_oh, ow);

        // reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_ofmap[flip]->shape = ofmap_shape[flip];
        tl_ofmap[flip]->stride = ctx.tensor_lmem_default_stride(tl_ofmap[flip]->shape, 1);

        LLVM_DEBUG(llvm::errs() << llvm::format("      reshape flip %d\n", flip));
        LLVM_DEBUG(llvm::errs() << llvm::format("      reshape ifmap (%d,%d,%d,%d)\n",
                                              ifmap_shape[flip].n, ifmap_shape[flip].c,
                                              ifmap_shape[flip].h, ifmap_shape[flip].w));
        LLVM_DEBUG(llvm::errs() << llvm::format("      reshape ofmap (%d,%d,%d,%d)\n",
                                              ofmap_shape[flip].n, ofmap_shape[flip].c,
                                              ofmap_shape[flip].h, ofmap_shape[flip].w));
        // load
        ga_ifmap_cur[flip] =
            ga_ifmap + (n_pos * input_c * ih * iw + ig * ic * ih * iw + ih_top * iw) * INT8_SIZE;
        ga_ofmap_cur[flip] =
            ga_ofmap + (n_pos * output_c * oh * ow + ig * oc * oh * ow + oh_top * ow) * INT8_SIZE;

        LLVM_DEBUG(llvm::errs() << llvm::format("[%d][%d][%d] tdma_load_stride, tl_ifmap[flip=%d]\n",
                                              ig, n_pos, oh_pos, flip));

        ctx.tdma_load_stride(tl_ifmap[flip], ga_ifmap_cur[flip], ifmap_stride, CTRL_NEURON);
        if (result_add) {
          ASSERT(0);  // TODO(wwcai)
        }
        LLVM_DEBUG(llvm::errs() << llvm::format("      load flip %d\n", flip));

        ctx.parallel_disable();
        ctx.parallel_enable();

        // reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_weight->shape = ctx.shape_t4(ic, oc, kh, kw);
        tl_weight->stride = ctx.tensor_lmem_default_stride(tl_weight->shape, 0);  // Not EU-aligned
        {
          bmk1880v2_tiu_convolution_param_t param;
          param.ofmap = tl_ofmap[flip];
          param.ifmap = tl_ifmap[flip];
          param.weight = tl_weight;
          param.bias = tl_bias;
          param.ins_h = param.ins_last_h = 0;
          param.ins_w = param.ins_last_w = 0;
          param.pad_top = ph_top;
          param.pad_bottom = ph_bot;
          param.pad_left = pad_left;
          param.pad_right = pad_right;
          param.stride_h = (cur_oh == 1) ? 1 : stride_h;
          param.stride_w = stride_w;
          param.dilation_h = dilation_h;
          param.dilation_w = dilation_w;
          param.relu_enable = false;
          param.rshift_bits = right_shift_width;
          param.enable_double_conv = 0;
          param.ps32_mode = 0;
          param.w_is_const = 0;
          param.layer_id = layer_id;
          ctx.tiu_convolution(&param);

          // reshape
          // bmk does not keep eu-align info, user need to update stride if shape changed
          tl_weight->shape = {1, static_cast<u32>(oc), static_cast<u32>(kh) * kw,
                              static_cast<u32>(ic)};
          tl_weight->stride =
              ctx.tensor_lmem_default_stride(tl_weight->shape, 0);  // Not EU-aligned
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
        param.relu_enable = 0;
        param.layer_id = layer_id;

        if (do_bn) {
          param.ofmap = tl_ofmap[flip];
          param.ifmap = tl_ofmap[flip];
          param.weight = tl_bn_variance;
          param.bias = tl_bn_mean;
          param.rshift_bits = bn_right_shift_width;
          ctx.tiu_depthwise_convolution(&param);
        }

        if (do_scale && do_scale_bias) {
          // computing x * scale + bias
          param.ofmap = tl_ofmap[flip];
          param.ifmap = tl_ofmap[flip];
          param.weight = tl_scale;
          param.bias = tl_scale_bias;
          param.rshift_bits = scale_right_shift_width;
          ctx.tiu_depthwise_convolution(&param);
        } else if (do_scale) {
          param.ofmap = tl_ofmap[flip];
          param.ifmap = tl_ofmap[flip];
          param.weight = tl_scale;
          param.bias = nullptr;
          param.rshift_bits = scale_right_shift_width;
          ctx.tiu_depthwise_convolution(&param);
        } else if (do_scale_bias) {
          ASSERT(0);  // TODO(zakk)
        }

        if (do_activation) {
          switch (activation_method) {
            case RELU:
              // TODO: handle activation_arg[0] != 0
              bmk1880v2_tiu_element_wise_max_param_t p13;
              p13.max = tl_ofmap[flip];
              p13.a = tl_ofmap[flip];
              p13.b_is_const = 1;
              p13.b_is_signed = 1;
              p13.b_val = 0;
              p13.layer_id = layer_id;
              ctx.tiu_element_wise_max(&p13);
              break;
            case PRELU:
              ASSERT(0);  // TODO(wwcai)
              break;
            default:
              ASSERT(0);
          }
        }
        LLVM_DEBUG(llvm::errs() << llvm::format("      compute flip %d\n", flip));
        // store last flip
        if (first) {
          // the other side is not ready, do nothing
          first = 0;
        } else {
          // store and free the other side
          int flip_back = (1 - flip);
          LLVM_DEBUG(llvm::errs() << llvm::format("      store flip %d\n", flip_back));
          ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_stride,
                                CTRL_NEURON);
        }

        flip = 1 - flip;
      }  // h
    }    // n
    ctx.parallel_disable();
    // the last iteration stored the other side, leave the last side not stored
    int flip_back = (1 - flip);
    LLVM_DEBUG(llvm::errs() << llvm::format("      last store flip %d\n", flip_back));
    ctx.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back], ofmap_stride, CTRL_NEURON);
  }  // ig

  // free the prealloc neuron tensors
  if (do_activation) {
    // ctx.tl_free(tl_activation[1]);
    // ctx.tl_free(tl_activation[0]);
  }

  ctx.lmem_free_tensor(tl_ofmap[1]);
  ctx.lmem_free_tensor(tl_ofmap[0]);
  ctx.lmem_free_tensor(tl_ifmap[1]);
  ctx.lmem_free_tensor(tl_ifmap[0]);

  // free tensors for coeff
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias);
  }
  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias);
  }
  ctx.lmem_free_tensor(tl_weight);
  if (do_bn) {
    // ctx.lmem_free_tensor(tl_bn_eps);
    // ctx.lmem_free_tensor(tl_bn_scale_negtive_factor);
    // ctx.lmem_free_tensor(tl_bn_scale_factor);
    ctx.lmem_free_tensor(tl_bn_mean);
    ctx.lmem_free_tensor(tl_bn_variance);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale);
  }
}

static int __calcuate_occuppied_lmem(const BM1880v2BackendContext &ctx, int ic, int ih, int iw,
                                     int oc, int oh, int ow, int kh, int kw, int bias_blobs_num,
                                     int oc_shape_align_blobs_num, int ofmap_blobs_num) {
  int ic_per_npu = ceiling_func_shift(ic, NPU_SHIFT);
  int oc_per_npu = ceiling_func_shift(oc, NPU_SHIFT);

  int bias_blob_size = oc_per_npu * INT8_SIZE * 2;  // INT16
  int weight_blob_size = ic * oc_per_npu * kh * kw * INT8_SIZE;
  int coeff_size_per_n = weight_blob_size + bias_blob_size * bias_blobs_num;

  int ifmap_blob_size = ic_per_npu * ALIGN(ih * iw, EU_NUM) * INT8_SIZE;
  int ofmap_blob_size = oc_per_npu * ALIGN(oh * ow, EU_NUM) * INT8_SIZE;
  int neuron_size_per_n = ifmap_blob_size + ofmap_blob_size * ofmap_blobs_num;

  return neuron_size_per_n + ALIGN(coeff_size_per_n, EU_NUM * INT8_SIZE);
}

/* calcuate depthwise conv occuppied memory */
static int __calcuate_dw_occuppied_lmem(const BM1880v2BackendContext &ctx, int ic, int ih, int iw,
                                        int oc, int oh, int ow, int kh, int kw, int bias_blobs_num,
                                        int oc_shape_align_blobs_num, int ofmap_blobs_num) {
  int ic_per_npu = ceiling_func_shift(ic, NPU_SHIFT);
  int oc_per_npu = ceiling_func_shift(oc, NPU_SHIFT);

  int bias_blob_size = oc_per_npu * INT8_SIZE * 2;  // INT16
  int weight_blob_size = oc_per_npu * ALIGN(kh * kw, EU_NUM) * INT8_SIZE;
  int coeff_size_per_n = weight_blob_size + bias_blob_size * bias_blobs_num;

  int ifmap_blob_size = ic_per_npu * ALIGN(ih * iw, EU_NUM) * INT8_SIZE;
  int ofmap_blob_size = oc_per_npu * ALIGN(oh * ow, EU_NUM) * INT8_SIZE;
  int neuron_size_per_n = ifmap_blob_size + ofmap_blob_size * ofmap_blobs_num;

  return neuron_size_per_n + ALIGN(coeff_size_per_n, EU_NUM * INT8_SIZE);
}

static bool _split_nch(const BM1880v2BackendContext &ctx, int n, int ic, int ih, int iw, int oc,
                       int oh, int ow, int kh, int kw, int dilation_h, int stride_h,
                       int bias_blobs_num, int oc_shape_align_blobs_num, int ofmap_blobs_num,
                       SLICES &slices) {
  int total_size_per_n =
      __calcuate_occuppied_lmem(ctx, ic, ih, iw, oc, oh, ow, kh, kw, bias_blobs_num,
                                oc_shape_align_blobs_num, ofmap_blobs_num);

  // no need to slice
  if (total_size_per_n * n <= LOCAL_MEM_SIZE) {
    slices.n = 1;
    LLVM_DEBUG(llvm::errs() << llvm::format("no need to split, n_slices = %d, c_slices=(%d, %d)\n",
                                          slices.n, slices.ic, slices.oc));
    return true;
  }

  if (total_size_per_n <= LOCAL_MEM_SIZE) {
    // localmem can host at least one n, no need to slice c
    int slice_n = LOCAL_MEM_SIZE / total_size_per_n;
    ASSERT(slice_n >= 1);
    slices.n = ceiling_func(n, slice_n);
    LLVM_DEBUG(llvm::errs() << llvm::format("split on n, n_slices = %d, c_slices=(%d, %d)\n",
                                          slices.n, slices.ic, slices.oc));
    return true;
  }

  slices.n = n;
  int cur_oh = oh, cur_ih = ih;
  int kh_extent = dilation_h * (kh - 1) + 1;

  if (oh > 10) {
    slices.h = ceiling_func(oh, 10);
    cur_oh = ceiling_func(oh, slices.h);
    cur_ih = (cur_oh - 1) * stride_h + kh_extent;
    total_size_per_n =
        __calcuate_occuppied_lmem(ctx, ic, cur_ih, iw, oc, cur_oh, ow, kh, kw, bias_blobs_num,
                                  oc_shape_align_blobs_num, ofmap_blobs_num);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      // localmem can host at least one n, no need to slice c
      LLVM_DEBUG(llvm::errs() << llvm::format("split on n & h, n_slices = %d, h_slices=%d\n",
                                            slices.n, slices.h));
      return true;
    }
  }

  cur_oh = ceiling_func(oh, slices.h);
  cur_ih = (cur_oh - 1) * stride_h + kh_extent;

  // split ic and oc
  while (slices.ic <= ic) {
    int cur_ic = ceiling_func(ic, slices.ic);

    total_size_per_n =
        __calcuate_occuppied_lmem(ctx, cur_ic, cur_ih, iw, oc, cur_oh, ow, kh, kw, bias_blobs_num,
                                  oc_shape_align_blobs_num, ofmap_blobs_num);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      // localmem can host at least one n, no need to slice c
      LLVM_DEBUG(llvm::errs() << llvm::format("split on n & c, n_slices = %d, c_slices=(%d, %d)\n",
                                            slices.n, slices.ic, slices.oc));
      return true;
    }

    while (slices.oc <= oc) {
      int cur_oc = ceiling_func(oc, slices.oc);
      total_size_per_n =
          __calcuate_occuppied_lmem(ctx, cur_ic, cur_ih, iw, cur_oc, cur_oh, ow, kh, kw,
                                    bias_blobs_num, oc_shape_align_blobs_num, ofmap_blobs_num);
      if (total_size_per_n <= LOCAL_MEM_SIZE) {
        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "2 split on n & c, n_slices = %d, c_slices=(%d, %d), h_slices:%d\n",
                        slices.n, slices.ic, slices.oc, slices.h));
        return true;
      }
      slices.oc += 1;
    }

    slices.oc = 1;
    slices.ic += 1;
  }

  return false;
}

int BM1880v2ConvFixedSerial::split(const BM1880v2BackendContext &ctx) {
  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int bias_blobs_num = 0;            // oc shape and int16
  int oc_shape_align_blobs_num = 0;  // eg. variance for depthwise conv
  int ofmap_blobs_num = 1;

  if (do_bias) {
    bias_blobs_num += 2;  // INT16
  }
  if (do_bn) {
    bias_blobs_num += 2;  // INT16
    oc_shape_align_blobs_num += 1;
  }
  if (do_scale) {
    oc_shape_align_blobs_num += 1;
  }
  if (do_scale_bias) {
    bias_blobs_num += 2;  // INT16
  }
  if (do_activation && activation_method == PRELU) {
    ofmap_blobs_num += 3;
  }
  if (do_activation && activation_method == RELU && activation_arg[0] != 0.0f) {
    ofmap_blobs_num += 2;
  }
  if (do_activation) {
    oc_shape_align_blobs_num++;  // tl_slope
  }

  bool ret = false;
  ret = _split_nch(ctx, input_n, ic, input_h, input_w, oc, oh, ow, kh, kw, dilation_h, stride_h,
                   bias_blobs_num, oc_shape_align_blobs_num, ofmap_blobs_num, slices);

  if (ret != true) {
    llvm::errs() << "[ConvFixedSerial::split] failed to split, not supported";
    return SPLIT_FAILED;
  }

  llvm::errs() << "<TRY SPLIT> [ConvFixedSerial::split], <n:" << slices.n << ", oc:" << slices.oc
            << ", ic:" << slices.ic << ", h:" << slices.h << ">, total "
            << slices.n * slices.oc * slices.ic * slices.h << " slices";

  return (slices.n * slices.oc * slices.ic * slices.h);
}

void BM1880v2ConvFixedSerial::do_conv(const BM1880v2BackendContext &ctx) {
  llvm::errs() << "ConvFixedSerial::do_conv";

  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices.n);
  int oc_step = ceiling_func(oc, slices.oc);
  int ic_step = ceiling_func(ic, slices.ic);
  int oh_step = ceiling_func(oh, slices.h);
  int ih_step = oh_step * stride_h + kh_ext - 1 - pad_top;  // TODO(wwcai)

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "Serial do_conv:\n"
                  "  in(%d, %d, %d, %d), weight(oc=%d, ic=%d, kh=%d, kw=%d), oh=%d, ow=%d\n",
                  input_n, input_c, input_h, input_w, oc, ic, kh, kw, oh, ow));

  bmk1880v2_tensor_lmem_shape_t oc_shape_ = ctx.shape_t4(1, oc_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(n_step, ic_step, ih_step, input_w);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(n_step, oc_step, oh_step, ow);

  bmk1880v2_tensor_lmem_t *tl_weight = nullptr, *tl_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_bn_mean = nullptr, *tl_bn_variance = nullptr;
  bmk1880v2_tensor_lmem_t *tl_scale = nullptr, *tl_scale_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_slope = nullptr;

  // some const tensors

  if (do_scale) {
    tl_scale = ctx.lmem_alloc_tensor(oc_shape_, FMT_I8, 1);  // weight of depthwise conv is aligned
  }

  if (do_bn) {
    tl_bn_variance = ctx.lmem_alloc_tensor(oc_shape_, FMT_I8,
                                           1);  // weight of depthwise conv is aligned
    tl_bn_mean = ctx.lmem_alloc_tensor(ctx.shape_t4(2, oc_step, 1, 1), FMT_I8, 0);  // INT16
  }

  tl_weight = ctx.lmem_alloc_tensor(ctx.shape_t4(1, oc_step, kh * kw, ic_step), FMT_I8,
                                    0);  // Not EU-aligned
  LLVM_DEBUG(llvm::errs() << llvm::format("tl_weight (%d %d %d %d) address 0x%x\n", ic_step, oc_step,
                                        kh, kw, tl_weight->start_address));

  if (do_scale_bias) {
    tl_scale_bias = ctx.lmem_alloc_tensor(ctx.shape_t4(2, oc_step, 1, 1), FMT_I8, 0);  // EU-aligned
  }
  if (do_activation && activation_method == PRELU) {
    tl_slope = ctx.lmem_alloc_tensor(oc_shape_, FMT_I8, 1);  // weight of depthwise conv is aligned
  }

  bmk1880v2_tensor_tgmem_stride_t ofmap_gstride = {static_cast<u32>(output_c) * oh * ow,
                                                   static_cast<u32>(oh) * ow, static_cast<u32>(ow)};
  bmk1880v2_tensor_tgmem_stride_t ifmap_gstride = {static_cast<u32>(input_c) * input_h * input_w,
                                                   static_cast<u32>(input_h) * input_w,
                                                   static_cast<u32>(input_w)};

  bmk1880v2_tensor_tgmem_stride_t bias_stride;
  // depthwise conv, bias stride is different
  if (input_c == groups && output_c == groups && 1 != groups) {
    bias_stride = {static_cast<u32>(input_c), 1, 1};
  } else {
    bias_stride = {static_cast<u32>(output_c), 1, 1};
  }
  for (int ig = 0; ig < groups; ig++) {
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = math_min(oc - oc_pos, oc_step);
      LLVM_DEBUG(llvm::errs() << llvm::format("ig=%d oc_pos=%d\n", ig, oc_pos));

      u64 coeff_offset = (ig * oc + oc_pos) * INT8_SIZE;
      bmk1880v2_tensor_lmem_shape_t coeff_shape_ = ctx.shape_t4(1, cur_oc, 1, 1);
      if (do_bias) {
        bmk1880v2_tensor_lmem_shape_t bias_coeff_shape_ = ctx.shape_t4(2, cur_oc, 1, 1);
        tl_bias = ctx.lmem_alloc_tensor(bias_coeff_shape_, FMT_I8, CTRL_NULL);
        ctx.tdma_load_stride(tl_bias, ga_bias + coeff_offset, bias_stride, CTRL_WEIGHT);

        LLVM_DEBUG(llvm::errs() << llvm::format("tl_bias (%d %d %d %d) address 0x%x\n", 2, cur_oc, 1,
                                              1, tl_bias->start_address));
      }

      if (do_bn) {
        // Reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bn_mean->shape = ctx.shape_t4(2, cur_oc, 1, 1);
        tl_bn_mean->stride = ctx.tensor_lmem_default_stride(tl_bn_mean->shape, 0);  // No EU-aligned

        ctx.tdma_load_stride(tl_bn_mean, ga_bn_mean + coeff_offset, bias_stride, CTRL_WEIGHT);

        // Reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bn_variance->shape = coeff_shape_;
        tl_bn_variance->stride =
            ctx.tensor_lmem_default_stride(tl_bn_variance->shape, 1);  // EU-aligne

        ctx.tdma_load(tl_bn_variance, ga_bn_variance + coeff_offset, CTRL_WEIGHT);
      }
      if (do_scale) {
        // Reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale->shape = coeff_shape_;
        tl_scale->stride = ctx.tensor_lmem_default_stride(tl_scale->shape, 1);  // EU-aligned

        ctx.tdma_load(tl_scale, ga_scale + coeff_offset, CTRL_WEIGHT);
      }
      if (do_scale_bias) {
        // Reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_scale_bias->shape = ctx.shape_t4(2, cur_oc, 1, 1);
        tl_scale_bias->stride =
            ctx.tensor_lmem_default_stride(tl_scale_bias->shape, 0);  // EU-aligned

        ctx.tdma_load_stride(tl_scale_bias, ga_scale_bias + coeff_offset, bias_stride, CTRL_WEIGHT);
      }
      if (do_activation && activation_method == PRELU) {
        // Reshape
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_slope->shape = coeff_shape_;
        tl_slope->stride = ctx.tensor_lmem_default_stride(tl_slope->shape, 1);  // EU-aligned

        ctx.tdma_load(tl_slope, activation_ga_slope + coeff_offset, CTRL_WEIGHT);
      }

      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = math_min(input_n - n_pos, n_step);

        LLVM_DEBUG(
            llvm::errs() << llvm::format("oh=%d, oh_step=%d, stride_h=%d, pad_top=%d, kh_ext=%d\n", oh,
                                      oh_step, stride_h, pad_top, kh_ext));
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = math_min(oh - oh_pos, oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = math_max(oh_top * stride_h - pad_top, 0);
          int ih_bot = math_min((oh_bot - 1) * stride_h + kh_ext - pad_top, input_h);
          int cur_ih = ih_bot - ih_top;

          LLVM_DEBUG(llvm::errs() << llvm::format(
                          "oh_top=%d, oh_bot=%d, cur_oh=%d, ih_top=%d, ih_bot=%d, cur_ih=%d\n",
                          oh_top, oh_bot, cur_oh, ih_top, ih_bot, cur_ih));

          int ph_top = 0;
          if (ih_top == 0) {
            ph_top = pad_top - oh_top * stride_h;
          }

          int ph_bot = 0;
          if (ih_bot == input_h) {
            ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top - input_h;
          }

          bmk1880v2_tensor_lmem_shape_t ofmap_shape = ctx.shape_t4(cur_n, cur_oc, cur_oh, ow);
          bmk1880v2_tensor_lmem_t *tl_ofmap =
              ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned
          u64 ofmap_offset = ga_ofmap + (n_pos * output_c * oh * ow + ig * oc * oh * ow +
                                         oc_pos * oh * ow + oh_top * ow) *
                                            INT8_SIZE;

          if (result_add) {
            ASSERT(0);  // TODO(wwcai)
          }

          for (int ic_pos = 0; ic_pos < ic; ic_pos += ic_step) {
            int cur_ic = math_min(ic - ic_pos, ic_step);

            bmk1880v2_tensor_lmem_t *tl_ifmap = ctx.lmem_alloc_tensor(
                ctx.shape_t4(cur_n, cur_ic, cur_ih, input_w), FMT_I8, 1);  // EU-aligned

            u64 ifmap_offset =
                ga_ifmap +
                INT8_SIZE * (n_pos * input_c * input_h * input_w + ig * ic * input_h * input_w +
                             ic_pos * input_h * input_w + ih_top * input_w);

            ctx.tdma_load_stride(tl_ifmap, ifmap_offset, ifmap_gstride, CTRL_NEURON);
            u64 weight_offset;
            weight_offset = ga_weight + INT8_SIZE * (oc_pos * ic * kh * kw +
                                                     ig * oc * ic * kh * kw + ic_pos * kh * kw);
            LLVM_DEBUG(llvm::errs() << llvm::format(
                            "weight_offset=0x%x ga_weight=0x%x ga_ifmap=0x%x\n",
                            (unsigned)weight_offset, (unsigned)ga_weight, (unsigned)ifmap_offset));

            ASSERT(cur_ic == ic);  // TODO(wwcai): how to split ic
            LLVM_DEBUG(llvm::errs() << llvm::format("cur_oc=%d kh=%d kw=%d cur_ic=%d\n", cur_oc, kh,
                                                  kw, cur_ic));
            // Reshape
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_weight->shape = ctx.shape_t4(1, cur_oc, kh * kw, cur_ic);
            tl_weight->stride = ctx.tensor_lmem_default_stride(tl_weight->shape, 0);  // EU-aligned

            bmk1880v2_tensor_tgmem_stride_t stride = {
                static_cast<u32>(cur_oc) * kh * kw * ic, static_cast<u32>(kh) * kw * ic,
                static_cast<u32>(ic)};  // TODO(wwcai): test SPLIT

            ctx.tdma_load_stride(tl_weight, weight_offset, stride, CTRL_WEIGHT);

            // Reshape
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_weight->shape = ctx.shape_t4(cur_ic, cur_oc, kh, kw);
            tl_weight->stride = ctx.tensor_lmem_default_stride(tl_weight->shape, 0);  // EU-aligned

            bmk1880v2_tiu_convolution_param_t param;
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ifmap;
            param.weight = tl_weight;
            param.bias = (ic_pos + ic_step >= ic - 1) ? tl_bias : nullptr;
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
            param.relu_enable = false;
            param.rshift_bits = right_shift_width;
            param.enable_double_conv = 0;
            param.ps32_mode = 0;
            param.w_is_const = 0;
            param.layer_id = layer_id;
            ctx.tiu_convolution(&param);
            ctx.lmem_free_tensor(tl_ifmap);
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
          param.relu_enable = 0;
          param.layer_id = layer_id;
          if (do_bn) {
            // out(n,c,h,w) = in(n,c,h,w) * variance(1,c,1,1) + mean(1,c,1,1)
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ofmap;
            param.weight = tl_bn_variance;
            param.bias = tl_bn_mean;
            param.rshift_bits = bn_right_shift_width;
            ctx.tiu_depthwise_convolution(&param);
          }
          if (do_scale && do_scale_bias) {
            // computing x * scale + bias
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ofmap;
            param.weight = tl_scale;
            param.bias = tl_scale_bias;
            param.rshift_bits = scale_right_shift_width;
            ctx.tiu_depthwise_convolution(&param);
          } else if (do_scale) {
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ofmap;
            param.weight = tl_scale;
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
                  bmk1880v2_tiu_element_wise_max_param_t p13;
                  p13.max = tl_ofmap;
                  p13.a = tl_ofmap;
                  p13.b_is_const = 1;
                  p13.b_is_signed = 1;
                  p13.b_val = 0;
                  p13.layer_id = layer_id;
                  ctx.tiu_element_wise_max(&p13);
                } else {
                  bmk1880v2_tensor_lmem_t *relu =
                      ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned
                  bmk1880v2_tensor_lmem_t *neg =
                      ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned

                  bmk1880v2_tiu_element_wise_max_param_t p1;
                  p1.max = relu;
                  p1.a = tl_ofmap;
                  p1.b_is_const = 1;
                  p1.b_is_signed = 1;
                  p1.b_val = 0;
                  p1.layer_id = layer_id;
                  ctx.tiu_element_wise_max(&p1);

                  bmk1880v2_tiu_element_wise_mul_param_t p2;
                  p2.res_high = nullptr;
                  p2.res_low = relu;
                  p2.a = relu;
                  p2.b_val = activation_gt_scale;
                  p2.b_is_signed = true;
                  p2.b_is_const = 1;
                  p2.rshift_bits = activation_gt_rshift;
                  p2.layer_id = layer_id;
                  ctx.tiu_element_wise_mul(&p2);

                  bmk1880v2_tiu_element_wise_min_param_t p3;
                  p3.min = neg;
                  p3.a = tl_ofmap;
                  p3.b_is_const = 1;
                  p3.b_val = 0;
                  p3.b_is_signed = 1;
                  p3.layer_id = layer_id;
                  ctx.tiu_element_wise_min(&p3);

                  bmk1880v2_tiu_element_wise_mul_param_t p4;
                  p4.res_high = nullptr;
                  p4.res_low = neg;
                  p4.a = neg;
                  p4.b_val = activation_le_scale;
                  p4.b_is_signed = true;
                  p4.b_is_const = 1;
                  p4.rshift_bits = activation_le_rshift;
                  p4.layer_id = layer_id;
                  ctx.tiu_element_wise_mul(&p4);

                  bmk1880v2_tiu_element_wise_or_int8_param_t p5;
                  p5.res = tl_ofmap;
                  p5.a = relu;
                  p5.b = neg;
                  p5.layer_id = layer_id;
                  ctx.tiu_element_wise_or_int8(&p5);

                  ctx.lmem_free_tensor(neg);
                  ctx.lmem_free_tensor(relu);
                }
                break;
              case PRELU: {
                ASSERT(!activation_channel_shared);
                bmk1880v2_tensor_lmem_t *relu =
                    ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned
                bmk1880v2_tensor_lmem_t *neg =
                    ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned
                bmk1880v2_tensor_lmem_t *zero =
                    ctx.lmem_alloc_tensor(ofmap_shape, FMT_I8, 1);  // EU-aligned

                bmk1880v2_tdma_tg2l_tensor_fill_constant_param_t p1;
                p1.constant = 0;
                p1.dst = zero;
                ctx.tdma_tg2l_tensor_fill_constant(&p1);

                // 0. relu = relu(tl_ofmap)
                // 1. relu = (relu * gt_scale) >> gt_rshift
                // 2. neg = neg(0, botom)
                // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
                // 4. tl_ofmap = or relu, neg
                bmk1880v2_tiu_element_wise_max_param_t p2;
                p2.max = relu;
                p2.a = tl_ofmap;
                p2.b_is_const = 1;
                p2.b_is_signed = 1;
                p2.b_val = 0;
                p2.layer_id = layer_id;
                ctx.tiu_element_wise_max(&p2);

                bmk1880v2_tiu_element_wise_mul_param_t p3;
                p3.res_high = nullptr;
                p3.res_low = relu;
                p3.a = relu;
                p3.b_val = activation_gt_scale;
                p3.b_is_signed = true;
                p3.b_is_const = 1;
                p3.rshift_bits = activation_gt_rshift;
                p3.layer_id = layer_id;
                ctx.tiu_element_wise_mul(&p3);

                bmk1880v2_tiu_element_wise_min_param_t p4;
                p4.min = neg;
                p4.a = tl_ofmap;
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
                p5.ofmap = neg;
                p5.ifmap = neg;
                p5.weight = tl_slope;
                p5.bias = nullptr;
                p5.rshift_bits = activation_le_rshift;
                p5.relu_enable = 0;
                p5.layer_id = layer_id;
                ctx.tiu_depthwise_convolution(&p5);

                bmk1880v2_tiu_element_wise_or_int8_param_t p6;
                p6.res = tl_ofmap;
                p6.a = relu;
                p6.b = neg;
                p6.layer_id = layer_id;
                ctx.tiu_element_wise_or_int8(&p6);

                // free
                ctx.lmem_free_tensor(zero);
                ctx.lmem_free_tensor(neg);
                ctx.lmem_free_tensor(relu);

              } break;
              default:
                ASSERT(0);
            }
          }

          ctx.tdma_store_stride(tl_ofmap, ofmap_offset, ofmap_gstride, CTRL_NEURON);

          ctx.lmem_free_tensor(tl_ofmap);
        }
      }
      if (do_bias) {
        ctx.lmem_free_tensor(tl_bias);
      }
    }
  }

  if (do_activation && activation_method == PRELU) {
    ctx.lmem_free_tensor(tl_slope);
  }
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias);
  }
  ctx.lmem_free_tensor(tl_weight);
  if (do_bn) {
    ctx.lmem_free_tensor(tl_bn_mean);
    ctx.lmem_free_tensor(tl_bn_variance);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale);
  }
}

#if 1  // TODO(wwcai): refine
// this is dwconv split scheme
static bool _split_nc(const BM1880v2BackendContext &ctx, int n, int ic, int ih, int iw, int oc,
                      int oh, int ow, int kh, int kw, int dilation_h, int stride_h,
                      int bias_blobs_num, int oc_shape_align_blobs_num, int ofmap_blobs_num,
                      int *n_slices, int *c_slices) {
  DEBUG_WITH_TYPE(
      DEBUG_SPLIT,
      llvm::errs() << llvm::format("n:%d, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, kh:%d, kw:%d, "
                                "bias_blobs_num:%d, oc_shape_align_blobs_num:%d, "
                                "ofmap_blobs_num:%d\n",
                                n, ic, ih, iw, oc, oh, ow, kh, kw, bias_blobs_num,
                                oc_shape_align_blobs_num, ofmap_blobs_num));

  int total_size_per_n =
      __calcuate_dw_occuppied_lmem(ctx, ic, ih, iw, oc, oh, ow, kh, kw, bias_blobs_num,
                                   oc_shape_align_blobs_num, ofmap_blobs_num);

  DEBUG_WITH_TYPE(DEBUG_SPLIT,
                        llvm::errs() << llvm::format("0 total_size_per_n:%d\n", total_size_per_n));

  *c_slices = 1;

  // no need to slice
  if (total_size_per_n * n <= LOCAL_MEM_SIZE) {
    *n_slices = 1;
    DEBUG_WITH_TYPE(DEBUG_SPLIT, llvm::errs() << llvm::format(
                                           "no need to split, n_slices = %d, c_slices=(%d, %d)\n",
                                           *n_slices, *c_slices, *c_slices));
    return true;
  }

  if (total_size_per_n <= LOCAL_MEM_SIZE) {
    // localmem can host at least one n, no need to slice c
    int slice_n = LOCAL_MEM_SIZE / total_size_per_n;
    ASSERT(slice_n >= 1);
    *n_slices = ceiling_func(n, slice_n);
    DEBUG_WITH_TYPE(
        DEBUG_SPLIT, llvm::errs() << llvm::format("split on n, n_slices = %d, c_slices=(%d, %d)\n",
                                               *n_slices, *c_slices, *c_slices));
    return true;
  }

  *n_slices = n;
  *c_slices += 1;
  while (*c_slices <= ic) {
    int cur_c = ceiling_func(ic, *c_slices);
    total_size_per_n =
        __calcuate_dw_occuppied_lmem(ctx, cur_c, ih, iw, cur_c, oh, ow, kh, kw, bias_blobs_num,
                                     oc_shape_align_blobs_num, ofmap_blobs_num);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      // localmem can host at least one n, no need to slice c
      DEBUG_WITH_TYPE(DEBUG_SPLIT, llvm::errs() << llvm::format(
                                             "split on n & c, n_slices = %d, c_slices=(%d, %d)\n",
                                             *n_slices, *c_slices, *c_slices));
      return true;
    }
    *c_slices += 1;
  }

  ASSERT(0);  // TODO
  return false;
}

static void node_depthwise_conv_forward_bmkernel_serial(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_bn_mean, gaddr_t ga_bn_variance,
    gaddr_t ga_scale, gaddr_t ga_scale_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, u16 kh, u16 kw, u16 dilation_h, u16 dilation_w, u8 pad_top,
    u8 pad_bottom, u8 pad_left, u8 pad_right, u8 stride_h, u8 stride_w, int result_add, int do_bias,
    int do_bn, int do_scale, int do_scale_bias, int do_activation, float bn_scale, float bn_eps,
    int activation_method, float activation_arg[], gaddr_t activation_ga_slope,
    bool activation_channel_shared, int right_shift_width, int bn_right_shift_width,
    int scale_right_shift_width) {
  llvm::errs() << "BM1880v2ConvFixedParallel::node_depthwise_conv_forward_bmkernel_serial";

  // this message is too long for llvm::format, so seperate it
  LLVM_DEBUG(llvm::errs() << llvm::format("node_depthwise_conv_forward_bmkernel_serial:\n"
                                        "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                                        "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                                        "    kernel = (%d, %d), dilation = (%d, %d)\n"
                                        "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
                                        ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n, input_c,
                                        input_h, input_w, groups, output_c, kh, kw, dilation_h,
                                        dilation_w, pad_top, pad_bottom, pad_left, pad_right,
                                        stride_h, stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "    result_add = %d, do_bias = %d, do_bn = %d, do_scale = %d\n"
                  "    do_scale_bias = %d, do_activation = %d activation_method = %d\n"
                  "    activation_ga_slope = %lx, activation_channel_shared = %d\n"
                  "    right_shift_width = %d, bn_right_shift_width = %d, "
                  "    scale_right_shift_width = %d\n",
                  result_add, do_bias, do_bn, do_scale, do_scale_bias, do_activation,
                  activation_method, activation_ga_slope, activation_channel_shared,
                  right_shift_width, bn_right_shift_width, scale_right_shift_width));

  int ic = groups;
  int oc = groups;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (input_h + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (input_w + pad_left + pad_right - kw_ext) / stride_w + 1;

  int bias_blobs_num = 0;
  int oc_shape_align_blobs_num = 0;
  int ofmap_blobs_num = 1;
  if (do_bias) {
    bias_blobs_num++;
  }
  if (do_bn) {
    bias_blobs_num += 1;
    oc_shape_align_blobs_num++;
  }
  if (do_scale) {
    oc_shape_align_blobs_num++;
  }
  if (do_scale_bias) {
    bias_blobs_num++;
  }
  if (do_activation && activation_method == PRELU) {
    ASSERT(0);
    bias_blobs_num++;
  }

  int nsecs = 1, csecs = 1;
  bool ret = _split_nc(ctx, input_n, ic, input_h, input_w, oc, oh, ow, kh, kw, dilation_h, stride_h,
                       bias_blobs_num, oc_shape_align_blobs_num, ofmap_blobs_num, &nsecs, &csecs);
  // TODO test split on n
  ASSERT(ret == true);
  int n_step = ceiling_func(input_n, nsecs);
  int c_step = ceiling_func(oc, csecs);

  bmk1880v2_tensor_lmem_shape_t oc_shape_ = ctx.shape_t4(1, c_step, 1, 1);
  bmk1880v2_tensor_lmem_shape_t ifmap_shape_ = ctx.shape_t4(n_step, c_step, input_h, input_w);
  bmk1880v2_tensor_lmem_shape_t ofmap_shape_ = ctx.shape_t4(n_step, c_step, oh, ow);

  bmk1880v2_tensor_lmem_t *tl_weight = nullptr, *tl_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_bn_mean = nullptr, *tl_bn_variance = nullptr;
  bmk1880v2_tensor_lmem_t *tl_scale = nullptr, *tl_scale_bias = nullptr;
  bmk1880v2_tensor_lmem_t *tl_slope = nullptr;

  if (do_scale) {
    tl_scale = ctx.lmem_alloc_tensor(oc_shape_, FMT_I8, 1);  // EU-aligned
  }

  if (do_bn) {
    tl_bn_variance = ctx.lmem_alloc_tensor(oc_shape_, FMT_I8, 1);                  // EU-aligned
    tl_bn_mean = ctx.lmem_alloc_tensor(ctx.shape_t4(2, c_step, 1, 1), FMT_I8, 0);  // Not EU-aligned
  }

  if (do_bias) {
    bmk1880v2_tensor_lmem_shape_t bias_shape = ctx.shape_t4(2, c_step, 1, 1);  // EU-aligned
    tl_bias = ctx.lmem_alloc_tensor(bias_shape, FMT_I8, 0);                    // Not EU-aligned
    LLVM_DEBUG(llvm::errs() << llvm::format("tl_bias (%d %d %d %d) address 0x%x\n", tl_bias->shape.n,
                                          tl_bias->shape.c, tl_bias->shape.h, tl_bias->shape.w,
                                          tl_bias->start_address));
  }

  if (do_scale_bias) {
    tl_scale_bias =
        ctx.lmem_alloc_tensor(ctx.shape_t4(2, c_step, 1, 1), FMT_I8, 0);  // Not EU-aligned
  }
  if (do_activation && activation_method == PRELU) {
    ASSERT(0);  // TODO(wwcai)
  }
  if (do_activation && activation_method == RELU && activation_arg[0] != 0.0f) {
    ASSERT(0);
  }
  bmk1880v2_tensor_tgmem_stride_t bias_stride = {static_cast<u32>(input_c), 1, 1};

  for (int c_pos = 0; c_pos < oc; c_pos += c_step) {
    int cur_c = math_min(oc - c_pos, c_step);

    u64 coeff_offset = c_pos * INT8_SIZE;
    bmk1880v2_tensor_lmem_shape_t coeff_shape_ = ctx.shape_t4(1, cur_c, 1, 1);
    if (do_bias) {
      bmk1880v2_tensor_lmem_shape_t bias_coeff_shape_ = ctx.shape_t4(2, cur_c, 1, 1);

      // tl_reshape(tl_bias, bias_coeff_shape_);
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_bias->shape = bias_coeff_shape_;
      tl_bias->stride = ctx.tensor_lmem_default_stride(tl_bias->shape, 0);  // Not EU-aligned

      ctx.tdma_load_stride(tl_bias, ga_bias + coeff_offset, bias_stride, CTRL_WEIGHT);
    }

    if (do_bn) {
      // tl_reshape(tl_bn_mean, shape_t4(2, cur_c, 1, 1));
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_bn_mean->shape = ctx.shape_t4(2, cur_c, 1, 1);
      tl_bn_mean->stride = ctx.tensor_lmem_default_stride(tl_bn_mean->shape, 0);  // Not EU-aligned

      ctx.tdma_load_stride(tl_bn_mean, ga_bn_mean + coeff_offset, bias_stride, CTRL_WEIGHT);

      // tl_reshape(tl_bn_variance, coeff_shape_);
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_bn_variance->shape = coeff_shape_;
      tl_bn_variance->stride =
          ctx.tensor_lmem_default_stride(tl_bn_variance->shape, 1);  // EU-aligned
      ctx.tdma_load(tl_bn_variance, ga_bn_variance + coeff_offset, CTRL_WEIGHT);
    }
    if (do_scale) {
      // tl_reshape(tl_scale, coeff_shape_);
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_scale->shape = coeff_shape_;
      tl_scale->stride = ctx.tensor_lmem_default_stride(tl_scale->shape, 1);  // EU-aligned

      ctx.tdma_load(tl_scale, ga_scale + coeff_offset, CTRL_WEIGHT);
    }
    if (do_scale_bias) {
      // tl_reshape(tl_scale_bias, shape_t4(2, cur_c, 1, 1));
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_scale_bias->shape = ctx.shape_t4(2, cur_c, 1, 1);
      tl_scale_bias->stride =
          ctx.tensor_lmem_default_stride(tl_scale_bias->shape, 0);  // Not EU-aligned

      ctx.tdma_load_stride(tl_scale_bias, ga_scale_bias + coeff_offset, bias_stride, CTRL_WEIGHT);
    }
    if (do_activation && activation_method == PRELU) {
      ASSERT(0);  // TODO(wwcai)
    }

    for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
      int cur_n = math_min(input_n - n_pos, n_step);

      bmk1880v2_tensor_lmem_t *tl_ofmap =
          ctx.lmem_alloc_tensor(ctx.shape_t4(cur_n, cur_c, oh, ow), FMT_I8, 1);  // EU-aligne
      u64 ofmap_offset = ga_ofmap + (n_pos * oc * oh * ow + c_pos * oh * ow) * INT8_SIZE;
      LLVM_DEBUG(llvm::errs() << llvm::format("tl_ofmap (%d %d %d %d) address 0x%x\n", cur_n, cur_c,
                                            oh, ow, tl_ofmap->start_address));

      if (result_add) {
        ASSERT(0);  // TODO(wwcai)
      }

      bmk1880v2_tensor_lmem_t *tl_ifmap = ctx.lmem_alloc_tensor(
          ctx.shape_t4(cur_n, cur_c, input_h, input_w), FMT_I8, 1);  // EU-aligned

      u64 ifmap_offset =
          ga_ifmap + (n_pos * ic * input_h * input_w + c_pos * input_h * input_w) * INT8_SIZE;
      LLVM_DEBUG(llvm::errs() << llvm::format("tl_ifmap (%d %d %d %d) address 0x%x\n", cur_n, cur_c,
                                            input_h, input_w, tl_ifmap->start_address));

      ctx.tdma_load(tl_ifmap, ifmap_offset, CTRL_NEURON);

      u64 weight_offset = ga_weight + (c_pos * kh * kw) * INT8_SIZE;
      tl_weight = ctx.lmem_alloc_tensor(ctx.shape_t4(1, cur_c, kh, kw), FMT_I8, 1);  // EU-aligned

      LLVM_DEBUG(llvm::errs() << llvm::format("tl_weight (%d %d %d %d) address 0x%x\n", 1, c_step, kh,
                                            kw, tl_weight->start_address));
      ctx.tdma_load(tl_weight, weight_offset, CTRL_WEIGHT);
      {
        bmk1880v2_tiu_depthwise_convolution_param_t param;
        param.ofmap = tl_ofmap;
        param.ifmap = tl_ifmap;
        param.weight = tl_weight;
        param.bias = tl_bias;
        param.ins_h = 0;
        param.ins_last_h = 0;
        param.ins_w = 0;
        param.ins_last_w = 0;
        param.pad_top = pad_top;
        param.pad_bottom = pad_bottom;
        param.pad_left = pad_left;
        param.pad_right = pad_right;
        param.stride_h = stride_h;
        param.stride_w = stride_w;
        param.rshift_bits = right_shift_width;
        param.relu_enable = 0;
        param.layer_id = layer_id;
        ctx.tiu_depthwise_convolution(&param);
      }
      ctx.lmem_free_tensor(tl_weight);
      ctx.lmem_free_tensor(tl_ifmap);

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
      param.relu_enable = 0;
      param.layer_id = layer_id;
      if (do_bn) {
        // out(n,c,h,w) = in(n,c,h,w) * variance(1,c,1,1) + mean(1,c,1,1)
        param.ofmap = tl_ofmap;
        param.ifmap = tl_ofmap;
        param.weight = tl_bn_variance;
        param.bias = tl_bn_mean;
        param.rshift_bits = bn_right_shift_width;
        ctx.tiu_depthwise_convolution(&param);
      }
      if (do_scale && do_scale_bias) {
        // computing x * scale + bias
        param.ofmap = tl_ofmap;
        param.ifmap = tl_ofmap;
        param.weight = tl_scale;
        param.bias = tl_scale_bias;
        param.rshift_bits = scale_right_shift_width;
        ctx.tiu_depthwise_convolution(&param);
      } else if (do_scale) {
        param.ofmap = tl_ofmap;
        param.ifmap = tl_ofmap;
        param.weight = tl_scale;
        param.bias = nullptr;
        param.rshift_bits = scale_right_shift_width;
        ctx.tiu_depthwise_convolution(&param);
      } else if (do_scale_bias) {
        ASSERT(0);  // TODO(zakk)
      }
      if (do_activation) {
        switch (activation_method) {
          case RELU: {
            bmk1880v2_tiu_element_wise_max_param_t p13;
            p13.max = tl_ofmap;
            p13.a = tl_ofmap;
            p13.b_is_const = 1;
            p13.b_is_signed = 1;
            p13.b_val = 0;
            p13.layer_id = layer_id;
            ctx.tiu_element_wise_max(&p13);

            break;
          }
          case PRELU:
            ASSERT(0);  // TODO(wwcai)
            break;
          default:
            ASSERT(0);
        }
      }

      ctx.tdma_store(tl_ofmap, ofmap_offset, CTRL_NEURON);

      ctx.lmem_free_tensor(tl_ofmap);
    }
  }

  if (do_activation && activation_method == PRELU) {
    ASSERT(0);  // TODO(wwcai)
  }
  if (do_scale_bias) {
    ctx.lmem_free_tensor(tl_scale_bias);
  }
  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias);
  }
  if (do_bn) {
    ctx.lmem_free_tensor(tl_bn_mean);
    ctx.lmem_free_tensor(tl_bn_variance);
  }
  if (do_scale) {
    ctx.lmem_free_tensor(tl_scale);
  }
}
#endif  // this is dwconv split scheme

static BM1880v2ConvFixed *find_best_conv_method(const BM1880v2BackendContext &ctx,
                                                ConvFixed_ARGS &args) {
  BM1880v2ConvFixed *conv_parallel = new BM1880v2ConvFixedParallel(args);
  BM1880v2ConvFixed *conv_serial = new BM1880v2ConvFixedSerial(args);
  int slices_num_parallel = conv_parallel->split(ctx);
  int slices_num_serial = conv_serial->split(ctx);

  if (slices_num_parallel == SPLIT_FAILED ||
      (float(slices_num_parallel) / float(slices_num_serial)) > 1.5 ||
      (args.do_activation && (args.activation_method == PRELU)) ||
      (args.do_activation && (args.activation_method == RELU) &&
       (args.activation_arg[0] != 0.0f))) {  // TODO: support PRELU in ConvFixedParallel

    delete conv_parallel;
    LLVM_DEBUG(llvm::errs() << "do conv_serial\n");
    return conv_serial;
  } else {
    delete conv_serial;
    LLVM_DEBUG(llvm::errs() << "do conv_parallel\n");
    return conv_parallel;
  }
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
    int scale_right_shift_width, bool use_winograd) {
  int nsec = ceiling_func_shift(input_n, NODECHIP_SHIFT);

  // this message is too long for llvm::format, so seperate it
  LLVM_DEBUG(llvm::errs() << llvm::format("bmnet_conv_parallel_fixed_forward_bmkernel:\n"
                                        "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
                                        "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
                                        "    kernel = (%d, %d), dilation = (%d, %d)\n"
                                        "    pad = (%d, %d,%d, %d), stride = (%d, %d)\n",
                                        ga_ifmap, ga_ofmap, ga_weight, ga_bias, nsec, input_c,
                                        input_h, input_w, groups, output_c, kh, kw, dilation_h,
                                        dilation_w, pad_top, pad_bottom, pad_left, pad_right,
                                        stride_h, stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "    result_add = %d, do_bias = %d, do_bn = %d, do_scale = %d\n"
                  "    do_scale_bias = %d, do_activation = %d, activation_method %d\n",
                  result_add, do_bias, do_bn, do_scale, do_scale_bias, do_activation,
                  activation_method));

  // depthwise conv // TODO(wwcai): parallel version
  if (input_c == groups && output_c == groups && 1 != groups) {
    return node_depthwise_conv_forward_bmkernel_serial(
        ctx, layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, ga_bn_mean, ga_bn_variance, ga_scale,
        ga_scale_bias, input_n, input_c, input_h, input_w, groups, output_c, kh, kw, dilation_h,
        dilation_w, pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, result_add,
        do_bias, do_bn, do_scale, do_scale_bias, do_activation, bn_scale, bn_eps, activation_method,
        activation_arg, activation_ga_slope, activation_channel_shared, right_shift_width,
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

//}
