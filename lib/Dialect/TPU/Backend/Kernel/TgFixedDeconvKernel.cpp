/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedDeconvKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "kernel_deconv"
#define DEBUG_SPLIT "kernel_deconv_split"

static void dump(cvk_tiu_convolution_param_t &param, int ifmap_offset,
                 int weight_offset, int ofmap_offset) {

  LLVM_DEBUG(llvm::errs() << llvm::format("ins_h: %d, ins_last_h: %d, "
                  "ins_w: %d, ins_last_w: %d, pad_top: %d, pad_bottom: %d, "
                  "pad_left: %d, pad_right: %d, dilation_h: %d, "
                  "dilation_w: %d\n", param.ins_h,  param.ins_last_h,
                  param.ins_w, param.ins_last_w, param.pad_top,
                  param.pad_bottom, param.pad_left, param.pad_right,
                  param.dilation_h, param.dilation_w););

  LLVM_DEBUG(llvm::errs() << llvm::format("input_offset: %d, "
                            "input shape: (%d, %d, %d, %d), "
                            "input stride: (%d, %d, %d, 1)\n",
                            ifmap_offset,
                            param.ifmap->shape.n, param.ifmap->shape.c,
                            param.ifmap->shape.h, param.ifmap->shape.w,
                            param.ifmap->stride.n,
                            param.ifmap->stride.c, param.ifmap->stride.h););

  LLVM_DEBUG(llvm::errs() << llvm::format("weight_offset: %d, "
                          "weight shape: (%d, %d, %d, %d), "
                          "weight stride:(%d, %d, %d, 1)\n",
                          weight_offset,
                          param.weight->shape.n, param.weight->shape.c,
                          param.weight->shape.h, param.weight->shape.w,
                          param.weight->stride.n,
                          param.weight->stride.c, param.weight->stride.h););

  LLVM_DEBUG(llvm::errs() << llvm::format("output_offset: %d, "
                            "output shape: (%d, %d, %d, %d), "
                            "output stride:(%d, %d, %d, 1)\n\n",
                            ofmap_offset,
                            param.ofmap->shape.n, param.ofmap->shape.c,
                            param.ofmap->shape.h, param.ofmap->shape.w,
                            param.ofmap->stride.n,
                            param.ofmap->stride.c, param.ofmap->stride.h););
}

static int __calcuate_occuppied_lmem(const CviBackendContext &ctx, int ic, int ih, int iw,
                                     int oc, int output_h, int output_w, int kh, int kw) {
  int ic_per_npu = ceiling_func(ic, NPU_NUM);
  int oc_per_npu = ceiling_func(oc, NPU_NUM);

  int bias_blob_size = oc_per_npu * sizeof(uint8_t) * 2;  // INT16
  int weight_blob_size = ic * oc_per_npu * kh * kw * sizeof(uint8_t);

  int ifmap_blob_size = ic_per_npu * ALIGN(ih * iw, EU_NUM);
  int ofmap_blob_size = oc_per_npu * ALIGN(output_h * output_w, EU_NUM);

  int neuron_size_per_n = ifmap_blob_size + ofmap_blob_size;
  int coeff_size_per_n = weight_blob_size + bias_blob_size;

  return neuron_size_per_n + coeff_size_per_n;
}

static int __calcuate_depthwise_occuppied_lmem(const CviBackendContext &ctx, int ic, int ih,
                                               int iw, int oc, int output_h, int output_w, int kh,
                                               int kw, bool do_bias, bool do_chan_quan) {
  int ic_per_npu = ceiling_func(ic, NPU_NUM);
  int oc_per_npu = ceiling_func(oc, NPU_NUM);

  int weight_blob_size = oc_per_npu * kh * kw * sizeof(uint8_t);
  int ifmap_blob_size = ic_per_npu * ALIGN(ih * iw, EU_NUM);
  int ofmap_blob_size = oc_per_npu * ALIGN(output_h * output_w, EU_NUM);

  int neuron_size_per_n = ifmap_blob_size + ofmap_blob_size;

  int coeff_size_per_n = weight_blob_size;
  if (do_chan_quan) {
    int param_size = ctx.chan_quan_param_size(do_bias);
    int param_blob_size = oc_per_npu * param_size;
    coeff_size_per_n += param_blob_size;
  } else if (do_bias) {
    int bias_blob_size = oc_per_npu * sizeof(uint8_t) * 2;  // INT16
    coeff_size_per_n += bias_blob_size;
  }

  return neuron_size_per_n + coeff_size_per_n;
}

bool do_depthwise_split(const CviBackendContext &ctx, int n, int ic, int ih, int iw, int oc,
                        int output_h, int output_w, int kh, int kw, int pad_h_top, int pad_h_bottom,
                        int stride_h, int dilation_h, int &nsecs, int &ocsecs, int &icsecs,
                        int &hsecs, bool do_bias, bool do_chan_quan) {
  const int kh_ext = dilation_h * (kh - 1) + 1;
  int total_size_per_n = __calcuate_depthwise_occuppied_lmem(
      ctx, ic, ih, iw, oc, output_h, output_w, kh, kw, do_bias, do_chan_quan);

  nsecs = ocsecs = icsecs = hsecs = 1;

  // no need to split.
  if (total_size_per_n * n <= LOCAL_MEM_SIZE) {
    LLVM_DEBUG(llvm::errs() << "no need to split\n");
    return true;
  }

  // localmem can host at least one n, no need to slice c
  if (total_size_per_n <= LOCAL_MEM_SIZE) {
    int slice_n = LOCAL_MEM_SIZE / total_size_per_n;
    assert(slice_n >= 1);
    nsecs = ceiling_func(n, slice_n);
    LLVM_DEBUG(llvm::errs() << llvm::format("split on n, n_slices = %d\n", nsecs));
    return true;
  }

  nsecs = n;
  int cur_oh = output_h, cur_ih = ih;

  int h_sliced = output_h;
  for (; h_sliced > 0; h_sliced--) {
    hsecs = ceiling_func(output_h, h_sliced);
    cur_oh = ceiling_func(output_h, hsecs);
    cur_ih = (cur_oh + pad_h_top + pad_h_bottom - kh_ext) / stride_h + 2;

    total_size_per_n = __calcuate_depthwise_occuppied_lmem(ctx, ic, cur_ih, iw, oc, cur_oh,
        output_w, kh, kw, do_bias, do_chan_quan);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      //hsecs = ceiling_func(output_h, 4);
      // localmem can host at least one n, no need to slice c
      LLVM_DEBUG(llvm::errs()<< llvm::format("split on n & h, n_slices = %d, h_slices=%d\n", nsecs, hsecs));
      return true;
    }

    cur_oh = ceiling_func(output_h, hsecs);
    cur_ih = (cur_oh + pad_h_top + pad_h_bottom - kh_ext) / stride_h + 2;

    // split ic = oc
    while (ocsecs <= oc) {
      int cur_oc = ceiling_func(oc, ocsecs);

      total_size_per_n = __calcuate_depthwise_occuppied_lmem(ctx, cur_oc, cur_ih, iw, cur_oc, cur_oh,
          output_w, kh, kw, do_bias, do_chan_quan);
      if (total_size_per_n <= LOCAL_MEM_SIZE) {
        // localmem can host at least one n, no need to slice c
        LLVM_DEBUG(llvm::errs()<< llvm::format(
              "split on n & c, n_slices = %d, c_slices=(%d, %d), h_slices=%d\n", nsecs,
              icsecs, ocsecs, hsecs));
        return true;
      }

      ocsecs += 1;
    }
  }
  return false;
}

bool do_split(const CviBackendContext &ctx, int n, int ic, int ih, int iw, int oc,
              int output_h, int output_w, int kh, int kw, int pad_h_top, int pad_h_bottom,
              int stride_h, int dilation_h, int &nsecs, int &ocsecs, int &icsecs, int &hsecs) {
  const int kh_ext = dilation_h * (kh - 1) + 1;
  int total_size_per_n = __calcuate_occuppied_lmem(ctx, ic, ih, iw, oc, output_h, output_w, kh, kw);

  nsecs = ocsecs = icsecs = hsecs = 1;

  // no need to split.
  if (total_size_per_n * n <= LOCAL_MEM_SIZE) {
    LLVM_DEBUG(llvm::errs() << "no need to split\n");
    return true;
  }

  // localmem can host at least one n, no need to slice c
  if (total_size_per_n <= LOCAL_MEM_SIZE) {
    int slice_n = LOCAL_MEM_SIZE / total_size_per_n;
    assert(slice_n >= 1);
    nsecs = ceiling_func(n, slice_n);
    LLVM_DEBUG(llvm::errs()<< llvm::format("split on n, n_slices = %d\n", nsecs));
    return true;
  }

  nsecs = n;
  int cur_oh = output_h, cur_ih = ih;

  if (output_h > 20) {
    hsecs = ceiling_func(output_h, 20);
    cur_oh = ceiling_func(output_h, hsecs);
    cur_ih = (cur_oh + pad_h_top + pad_h_bottom - kh_ext) / stride_h + 2;

    total_size_per_n = __calcuate_occuppied_lmem(ctx, ic, cur_ih, iw, oc, cur_oh, output_w, kh, kw);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      // localmem can host at least one n, no need to slice c
      LLVM_DEBUG(llvm::errs()<< llvm::format("split on n & h, n_slices = %d, h_slices=%d\n", nsecs, hsecs));
      return true;
    }
  }

  cur_oh = ceiling_func(output_h, hsecs);
  cur_ih = (cur_oh + pad_h_top + pad_h_bottom - kh_ext) / stride_h + 2;

  // split ic and oc
  while (icsecs <= ic) {
    int cur_ic = ceiling_func(ic, icsecs);

    total_size_per_n =
        __calcuate_occuppied_lmem(ctx, cur_ic, cur_ih, iw, oc, cur_oh, output_w, kh, kw);
    if (total_size_per_n <= LOCAL_MEM_SIZE) {
      // localmem can host at least one n, no need to slice c
      LLVM_DEBUG(llvm::errs()<< llvm::format(
                           "split on n & c, n_slices = %d, c_slices=(%d, %d), h_slices=%d\n", nsecs,
                           icsecs, ocsecs, hsecs));
      return true;
    }

    while (ocsecs <= oc) {
      int cur_oc = ceiling_func(oc, ocsecs);
      total_size_per_n =
          __calcuate_occuppied_lmem(ctx, cur_ic, cur_ih, iw, cur_oc, cur_oh, output_w, kh, kw);
      if (total_size_per_n <= LOCAL_MEM_SIZE) {
        LLVM_DEBUG(llvm::errs()<< llvm::format(
                                  "split on n & c, n_slices = %d, c_slices=(%d, %d), h_slices=%d\n",
                                  nsecs, icsecs, ocsecs, hsecs));
        return true;
      }
      ocsecs += 1;
    }

    icsecs += 1;
  }

  return false;
}

void node_depthwise_cvi_backend_tg_int8_deconv_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int output_h,
    int output_w, int kh, int kw, int dh, int dw, int pad_h_top, int pad_h_bottom, int pad_w_left,
    int pad_w_right, int stride_h, int stride_w, bool do_bias, bool result_add, bool do_relu,
    int right_shift_width, bool use_winograd, int right_shift_array_len, gaddr_t ga_per_channel) {
  //assert(input_n == 1);
  int kh_ext = (kh - 1) * dh + 1;
  int kw_ext = (kw - 1) * dw + 1;
  int ins_h = stride_h - 1;
  int ins_w = stride_w - 1;
  int height_insert0 = (input_h - 1) * stride_h + 1;
  int ins_w_last = (output_w + pad_w_left + pad_w_right - kw_ext) % stride_w;
  bool do_chan_quan = right_shift_array_len ? true : false;

  int ic = input_c;
  int oc = output_c;
  assert(ic == oc);
  assert(ic == groups);
  int nsecs, ocsecs, icsecs, hsecs;
  bool ret = do_depthwise_split(ctx, input_n, ic, input_h, input_w, oc, output_h, output_w, kh, kw,
                                pad_h_top, pad_h_bottom, stride_h, dh, nsecs, ocsecs, icsecs, hsecs,
                                do_bias, do_chan_quan);
  LLVM_DEBUG(llvm::errs() << "<TRY SPLIT> [Deconv Depthwise ConvFixedSerial::split], <n:" << nsecs
                        << ", oc:" << ocsecs << ", ic:" << icsecs << ", h:" << hsecs << ">, total "
                        << nsecs * ocsecs * icsecs * hsecs << " slices");

  assert(ret == true);
  pad_h_top = kh_ext - pad_h_top - 1;
  pad_h_bottom = kh_ext - pad_h_bottom - 1;
  pad_w_left = kw_ext - pad_w_left - 1;
  pad_w_right = kw_ext - pad_w_right - 1;

  int n_step = ceiling_func(input_n, nsecs);
  int oc_step = ceiling_func(oc, ocsecs);
  int oh_step = ceiling_func(output_h, hsecs);

  cvk_tl_t *tl_weight = nullptr;
  cvk_tl_t *tl_bias = nullptr;
  cvk_tl_t *tl_quan_param = nullptr;

  cvk_tg_stride_t ofmap_gstride = {static_cast<uint32_t>(output_c) * output_h * output_w,
                                                   static_cast<uint32_t>(output_h) * output_w,
                                                   static_cast<uint32_t>(output_w)};

  cvk_tg_stride_t ifmap_gstride = {static_cast<uint32_t>(input_c) * input_h * input_w,
                                                   static_cast<uint32_t>(input_h) * input_w,
                                                   static_cast<uint32_t>(input_w)};
  cvk_tg_stride_t bias_stride = {static_cast<uint32_t>(output_c), 1, 1};

  for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
    int cur_oc = std::min(oc - oc_pos, oc_step);
    uint64_t coeff_offset = (oc_pos);

    if (do_chan_quan) {
      // Channel quantization
      uint32_t param_size = ctx.chan_quan_param_size(do_bias);

      cvk_tl_shape_t tl_quan_param_shape = {1, static_cast<uint32_t>(cur_oc), 1,
                                                           param_size};
      tl_quan_param = ctx.lmem_alloc_tensor(tl_quan_param_shape, CVK_FMT_I8, /*eu_align=*/0);

      ctx.tdma_load(tl_quan_param, ga_per_channel + oc_pos * param_size);
    } else if (do_bias) {
      cvk_tl_shape_t bias_coeff_shape_ = ctx.tl_shape_t4(2, cur_oc, 1, 1);
      tl_bias = ctx.lmem_alloc_tensor(bias_coeff_shape_, CVK_FMT_I8, /*eu_align=*/0);
      ctx.tdma_load_stride(tl_bias, ga_bias + coeff_offset, bias_stride);
    }

    for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
      int cur_n = std::min(input_n - n_pos, n_step);
      for (int oh_pos = 0; oh_pos < output_h; oh_pos += oh_step) {
        int cur_oh = std::min(output_h - oh_pos, oh_step);
        int oh_top = oh_pos;
        int if_pad_h_t = oh_top;
        int if_pad_h_b = oh_top + cur_oh + kh_ext - 1;
        int if_insert_h_t = 0;
        int pad_h_t = 0;
        if (if_pad_h_t < pad_h_top) {
          pad_h_t = pad_h_top - if_pad_h_t;
        } else {
          if_insert_h_t = if_pad_h_t - pad_h_top;
        }
        int if_insert_h_b = height_insert0;
        int pad_h_b = 0;
        if ((if_pad_h_b - pad_h_top) < height_insert0) {
          if_insert_h_b = if_pad_h_b - pad_h_top;
        } else {
          pad_h_b = if_pad_h_b - height_insert0 - pad_h_top;
        }

        int ih_top = (if_insert_h_t + stride_h - 1) / stride_h;
        int ih_bot = (if_insert_h_b + stride_h - 1) / stride_h;

        int cur_ih = ih_bot - ih_top;

        int hinsert0_t = if_insert_h_t % stride_h == 0 ? 0 : (stride_h - if_insert_h_t % stride_h);
        int hinsert0_b = (if_insert_h_b + stride_h - 1) % stride_h;

        cvk_tl_t *tl_ofmap =
            ctx.lmem_alloc_tensor(ctx.tl_shape_t4(cur_n, cur_oc, cur_oh, output_w), CVK_FMT_I8, /*eu_align=*/1);

        uint64_t ofmap_offset = ga_ofmap + (n_pos * output_c * output_h * output_w +
                                       oc_pos * output_h * output_w + oh_top * output_w);

        if (result_add) {
          ctx.tdma_load_stride(tl_ofmap, ofmap_offset, ofmap_gstride);
        }

        {
          int ic_pos = oc_pos;
          int cur_ic = cur_oc;

          cvk_tl_t *tl_ifmap =
              ctx.lmem_alloc_tensor(ctx.tl_shape_t4(cur_n, cur_ic, cur_ih, input_w), CVK_FMT_I8, /*eu_align=*/1);
          uint64_t ifmap_offset = ga_ifmap + (n_pos * input_c * input_h * input_w +
                                         ic_pos * input_h * input_w + ih_top * input_w);
          ctx.tdma_load_stride(tl_ifmap, ifmap_offset, ifmap_gstride);

          tl_weight = ctx.lmem_alloc_tensor(ctx.tl_shape_t4(1, cur_ic, kh, kw), CVK_FMT_I8, /*eu_align=*/1);
          uint64_t weight_offset = ga_weight + sizeof(uint8_t) * (oc_pos * kh * kw);

          cvk_tg_stride_t stride = {
              static_cast<uint32_t>(oc) * kh * kw, static_cast<uint32_t>(kh) * kw,
              static_cast<uint32_t>(kw)};  // TODO(wwcai): test SPLIT
          ctx.tdma_load_stride(tl_weight, weight_offset, stride);

          if (do_chan_quan) {
            // tiu logical shape != allocated local memory shape
            cvk_tl_t tl_chl_quan_tiu = {0};
            tl_chl_quan_tiu.start_address = tl_quan_param->start_address;
            tl_chl_quan_tiu.fmt = tl_quan_param->fmt;
            tl_chl_quan_tiu.shape = {1, tl_quan_param->shape.c, 1, 1};
            tl_chl_quan_tiu.stride =
                ctx.tl_default_stride(tl_chl_quan_tiu.shape, CVK_FMT_I8,
                                      /*eu_align=*/0);

            cvk_tiu_depthwise_convolution_param_t param = {0};
            memset(&param, 0, sizeof(param));
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ifmap;
            param.weight = tl_weight;
            param.chl_quan_param = &tl_chl_quan_tiu;
            param.ins_h = ins_h;
            param.ins_last_h = 0;
            param.ins_w = ins_w;
            param.ins_last_w = ins_w_last;
            param.stride_h = 1;
            param.stride_w = 1;
            param.dilation_h = dh;
            param.dilation_w = dw;
            param.pad_top = hinsert0_t + pad_h_t;
            param.pad_bottom = hinsert0_b + pad_h_b;
            param.pad_left = pad_w_left;
            param.pad_right = pad_w_right;
            param.has_bias = do_bias;
            param.relu_enable = do_relu;
            param.layer_id = layer_id;
            ctx.tiu_depthwise_convolution(&param);

          } else {
            cvk_tiu_depthwise_pt_convolution_param_t param = {0};
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ifmap;
            param.weight = tl_weight;
            param.bias = tl_bias;
            param.ins_h = ins_h;
            param.ins_last_h = 0;
            param.ins_w = ins_w;
            param.ins_last_w = ins_w_last;
            param.pad_top = hinsert0_t + pad_h_t;
            param.pad_bottom = hinsert0_b + pad_h_b;
            param.pad_left = pad_w_left;
            param.pad_right = pad_w_right;
            param.stride_h = 1;
            param.stride_w = 1;
            param.dilation_h = dh;
            param.dilation_w = dw;
            param.relu_enable = do_relu;
            param.rshift_bits = right_shift_width;
            param.layer_id = layer_id;
            ctx.tiu_pt_depthwise_convolution(&param);
          }
          ctx.lmem_free_tensor(tl_weight);
          ctx.lmem_free_tensor(tl_ifmap);
        }
        ctx.tdma_store_stride(tl_ofmap, ofmap_offset, ofmap_gstride);
        ctx.lmem_free_tensor(tl_ofmap);
      }
    }
    if (tl_quan_param) {
      ctx.lmem_free_tensor(tl_quan_param);
    } else if (tl_bias) {
      ctx.lmem_free_tensor(tl_bias);
    }
  }
}

void cvi_backend_tg_fixed_deconv_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_h, int input_w, int groups, int output_c, int output_h,
    int output_w, int kh, int kw, int dh, int dw, int pad_h_top, int pad_h_bottom, int pad_w_left,
    int pad_w_right, int stride_h, int stride_w, bool do_bias, bool result_add, bool do_relu,
    int right_shift_width, bool use_winograd, int right_shift_array_len, gaddr_t ga_per_channel) {

    LLVM_DEBUG(llvm::errs() << llvm::format(
             "cvi_backend_tg_fixed_deconv_kernel:\n"
             "    layer_id %d\n"
             "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
             "    bottom shape = (%d, %d, %d, %d)\n"
             "    top shape = (%d, %d, %d, %d)\n"
             "    group = %d kernel = (%d, %d), dilation = (%d, %d)\n"
             "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
             layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n,
             input_c, input_h, input_w, input_n, output_c, output_h, output_w,
             groups, kh, kw, dh, dw, pad_h_top, pad_h_bottom,
             pad_w_left, pad_w_right, stride_h, stride_w););

  if (input_c == groups && output_c == groups && groups != 1) {
    node_depthwise_cvi_backend_tg_int8_deconv_kernel(
        ctx, stream_id, inst_id, layer_id, depends, depends_len, ga_ifmap, ga_ofmap, ga_weight,
        ga_bias, input_n, input_c, input_h, input_w, groups, output_c, output_h, output_w, kh, kw,
        dh, dw, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, stride_h, stride_w, do_bias,
        result_add, do_relu, right_shift_width, use_winograd, right_shift_array_len,
        ga_per_channel);
    return;
  }

  int kh_ext = (kh - 1) * dh + 1;
  int kw_ext = (kw - 1) * dw + 1;
  int ins_h = stride_h - 1;
  int ins_w = stride_w - 1;
  int height_insert0 = (input_h - 1) * stride_h + 1;
  int ins_w_last = (output_w + pad_w_left + pad_w_right - kw_ext) % stride_w;
  bool do_chan_quan = right_shift_array_len ? true : false;

  int ic = input_c / groups;
  int oc = output_c / groups;

  int nsecs, ocsecs, icsecs, hsecs;
  bool ret = do_split(ctx, input_n, ic, input_h, input_w, oc, output_h,
                      output_w, kh, kw, pad_h_top, pad_h_bottom, stride_h, dh,
                      nsecs, ocsecs, icsecs, hsecs);

  LLVM_DEBUG(llvm::errs() << "<TRY SPLIT> [ConvFixedSerial::split], <n:"
                          << nsecs << ", oc:" << ocsecs << ", ic:" << icsecs
                          << ", h:" << hsecs << ">, total "
                          << nsecs * ocsecs * icsecs * hsecs << " slices\n");

  assert(ret == true);
  assert(icsecs == 1);

  pad_h_top = kh_ext - pad_h_top - 1;
  pad_h_bottom = kh_ext - pad_h_bottom - 1;
  pad_w_left = kw_ext - pad_w_left - 1;
  pad_w_right = kw_ext - pad_w_right - 1;

  int n_step = ceiling_func(input_n, nsecs);
  int oc_step = ceiling_func(oc, ocsecs);
  int oh_step = ceiling_func(output_h, hsecs);

  cvk_tl_t *tl_weight = nullptr;
  cvk_tl_t *tl_bias = nullptr;
  cvk_tl_t *tl_quan_param = nullptr;

  tl_weight = ctx.lmem_alloc_tensor(ctx.tl_shape_t4(1, oc_step, kh * kw, ic), CVK_FMT_I8,
                                    0);  // Not EU-aligned

  cvk_tg_stride_t ofmap_gstride = {static_cast<uint32_t>(output_c) * output_h * output_w,
                                                   static_cast<uint32_t>(output_h) * output_w,
                                                   static_cast<uint32_t>(output_w)};
  cvk_tg_stride_t ifmap_gstride = {static_cast<uint32_t>(input_c) * input_h * input_w,
                                                   static_cast<uint32_t>(input_h) * input_w,
                                                   static_cast<uint32_t>(input_w)};
  cvk_tg_stride_t bias_stride = {static_cast<uint32_t>(output_c), 1, 1};

  for (int ig = 0; ig < groups; ig++) {
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = std::min(oc - oc_pos, oc_step);
      uint64_t coeff_offset = (ig * oc + oc_pos);

      if (do_chan_quan) {
        // Channel quantization
        uint32_t param_size = ctx.chan_quan_param_size(do_bias);

        cvk_tl_shape_t tl_quan_param_shape = {1, static_cast<uint32_t>(cur_oc), 1,
                                                             param_size};
        tl_quan_param = ctx.lmem_alloc_tensor(tl_quan_param_shape, CVK_FMT_I8, /*eu_align=*/0);

        ctx.tdma_load(tl_quan_param, ga_per_channel + oc_pos * param_size,
                      CviBackendContext::WEIGHT_MEMORY);

      } else if (do_bias) {
        cvk_tl_shape_t bias_coeff_shape_ = ctx.tl_shape_t4(2, cur_oc, 1, 1);
        tl_bias = ctx.lmem_alloc_tensor(bias_coeff_shape_, CVK_FMT_I8, /*eu_align=*/0);
        ctx.tdma_load_stride(tl_bias, ga_bias + coeff_offset, bias_stride);
      }

      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, n_step);
        for (int oh_pos = 0; oh_pos < output_h; oh_pos += oh_step) {
          int cur_oh = std::min(output_h - oh_pos, oh_step);
          int oh_top = oh_pos;
          int if_pad_h_t = oh_top;
          int if_pad_h_b = oh_top + cur_oh + kh_ext - 1;
          int if_insert_h_t = 0;
          int pad_h_t = 0;
          if (if_pad_h_t < pad_h_top) {
            pad_h_t = pad_h_top - if_pad_h_t;
          } else {
            if_insert_h_t = if_pad_h_t - pad_h_top;
          }
          int if_insert_h_b = height_insert0;
          int pad_h_b = 0;
          if ((if_pad_h_b - pad_h_top) < height_insert0) {
            if_insert_h_b = if_pad_h_b - pad_h_top;
          } else {
            pad_h_b = if_pad_h_b - height_insert0 - pad_h_top;
          }

          int ih_top = (if_insert_h_t + stride_h - 1) / stride_h;
          int ih_bot = (if_insert_h_b + stride_h - 1) / stride_h;

          int cur_ih = ih_bot - ih_top;

          int hinsert0_t =
              if_insert_h_t % stride_h == 0 ? 0 : (stride_h - if_insert_h_t % stride_h);
          int hinsert0_b = (if_insert_h_b + stride_h - 1) % stride_h;

          cvk_tl_t *tl_ofmap =
              ctx.lmem_alloc_tensor(ctx.tl_shape_t4(cur_n, cur_oc, cur_oh, output_w), CVK_FMT_I8, /*eu_align=*/1);

          uint64_t ofmap_offset =
              ga_ofmap + (n_pos * output_c * output_h * output_w + ig * oc * output_h * output_w +
                          oc_pos * output_h * output_w + oh_top * output_w);

          if (result_add) {
            ctx.tdma_load_stride(tl_ofmap, ofmap_offset, ofmap_gstride);
          }

          cvk_tl_t *tl_ifmap = ctx.lmem_alloc_tensor(
                ctx.tl_shape_t4(cur_n, ic, cur_ih, input_w), CVK_FMT_I8, 1);

          uint64_t ifmap_offset =
                ga_ifmap + (n_pos * input_c * input_h * input_w + ig * ic * input_h * input_w +
                            + ih_top * input_w);

          ctx.tdma_load_stride(tl_ifmap, ifmap_offset, ifmap_gstride);

          uint64_t weight_offset = ga_weight + sizeof(uint8_t) * (oc_pos * ic * kh * kw +
                                                         ig * oc * ic * kh * kw);
          tl_weight->shape = ctx.tl_shape_t4(1, cur_oc, kh * kw, ic);
          tl_weight->stride =
              ctx.tl_default_stride(tl_weight->shape, CVK_FMT_I8, 0);  // Not EU-aligned

          cvk_tg_stride_t stride = {
              static_cast<uint32_t>(cur_oc) * kh * kw * ic, static_cast<uint32_t>(kh) * kw * ic,
              static_cast<uint32_t>(ic)};  // TODO(wwcai): test SPLIT
          ctx.tdma_load_stride(tl_weight, weight_offset, stride);
          // Reshape
          // bmk does not keep eu-align info, user need to update stride if shape changed
          // tl_reshape(tl_weight, tl_shape_t4(cur_ic, cur_oc, kh, kw));
          tl_weight->shape = ctx.tl_shape_t4(ic, cur_oc, kh, kw);
          tl_weight->stride =
              ctx.tl_default_stride(tl_weight->shape, CVK_FMT_I8, 0);  // Not EU-aligned

          if (do_chan_quan) {
            cvk_tl_t tl_chl_quan_tiu = {0};
            tl_chl_quan_tiu.start_address = tl_quan_param->start_address;
            tl_chl_quan_tiu.fmt = tl_quan_param->fmt;
            tl_chl_quan_tiu.shape = {1, tl_quan_param->shape.c, 1, 1};
            tl_chl_quan_tiu.stride =
                ctx.tl_default_stride(tl_chl_quan_tiu.shape, CVK_FMT_I8, 0);

            cvk_tiu_convolution_param_t param = {0};
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ifmap;
            param.weight = tl_weight;
            param.chl_quan_param = &tl_chl_quan_tiu;

            param.ins_h = (tl_ifmap->shape.h > 1) ? ins_h : 0;
            param.ins_last_h = 0;
            param.ins_w = (tl_ifmap->shape.w > 1) ? ins_w : 0;
            param.ins_last_w = ins_w_last;
            param.pad_top = hinsert0_t + pad_h_t;
            param.pad_bottom = hinsert0_b + pad_h_b;
            param.pad_left = pad_w_left;
            param.pad_right = pad_w_right;
            param.stride_h = 1;
            param.stride_w = 1;
            param.dilation_h = dh;
            param.dilation_w = dw;
            param.relu_enable = do_relu;
            param.ps32_mode = 0;
            param.w_is_const = 0;
            param.layer_id = layer_id;
            param.has_bias = do_bias ? 1 : 0;
            ctx.tiu_convolution(&param);
            dump(param, ifmap_offset, weight_offset, ofmap_offset);
          } else {
            cvk_tiu_pt_convolution_param_t param = {0};
            param.ofmap = tl_ofmap;
            param.ifmap = tl_ifmap;
            param.weight = tl_weight;
            param.bias = tl_bias;
            param.ins_h = ins_h;
            param.ins_last_h = 0;
            param.ins_w = ins_w;
            param.ins_last_w = ins_w_last;
            param.pad_top = hinsert0_t + pad_h_t;
            param.pad_bottom = hinsert0_b + pad_h_b;
            param.pad_left = pad_w_left;
            param.pad_right = pad_w_right;
            param.stride_h = 1;
            param.stride_w = 1;
            param.dilation_h = dh;
            param.dilation_w = dw;
            param.relu_enable = do_relu;
            param.rshift_bits = right_shift_width;
            param.ps32_mode = 0;
            param.w_is_const = 0;
            param.layer_id = layer_id;
            ctx.tiu_pt_convolution(&param);
          }
          ctx.tdma_store_stride(tl_ofmap, ofmap_offset, ofmap_gstride);
          ctx.lmem_free_tensor(tl_ifmap);
          ctx.lmem_free_tensor(tl_ofmap);
        }
      }
      if (tl_quan_param) {
        ctx.lmem_free_tensor(tl_quan_param);
      } else if (tl_bias) {
        ctx.lmem_free_tensor(tl_bias);
      }
    }
  }

  ctx.lmem_free_tensor(tl_weight);
}
