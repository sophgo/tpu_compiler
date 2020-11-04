/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedScaleKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "bm1880v2_kernel_scale"
#define DEBUG_SPLIT "bm1880v2_kernel_scale_split"

#define MAX_W (1 << 11)
// align to EU_NUM point, return in unit of byte

void cvi_backend_tg_fixed_scale_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t scale_gaddr, gaddr_t bias_gaddr, gaddr_t output_gaddr, int input_n,
    int input_c, int input_h, int input_w, int scale_dim, int inner_dim,
    bool is_scale_const, int const_scale, int right_shift_width,
    int do_activation, int activation_method, float activation_arg[],
    const int *i8_multiplier, // INT8_PER_LAYER
    bool do_bias,
    bool second_is_blob // true means second comes from weight, otherwise comes
                        // from another input
) {
#define RELU (0)
    bool fused_relu = (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f));
    LLVM_DEBUG(llvm::errs() << "fused_relu is " << fused_relu;);

  /*
     CviBackendContext::QuantizeMode qmode =
     static_cast<CviBackendContext::QuantizeMode>(getQuantizeMode(i8_multiplier));
  */
  // TODO: use `getQuantizeMode` to determin quantize mode
  CviBackendContext::QuantizeMode qmode;
  if (i8_multiplier != nullptr) {
    qmode = CviBackendContext::QuantizeMode::INT8_PER_LAYER;
  }
  else {
    qmode = CviBackendContext::QuantizeMode::INT8_32_MULTIPLER;
  }

  if (is_scale_const) {
    LLVM_DEBUG(llvm::errs() << llvm::format("quantization_fixed_forward_bmkernel:\n"
                                          " layer_id, input_gaddr 0x%lx, output_gaddr 0x%lx\n"
                                          "  in(%d, %d, %d, %d), quantization_num %d\n",
                                          layer_id, input_gaddr, output_gaddr, input_n, input_c,
                                          input_h, input_w));
    LLVM_DEBUG(llvm::errs() << "quantization i8_multiplier=" << const_scale
                          << ",right_shift_width=" << right_shift_width << "\n");
    int sec_n = input_n;
    int count = ALIGN(sec_n * input_c * input_h * input_w, MAX_W);
    int blob_num = 1;  // use 8bit output
    int slice_num = ctx.split(blob_num, count);
    int step = ALIGN(ceiling_func(count, slice_num), MAX_W);
    int offset_input = 0;
    LLVM_DEBUG(llvm::errs() << llvm::format("sec_n %d count %d blob_num %d slice_num %d step %d\n",
                                          sec_n, count, blob_num, slice_num, step));

    for (int pos = 0; pos < count; pos += step) {
      int slice = std::min(count - pos, step);
      LLVM_DEBUG(llvm::errs() << "slice=" << slice;);
      cvk_ml_shape_t shape_ = ctx.ml_shape_t1(slice, CVK_FMT_I8);

#if 0
     tl_tensor_lmem_output_l.start_address = tl_output_l->start_address;
     tl_tensor_lmem_output_l.fmt = tl_output_l->fmt;
     tl_tensor_lmem_output_l.shape = {tl_output_l->shape.n, tl_output_l->shape.c,
                                      static_cast<uint32_t>(1), tl_output_l->shape.w};
     tl_tensor_lmem_output_l.stride = {tl_output_l->stride.n, tl_output_l->stride.c,
                                       tl_output_l->stride.h, 1};
#endif
      // cvk_tl_t *ifmap = ctx.tl_alloc(shape_, CVK_FMT_I8, /*eu_align=*/1);
      cvk_tl_t ifmap;
      ifmap.start_address = 0;
      ifmap.fmt = CVK_FMT_I8;
      ifmap.shape = {shape_.n, shape_.c, 1, shape_.w};
      ifmap.stride = ctx.tl_default_stride(ifmap.shape, CVK_FMT_I8, /*eu_aligned=*/1);

      ctx.tdma_load(&ifmap, input_gaddr + offset_input);

      // FIXME: need to verify INT8_PER_LAYER value
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = &ifmap;
      p.a = &ifmap;
      p.b_is_const = 1;
      p.b_const.val = const_scale * i8_multiplier[0];
      p.b_const.is_signed = false;
      p.rshift_bits = right_shift_width;
      p.layer_id = layer_id;
      p.relu_enable = fused_relu;
      ctx.tiu_mul(&p);

      ctx.tdma_store(&ifmap, output_gaddr + offset_input);
      offset_input += slice * sizeof(uint8_t);
    }
    return;
  }


  LLVM_DEBUG(llvm::errs() << llvm::format(
      "input_gaddr 0x%lx scale_gaddr 0x%lx, "
      "bias_gaddr:0x%lx, output_gaddr:0x%lx, "
      "input_n %d input_c %d input_h %d input_w %d "
      "scale_dim %d inner_dim %d \n",
      input_gaddr, scale_gaddr, bias_gaddr, output_gaddr, input_n, input_c,
      input_h, input_w, scale_dim, inner_dim););

  assert(input_n * input_c * input_h * input_w == scale_dim * inner_dim);

  // input_c = scale_dim;
  if (inner_dim != input_h * input_w) {
    input_h = inner_dim;
    input_w = 1;
  }
  LLVM_DEBUG(llvm::errs() << "input_c is " << input_c;);

  /*
   * Calculate shape and stride for bottom
   */

  cvk_tg_shape_t ts_bottom_shape;
  ts_bottom_shape.n = input_n;
  ts_bottom_shape.c = input_c;
  ts_bottom_shape.h = input_h;
  ts_bottom_shape.w = input_w;

  cvk_tg_stride_t ts_bottom_stride;
  ts_bottom_stride = ctx.tg_default_stride(ts_bottom_shape, CVK_FMT_I8);

  /*
   * put scale to lmem
   */

  cvk_tl_shape_t tl_scale_shape;
  tl_scale_shape.n = 1;
  tl_scale_shape.c = input_c;
  tl_scale_shape.h = 1;
  tl_scale_shape.w = 1;
  cvk_tl_t *tl_scale = ctx.lmem_alloc_tensor(tl_scale_shape, CVK_FMT_I8, /*eu_align=*/1);
  ctx.tdma_load(tl_scale, scale_gaddr);


  if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = tl_scale;
    p.a = tl_scale;
    p.b_is_const = 1;
    p.b_const.val = i8_multiplier[0];
    p.b_const.is_signed = true;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  }


  /*
   * put bias to lmem
   */

  cvk_tl_t _tl_bias;
  cvk_tl_t *tl_bias = &_tl_bias;

  ctx.load_bias_multiplier(input_c, do_bias, bias_gaddr, qmode, &tl_bias);

  int coeff_usage = ctx.get_lmem_usage(1, input_c, 1, 1); // scale

  if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    coeff_usage += ctx.get_lmem_usage(1, input_c, 1, 1) * 2; // 2 for 16byte bias
  }
  else if (qmode == CviBackendContext::QuantizeMode::INT8_32_MULTIPLER) {
    int bias_len = ctx.chan_quan_param_size(do_bias); // hw config
    coeff_usage += ctx.get_lmem_usage(1, input_c, 1, bias_len); // for 32 bit multiplier
  }

  int nsecs = 1, hsecs = 1;
  ctx.split_nh(input_n, input_c, input_h, input_w, 1,
            coeff_usage, &nsecs,
            &hsecs);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "scale inference, <%d,%d,%d,%d>, nsecs:%d, hsecs:%d\n",
      input_n, input_c, input_h, input_w, nsecs, hsecs););

  int nslice = input_n / nsecs;
  int hslice = input_h / hsecs;
  int nresidual = input_n - nslice * nsecs;
  int hresidual = input_h - hslice * hsecs;

  for (int nidx = 0, nstart = 0; nidx < nsecs; nidx++) {
    int sec_len_n = nslice + (nidx < nresidual);

    for (int hidx = 0, hstart = 0; hidx < hsecs; hidx++) {
      int sec_len_h = hslice + (hidx < hresidual);
      uint64_t offset = (nstart * ts_bottom_stride.n + hstart * input_w) * sizeof(uint8_t);

      LLVM_DEBUG(llvm::errs() << llvm::format(
          "loop, nstart:%d,hstart:%d, sec_len_n:%d,sec_len_h:%d, offset:%lu, "
          "ts_bottom_stride.n:%u\n",
          nstart, hstart, sec_len_n, sec_len_h, offset, ts_bottom_stride.n););

      /*
       * put sliced bottom to lmem
       */

      cvk_tl_shape_t tl_bslice_shape;
      tl_bslice_shape.n = sec_len_n;
      tl_bslice_shape.c = input_c;
      tl_bslice_shape.h = sec_len_h;
      tl_bslice_shape.w = input_w;

      cvk_tl_t *tl_bslice = ctx.lmem_alloc_tensor(tl_bslice_shape, CVK_FMT_I8, 1);
      ctx.tdma_load_stride(tl_bslice, input_gaddr + offset, ts_bottom_stride);

      /*
       * Res(n, c, h, w) = A(n, c, h, w) * B(1,c,1,1) + Bias(1,c,1,1)
       * Use depthwise-conv to implement linear-arithmatic MAC
       * (channel-wise mac).
       */

      if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
        cvk_tiu_depthwise_pt_convolution_param_t param = {0};
        param.ofmap = tl_bslice;
        param.ifmap = tl_bslice;
        param.weight = tl_scale;
        param.bias = do_bias ? tl_bias : nullptr;
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
        param.rshift_bits = right_shift_width;
        param.relu_enable = fused_relu;
        param.layer_id = layer_id;
        param.ins_val = 0;                            // symmetric quantization
        param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
        ctx.tiu_pt_depthwise_convolution(&param);
      }
      else if (qmode == CviBackendContext::QuantizeMode::INT8_32_MULTIPLER) {
        tl_bias->shape = ctx.tl_shape_t4(1, input_c, 1, 1);
        tl_bias->stride = ctx.tl_default_stride(
            tl_bias->shape, CVK_FMT_I8, /*eu_aign=*/0);

        cvk_tiu_depthwise_convolution_param_t param = {nullptr};
        param.ofmap = tl_bslice;
        param.ifmap = tl_bslice;
        param.weight = tl_scale;
        param.chl_quan_param = tl_bias;
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
        param.has_bias = do_bias ? 1 : 0;
        param.relu_enable = fused_relu;
        param.layer_id = layer_id;
        param.ins_val = 0;                            // symmetric quantization
        param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
        ctx.tiu_depthwise_convolution(&param);
      }

      /*
       * Get sliced output back to gmem
       */
      ctx.tdma_store_stride(tl_bslice, output_gaddr + offset, ts_bottom_stride);

      hstart += sec_len_h;
      ctx.lmem_free_tensor(tl_bslice);
    }
    nstart += sec_len_n;
  }

  if (do_bias) {
    ctx.lmem_free_tensor(tl_bias);
  }
  ctx.lmem_free_tensor(tl_scale);
}

// wrapper for quantize for int 8, INT8_PER_LAYER
void cvi_backend_tg_fixed_scale_qi8_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t scale_gaddr, gaddr_t bias_gaddr, gaddr_t output_gaddr, int input_n,
    int input_c, int input_h, int input_w, int scale_dim, int inner_dim,
    bool is_scale_const, int const_scale, int right_shift_width,
    int do_activation, int activation_method, float activation_arg[],
    const int *i8_multiplier, // INT8_PER_LAYER
    bool do_bias,
    bool second_is_blob // true means second comes from weight, otherwise comes
                        // from another input
) {
  assert(i8_multiplier && "must give scalar");

  // For tdma
  ctx.set_layer_id(layer_id);

  cvi_backend_tg_fixed_scale_kernel(ctx,
      layer_id,
      input_gaddr,
      scale_gaddr,
      bias_gaddr,
      output_gaddr,
      input_n,
      input_c, input_h, input_w, scale_dim,
      inner_dim, is_scale_const, const_scale,
      right_shift_width,
      do_activation,
      activation_method,
      activation_arg,
      i8_multiplier, // INT8_PER_LAYER
      do_bias,
      second_is_blob // true means second comes from weight, otherwise comes from another input
      );
}

// wrapper for quantize for int 32, INT8_32_MULTIPLER
void cvi_backend_tg_fixed_scale_qi32_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t scale_gaddr, gaddr_t bias_gaddr, gaddr_t output_gaddr, int input_n,
    int input_c, int input_h, int input_w, int scale_dim, int inner_dim,
    bool is_scale_const, int const_scale, int do_activation,
    int activation_method, float activation_arg[], bool do_bias,
    bool second_is_blob // true means second comes from weight, otherwise comes
                        // from another input
) {
  // For tdma
  ctx.set_layer_id(layer_id);

  cvi_backend_tg_fixed_scale_kernel(ctx,
      layer_id,
      input_gaddr,
      scale_gaddr,
      bias_gaddr,
      output_gaddr,
      input_n,
      input_c, input_h, input_w, scale_dim,
      inner_dim, is_scale_const, const_scale,
      0,
      do_activation,
      activation_method,
      activation_arg,
      nullptr,
      do_bias,
      second_is_blob // true means second comes from weight, otherwise comes from another input
      );
}
