/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_TgFixedScaleKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bm1880v2_scale"

#define ASSERT(x) assert(x)
#define MAX_W (1 << 11)
// #define LOCAL_MEM_ADDRWIDTH (ctx.hw.local_mem_shift)
// #define LOCAL_MEM_SIZE (1 << LOCAL_MEM_ADDRWIDTH)

void cvi_backend_tg_bf16_scale_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                               uint32_t inst_id, uint32_t layer_id,
                               const uint32_t *depends, uint32_t depends_len,
                               gaddr_t input_gaddr, gaddr_t scale_gaddr, gaddr_t bias_gaddr, gaddr_t output_gaddr,
                               int input_n, int input_c, int input_h, int input_w,
                               int scale_dim, int inner_dim,
                               bool is_scale_const, int const_scale,
                               int do_activation,
                               int activation_method,
                               float activation_arg[],
                               bool do_bias,
                               bool second_is_load_weight // true means second comes from weight, otherwise comes from another input
) {
#define RELU (0)
    bool fused_relu = (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f));
    LLVM_DEBUG(llvm::errs() << "fused_relu is " << fused_relu;);

  if (is_scale_const) {
    assert(0 && "TODO: Scale Const");
  }
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "cvi_backend_tg_bf16_scale_kernel:\n"
      "    layer_id %d\n"
      "    bottom = %lx, scale_gaddr = 0x%lx, bias_gaddr = 0x%lx, top = %lx\n"
      "    nchw = (%d, %d, %d, %d)\n"
      "    do_bias = %d"
      "    scale_dim = %d, inner_dim = %d \n",
      layer_id, input_gaddr, scale_gaddr, bias_gaddr, output_gaddr,
      input_n, input_c, input_h, input_w, do_bias, scale_dim, inner_dim););

  assert(input_n * input_c * input_h * input_w == scale_dim * inner_dim);
  //input_c = scale_dim;
  if (inner_dim != input_h * input_w) {
    input_h = inner_dim;
    input_w = 1;
  }

  /*
   * Calculate shape and stride for bottom
   */

  cvk_tg_shape_t ts_bottom_shape;
  ts_bottom_shape.n = input_n;
  ts_bottom_shape.c = input_c;
  ts_bottom_shape.h = input_h;
  ts_bottom_shape.w = input_w;

  cvk_tg_stride_t ts_bottom_stride;
  ts_bottom_stride = ctx.tg_default_stride(ts_bottom_shape, CVK_FMT_BF16);

  /*
   * put scale to lmem
   */

  cvk_tl_shape_t tl_scale_shape;
  tl_scale_shape.n = 1;
  tl_scale_shape.c = input_c;
  tl_scale_shape.h = 1;
  tl_scale_shape.w = 1;
  cvk_tl_t *tl_scale =
      ctx.lmem_alloc_tensor(tl_scale_shape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_shape_t tl_bias_shape;
  cvk_tl_t *tl_bias;
  if (do_bias) {
    tl_bias_shape.n = 2;
    tl_bias_shape.c = input_c;
    tl_bias_shape.h = 1;
    tl_bias_shape.w = 1;

    tl_bias = ctx.lmem_alloc_tensor(tl_bias_shape, CVK_FMT_BF16, /*eu_align=*/1);

  }
  ctx.tdma_load_bf16(tl_scale, scale_gaddr);

  /*
   * put bias to lmem
   */

  if (do_bias) {
    ctx.tdma_load_bf16(tl_bias, bias_gaddr);
  }
  int coeff_usage = ctx.get_lmem_usage(
      1, do_bias ? input_c * 2 * sizeof(uint16_t) : input_c * sizeof(uint16_t), 1, 1); // scale
  int nsecs = 1, hsecs = 1;
  ctx.split_nh(input_n, input_c, input_h, input_w * sizeof(uint16_t), 2, coeff_usage,
            &nsecs, &hsecs);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      ("scale inference, <%d,%d,%d,%d>, nsecs:%d, hsecs:%d\n"), input_n,
      input_c, input_h, input_w, nsecs, hsecs););

  int nslice = input_n / nsecs;
  int hslice = input_h / hsecs;
  int nresidual = input_n - nslice * nsecs;
  int hresidual = input_h - hslice * hsecs;

  for (int nidx = 0, nstart = 0; nidx < nsecs; nidx++) {
    int sec_len_n = nslice + (nidx < nresidual);

    for (int hidx = 0, hstart = 0; hidx < hsecs; hidx++) {
      int sec_len_h = hslice + (hidx < hresidual);
      uint64_t offset = nstart * ts_bottom_stride.n + (hstart * input_w) * sizeof(uint16_t);

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

      cvk_tl_t *tl_bslice =
          ctx.lmem_alloc_tensor(tl_bslice_shape, CVK_FMT_BF16, /*eu_align=*/1);

      ctx.tdma_load_stride_bf16(tl_bslice, input_gaddr + offset, ts_bottom_stride);

      /*
       * Res(n, c, h, w) = A(n, c, h, w) * B(1,c,1,1) + Bias(1,c,1,1)
       * Use depthwise-conv to implement linear-arithmatic MAC
       * (channel-wise mac).
       */
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
      param.relu_enable = fused_relu;
      param.layer_id = layer_id;
      ctx.tiu_pt_depthwise_convolution(&param);


      /*
       * Get sliced output back to gmem
       */
      ctx.tdma_store_stride_bf16(tl_bslice, output_gaddr + offset, ts_bottom_stride);

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

