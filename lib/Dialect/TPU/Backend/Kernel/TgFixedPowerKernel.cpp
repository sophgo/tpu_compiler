/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedPowerKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "bm1880v2_power"

static void one_step(
        const CviBackendContext &ctx, uint32_t layer_id,
        gaddr_t input_gaddr, gaddr_t output_gaddr,
        cvk_tl_t *tl_scale,
        cvk_tl_t *tl_shift,
        int input_n, int input_c, int input_h, int input_w,
        int power, cvk_fmt_t fmt, int right_shift_width,
        gaddr_t mulpy_offset) {


  cvk_tl_t *tl_ifmap;
  bool use_mulpy = mulpy_offset != uint32_t(-1);

  if (fmt == CVK_FMT_I8) {
    if (power == 0) {
      cvk_tdma_l2g_tensor_fill_constant_param_t p = {0};
      cvk_tg_t dst;

      dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(output_gaddr);
      dst.start_address = output_gaddr;
      dst.fmt = fmt;
      dst.shape = ctx.tg_shape_t4(input_n,input_c,input_h,input_w);
      dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

      p.constant = 1;
      p.dst = &dst;
      ctx.tdma_l2g_tensor_fill_constant(&p);
    }
    else {
      tl_ifmap = ctx.lmem_alloc_tensor(ctx.tl_shape_t4(input_n, input_c, input_h, input_w), fmt, /*eu_align=*/1);
      ctx.tdma_load(tl_ifmap, input_gaddr);
      if (use_mulpy) {
        // FIXME: try to use qdm
        assert(0);
#if 0
        cvk_tl_t tl_chl_quan_tiu = {0};
        tl_chl_quan_tiu.start_address = tl_chl_quan_param[coeff_flip]->start_address;
        tl_chl_quan_tiu.fmt = tl_chl_quan_param[coeff_flip]->fmt;
        tl_chl_quan_tiu.shape = {1, tl_chl_quan_param[coeff_flip]->shape.c, 1, 1};
        tl_chl_quan_tiu.stride =
          ctx.tl_default_stride(tl_chl_quan_tiu.shape, /*eu_align=*/0);

        cvk_tiu_depthwise_convolution_param_t param = {nullptr};
        param.ofmap = tl_ifmap;
        param.ifmap = tl_ifmap;
        param.weight = tl_weight[coeff_flip];
        param.chl_quan_param = &tl_chl_quan_tiu;
        param.ins_h = param.ins_last_h = 0;
        param.ins_w = param.ins_last_w = 0;
        param.pad_top = 0;
        param.pad_bottom = 0;
        param.pad_left = 0;
        param.pad_right = 0;
        param.stride_h = 1;
        param.stride_w = 1;
        param.dilation_h = 1;
        param.dilation_w = 1;
        param.has_bias = 1;
        param.relu_enable = 0;
        param.layer_id = layer_id;

        ctx.tiu_depthwise_convolution(&param);
#endif
      }
      else {
        cvk_tiu_depthwise_pt_convolution_param_t p5 = {0};
        for (int i = 0; i < power; i++) {
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
          p5.ofmap = tl_ifmap;
          p5.ifmap = tl_ifmap;
          p5.weight = tl_scale;
          p5.bias = tl_shift;
          p5.relu_enable = 0;
          p5.layer_id = layer_id;
          p5.rshift_bits = right_shift_width;
          p5.ins_val = 0;                            // symmetric quantization
          p5.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
          ctx.tiu_pt_depthwise_convolution(&p5);
        }
      }
      ctx.tdma_store(tl_ifmap, output_gaddr);
    }

    ctx.lmem_free_tensor(tl_ifmap);
  }
  else {
    assert(0);
  }
}

void cvi_backend_tg_fixed_power_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w,
    const int power, const gaddr_t scale_gaddr, const gaddr_t shift_gaddr,
    int right_shift_width, gaddr_t mulpy_offset, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  ">cvi_backend_tg_fixed_power_kernel:\n"
                  "    layer_id %d\n"
                  "    input gaddr 0x%lx, shape (%d, %d, %d, %d)\n"
                  "    output gaddr 0x%lx, scale_gaddr 0x%lx, shift_gaddr 0x%lx\n"
                  "    power %d\n",
                  layer_id, input_gaddr, input_n, input_c, input_h, input_w,
                  output_gaddr, scale_gaddr, shift_gaddr,
                  power));


  assert(power >= 0);
  // FIXME: support int8
  assert(fmt == CVK_FMT_I8);

  // tiling input
  int blob_num = 1;
  int require_shape = input_n * input_c * input_h * input_w;

  //<! tilling MUST align `npu_num`
  int coeff_lane_shape = 3; // 3 means scale / shift(2bytes)

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  //uintptr_t tiling_info_ptr = reinterpret_cast<std::uintptr_t>(&tiling_info);
  //ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt, tiling_info_ptr);
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, (cvk_fmt_t)fmt, &tiling_info);

  // eu_align set to 1 imply align with c
  cvk_tl_t *tl_scale = ctx.lmem_alloc_tensor(ctx.tl_shape_t4(1,
        NPU_NUM, 1, 1), (cvk_fmt_t)fmt, /*eu_align=*/1);
  // 2 means 16bits
  cvk_tl_t *tl_shift = ctx.lmem_alloc_tensor(ctx.tl_shape_t4(2,
        NPU_NUM, 1, 1), (cvk_fmt_t)fmt, /*eu_align=*/0);
  tl_shift->stride =
    ctx.tl_default_stride(tl_shift->shape, CVK_FMT_I8, /*eu_aign=*/0);
  cvk_tg_stride_t bias_gstride = {static_cast<uint32_t>(NPU_NUM), 1, 1};

  ctx.tdma_load(tl_scale, scale_gaddr);
  ctx.tdma_load_stride(tl_shift, shift_gaddr, bias_gstride);

  for (size_t i = 0; i < tiling_info.size(); i++) {
      int n = tiling_info[i].first.n;
      int c = tiling_info[i].first.c;
      int h = tiling_info[i].first.h;
      int w = tiling_info[i].first.w;
      // its fine to reshape it
      tl_scale->shape.c = c;
      tl_scale->stride =
            ctx.tl_default_stride(tl_scale->shape, CVK_FMT_I8, /*eu_aign=*/1);

      // bias stride MUST pack, its safe to re-assign stride
      tl_shift->shape.c = c;
      tl_shift->stride =
            ctx.tl_default_stride(tl_shift->shape, CVK_FMT_I8, /*eu_aign=*/0);

      gaddr_t gaddr_offset = tiling_info[i].second;
      one_step(ctx, layer_id,
          input_gaddr + gaddr_offset, output_gaddr + gaddr_offset,
          tl_scale, tl_shift,
          n, c, h, w,
          power, CVK_FMT_I8, right_shift_width, mulpy_offset);
  }

  ctx.lmem_free_tensor(tl_shift);
  ctx.lmem_free_tensor(tl_scale);
}
