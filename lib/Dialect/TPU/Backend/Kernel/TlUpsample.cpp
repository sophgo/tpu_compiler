/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_upsampling.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_upsample"

void cvi_backend_tl_upsample(
        const CviBackendContext &ctx, uint32_t layer_id,
        laddr_t input_laddr, laddr_t output_laddr,
        int input_n, int input_c, int input_h, int input_w,
        int scale_h, int scale_w) {
  // input
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_laddr;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_input_shape;
  tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_I8, 1);

  // output
  auto output_n = input_n;
  auto output_c = input_c;
  auto output_h = input_h * scale_h;
  auto output_w = input_w * scale_w;

  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(output_n, output_c,output_h, output_w);
  cvk_tl_t tl_output;
  tl_output.start_address = output_laddr;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_output_shape;
  tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_I8, 1);

  cvk_tiu_average_pooling_param_t param = {0};
  param.ofmap = &tl_output;
  param.ifmap = &tl_input;
  param.kh = scale_h;
  param.kw = scale_w;
  param.ins_h = scale_h - 1;
  param.ins_last_h = 0;
  param.ins_w = scale_w - 1;
  param.ins_last_w = 0;
  param.pad_top = scale_h - 1;
  param.pad_bottom = scale_h - 1;
  param.pad_left = scale_w - 1;
  param.pad_right = scale_w - 1;
  param.stride_h = 1;
  param.stride_w = 1;
  param.avg_pooling_const = 1.0;
  param.rshift_bits = 0;
  param.layer_id = layer_id;
  param.ins_val = param.avg_pooling_const;
  param.ins_fp = ctx.convert_fp32_to_bf16(param.avg_pooling_const);
  ctx.tiu_average_pooling(&param);
}

