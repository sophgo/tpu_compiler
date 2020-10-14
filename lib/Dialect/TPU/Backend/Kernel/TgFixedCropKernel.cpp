/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedCropKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "bm1880_kernel_crop"
#define DEBUG_SPLIT "bm1880_kernel_crop_split"

void cvi_backend_tg_fixed_crop_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                 uint32_t inst_id, uint32_t layer_id,
                                 const uint32_t *depends, uint32_t depends_len,
                                 gaddr_t bottom_gaddr, gaddr_t top_gaddr, int *input1_dim,
                                 int *input2_dim, int *output_dim, int *offsets,
                                 cvk_fmt_t fmt) {

  int data_size = ctx.bytesize_of_fmt(fmt);
  int offset_n = offsets[0];
  int offset_c = offsets[1];
  int offset_h = offsets[2];
  int offset_w = offsets[3];
  int input_c = input1_dim[1];
  int input_h = input1_dim[2];
  int input_w = input1_dim[3];
  int output_n = output_dim[0];
  int output_c = output_dim[1];
  int output_h = output_dim[2];
  int output_w = output_dim[3];

  gaddr_t src_gaddr =
      bottom_gaddr +
      (offset_n * input_c * input_h * input_w +
       offset_c * input_h * input_w +
       offset_h * input_w + offset_w) * data_size;

  uint32_t src_N_stride = static_cast<uint32_t>(input_c * input_h * input_w * data_size);
  uint32_t src_C_stride = static_cast<uint32_t>(input_h * input_w * data_size);
  uint32_t src_H_stride = static_cast<uint32_t>(input_w * data_size);
  cvk_tg_stride_t src_gstride = {src_N_stride, src_C_stride, src_H_stride};

  uint32_t dst_N_stride =
      static_cast<uint32_t>(output_c * output_h * output_w * data_size);
  uint32_t dst_C_stride = static_cast<uint32_t>(output_h * output_w * data_size);
  uint32_t dst_H_stride = static_cast<uint32_t>(output_w * data_size);
  cvk_tg_stride_t dst_gstride = {dst_N_stride, dst_C_stride, dst_H_stride};

  // TODO: support tiling w
  // we load origin shape and overwrite new crop's shape
  int require_shape = 0;
  int coeff_lane_shape = 0;
  int blob_num = 1; // 1 means only one blob and it chould overwrite itself
  cvk_tg_shape_t tg_shape;
  tg_shape.n = 1; // TODO: support n dim shift
  tg_shape.c = output_c;
  tg_shape.h = input_h;
  tg_shape.w = input_w;
  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;
  dst_gstride = {dst_N_stride, dst_C_stride, dst_H_stride, (uint32_t)data_size};

  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt, &tiling_info,
      ctx.TilingDimNH, &tg_shape);

  // accumuate store
  for (int batch = 0; batch < output_n; batch++) {
    int residual_h = output_h;
    int top_local_shift = 0;
    int bottom_local_shift = 0;
    for (size_t i = 0; i < tiling_info.size(); i++) {
      int n = tiling_info[i].first.n;
      int c = tiling_info[i].first.c;
      int h = tiling_info[i].first.h;
      int w = tiling_info[i].first.w;
      //gaddr_t gaddr_offset = tiling_info[i].second;

      residual_h -= h;
      int _output_h = residual_h >= 0 ? h : residual_h + h;
      bool is_last = residual_h <= 0;
      if (is_last) {
        h = _output_h;
      }

      assert(w == input_w && "not support tiling w");

      // load
      cvk_tl_shape_t input_shape = ctx.tl_shape_t4(n, c, h, w);
      cvk_tl_t *bottom = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);
      src_gstride.h = bottom->stride.h;
      ctx.tdma_load_stride_bf16(bottom,
          src_gaddr + bottom_local_shift + batch * src_N_stride, src_gstride);
      bottom_local_shift += bottom->stride.c;

      cvk_tl_t top = *bottom; // leverage structure
      cvk_tl_shape_t output_shape = ctx.tl_shape_t4(n, c, _output_h, output_w);
      top.shape = output_shape;
      top.stride = ctx.tl_default_stride(top.shape, fmt, /*eu_align=*/0);
      bottom->shape = output_shape; // shape should be same for tiu_copy constrain

      // crop in lmem, move it contiguous
      cvk_tiu_copy_param_t param = {0};
      param.src = bottom;
      param.dst = &top;
      param.layer_id = layer_id;
      ctx.tiu_copy(&param);

      // store back
      dst_gstride.h = top.stride.h;
      ctx.tdma_store_stride_bf16(&top,
          top_gaddr + top_local_shift + batch * dst_N_stride, dst_gstride,
          /*do_transpose=*/0);

      // shift lmem
      top_local_shift += top.stride.c;

      // release
      bottom->shape = input_shape;
      ctx.lmem_free_tensor(bottom);

      if (is_last) {
        // all needs moved, break it
        break;
      }
    }
  }
}
