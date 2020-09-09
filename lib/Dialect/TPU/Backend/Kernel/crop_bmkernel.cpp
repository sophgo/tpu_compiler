/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: crop_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880_bmkernel_crop"
#define DEBUG_SPLIT "bmnet_bm1880_bmkernel_crop_split"

void crop_fixed_forward_bmkernel(const CviBackendContext &ctx, uint32_t stream_id,
                                 uint32_t inst_id, uint32_t layer_id,
                                 const uint32_t *depends, uint32_t depends_len,
                                 gaddr_t bottom_gaddr, gaddr_t top_gaddr, int *input1_dim,
                                 int *input2_dim, int *output_dim, int *offsets,
                                 cvi_backend_fmt_t fmt) {

  int data_size = (fmt == CVI_FMT_BF16) ? sizeof(uint16_t) : sizeof(uint8_t);
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

  cvk_tg_shape_t output_shape = {
      (uint32_t)output_n * data_size, (uint32_t)output_c * data_size,
      (uint32_t)output_h * data_size, (uint32_t)output_w * data_size};

  if (fmt == CVI_FMT_I8) {
    // crop the bottom to top from global to global
    tdma_g2g_tensor_copy(ctx, src_gaddr, output_shape, src_gstride, CVK_FMT_I8, top_gaddr,
                         output_shape, dst_gstride, CVK_FMT_I8);
  } else if (fmt == CVI_FMT_BF16) {
    tdma_g2g_tensor_copy(ctx, src_gaddr, output_shape, src_gstride, CVK_FMT_BF16,
                         top_gaddr, output_shape, dst_gstride, CVK_FMT_BF16);
  }
}
