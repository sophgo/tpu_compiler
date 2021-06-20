/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReorgKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

void cvi_backend_tg_fixed_reorg_kernel(const CviBackendContext &ctx,
                                       uint32_t layer_id, gaddr_t input_gaddr,
                                       gaddr_t output_gaddr, int batch,
                                       int channel, int height, int width,
                                       int stride) {
  uint64_t output_c_stride = channel * (height / stride) * (width / stride) * sizeof(uint8_t);
  for (int n = 0; n < batch; n++) {
    for (int h = 0; h < stride; h++) {
      for (int w = 0; w < stride; w++) {
        cvi_backend_tg_fixed_avg_pooling_kernel(
            ctx, layer_id,
            input_gaddr + (h * width + w) * sizeof(uint8_t),
            output_gaddr + (h * stride + w) * output_c_stride,
            batch, channel, // n, c
            height, width, // h, w
            1, 1,
            0, 0, 0, 0,
            stride, stride,
            false, // do_relu
            0, 1, // rshift, multiplier
            true);
      }
    }
    input_gaddr += channel * height * width * sizeof(uint8_t);
    output_gaddr +=
        (channel*stride*stride) * (height / stride) * (width /stride) * sizeof(uint8_t);
  }
}

void cvi_backend_tg_bf16_reorg_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                               gaddr_t input_gaddr, gaddr_t output_gaddr,
                               int batch, int channel, int height, int width, int stride) {
  uint64_t output_c_stride = channel * (height / stride) * (width / stride) * sizeof(uint16_t);
  for (int n = 0; n < batch; n++) {
    for (int h = 0; h < stride; h++) {
      for (int w = 0; w < stride; w++) {
          cvi_backend_tg_bf16_pooling_kernel(
              ctx,
              layer_id,
              input_gaddr + (h * width + w) * sizeof(uint16_t),
              output_gaddr + (h * stride + w) * output_c_stride,
              uint32_t(-1),
              uint32_t(-1),
              batch, channel, height, width,
              1, 1,
              0, 0, 0, 0,
              stride, stride,
              1,
              0.0,
              0,
              1);
      }
    }
    input_gaddr += channel * height * width * sizeof(uint16_t);
    output_gaddr +=
        (channel*stride*stride) * (height / stride) * (width /stride) * sizeof(uint16_t);
  }
}