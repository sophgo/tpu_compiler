#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "TgStrideCopyKernel"

void cvi_backend_tg_stride_copy_kernel(const CviBackendContext &ctx,
                                       gaddr_t ga_input, gaddr_t ga_output,
                                       const std::vector<int64_t> &shape,
                                       const std::vector<int32_t> &i_stride,
                                       const std::vector<int32_t> &o_stride,
                                       cvk_fmt_t fmt) {
  assert(shape.size() == 4);
  cvk_tg_shape_t output_shape = ctx.tg_shape_t4(shape[0], shape[1], shape[2], shape[3]);
  cvk_tg_shape_t input_shape = output_shape;
  cvk_tg_stride_t input_gstride;
  input_gstride.n = i_stride[0];
  input_gstride.c = i_stride[1];
  input_gstride.h = i_stride[2];
  input_gstride.w = i_stride[3];

  cvk_tg_stride_t output_gstride;
  output_gstride.n = o_stride[0];
  output_gstride.c = o_stride[1];
  output_gstride.h = o_stride[2];
  output_gstride.w = o_stride[3];

  ctx.tdma_g2g_tensor_copy(ga_input, input_shape, input_gstride, fmt,
                                   ga_output, output_shape, output_gstride,
                                   fmt);
}