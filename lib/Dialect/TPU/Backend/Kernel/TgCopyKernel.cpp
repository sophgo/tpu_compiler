#include "CviBackendContext.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "TgCopyKernel"

void cvi_backend_tg_copy_kernel(const CviBackendContext &ctx, gaddr_t ga_input,
                                gaddr_t ga_output,
                                const std::vector<int> &shape,
                                const std::vector<int> &i_stride,
                                const std::vector<int> &o_stride,
                                cvk_fmt_t fmt) {
  assert(shape.size() == 4);
  int fmt_size = ctx.bytesize_of_fmt(fmt);
  cvk_tg_shape_t output_shape =
      ctx.tg_shape_t4(shape[0], shape[1], shape[2], shape[3]);
  cvk_tg_shape_t input_shape = output_shape;
  cvk_tg_stride_t input_gstride;
  input_gstride.n = i_stride[0] * fmt_size;
  input_gstride.c = i_stride[1] * fmt_size;
  input_gstride.h = i_stride[2] * fmt_size;
  input_gstride.w = i_stride[3] * fmt_size;

  cvk_tg_stride_t output_gstride;
  output_gstride.n = o_stride[0] * fmt_size;
  output_gstride.c = o_stride[1] * fmt_size;
  output_gstride.h = o_stride[2] * fmt_size;
  output_gstride.w = o_stride[3] * fmt_size;

  ctx.tdma_g2g_tensor_copy(ga_input, input_shape, input_gstride, fmt, ga_output,
                           output_shape, output_gstride, fmt);
}