#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"

void cvi_backend_tg_tensor_copy_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    uint32_t n, uint32_t c, uint32_t h, uint32_t w,
    uint32_t align_bytes, cvk_fmt_t fmt) {

  uint32_t size = (fmt == CVK_FMT_BF16) ? 2 : 1;
  uint32_t aligned_width = align_up(w, align_bytes);

  cvk_tg_shape_t shape = {n, c, h, w};

  llvm::errs() << "shape: n=" << n << ", c=" << c << ", h=" << h << ", w=" << w <<"\n";

  cvk_tg_stride_t src_stride = {c * h * aligned_width * size,
                                 h * aligned_width * size,
                                 aligned_width * size,
                                 size};

  llvm::errs() << "src stride: (" << src_stride.n << ", " << src_stride.c << ", "
                                  << src_stride.h << ", " << src_stride.w << ")\n";

  cvk_tg_stride_t dst_stride = ctx.tg_default_stride(shape, fmt);

  llvm::errs() << "dst stride: (" << dst_stride.n << ", " << dst_stride.c << ", "
                                  << dst_stride.h << ", " << dst_stride.w << ")\n";

  ctx.tdma_g2g_tensor_copy(ga_ifmap, shape, src_stride, fmt, ga_ofmap, shape,
                             dst_stride, fmt);
}