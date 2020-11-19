/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReverseKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "cvi_backend_reverse_kernel"

void cvi_backend_tg_reverse_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int axis, cvk_fmt_t fmt) {
  LLVM_DEBUG(llvm::dbgs() << "cvi_backend_tg_reverse_kernel\n"
                          << "  layer_id " << layer_id << "\n";);
  assert(axis == 0 && "only support axis = 0 now");
  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, 1, c, h, w, fmt, 1, 0,
                     CviBackendContext::TilingAll);
  uint64_t n_stride = c * h * w * ctx.bytesize_of_fmt(fmt);
  for (int in = 0; in < n; in++) {
    for (auto &tile : tiles) {
      cvk_tl_t *tl_input =
          ctx.lmem_alloc_tensor(ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w), fmt, 1);
      gaddr_t ga_ifmap = ga_input + in * n_stride + tile.offset;
      gaddr_t ga_ofmap = ga_output + (n - 1 - in) * n_stride + tile.offset;
      ctx.tdma_load(tl_input, ga_ifmap);
      ctx.tdma_store(tl_input, ga_ofmap);
      ctx.lmem_free_tensor(tl_input);
    }
  }
}
