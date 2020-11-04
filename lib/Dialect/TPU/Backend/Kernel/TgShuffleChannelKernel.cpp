/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgShuffleChannelKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "cvi_backend_shuffle_channel_kernel"

void cvi_backend_tg_shuffle_channel_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id,
                                           gaddr_t input_gaddr,
                                           gaddr_t output_gaddr, int n, int c,
                                           int h, int w, int group,
                                           cvk_fmt_t fmt) {
  LLVM_DEBUG(llvm::dbgs() << "cvi_backend_tg_shuffle_channel_kernel\n"
                          << "  layer_id " << layer_id << "\n";);
  for (int batch = 0; batch < n; batch++) {
    uint64_t offset = batch * c * h * w * ctx.bytesize_of_fmt(fmt);
    cvi_backend_tg_permute_kernel(ctx, layer_id, input_gaddr + offset,
                                  output_gaddr + offset, group, c / group, h, w,
                                  1, 0, 2, 3, fmt);
  }
}
