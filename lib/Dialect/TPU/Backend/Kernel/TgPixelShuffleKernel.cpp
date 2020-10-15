/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgPixelShuffleKernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"

#define DEBUG_TYPE "bm1880v2_pixel_shuffle"

static void pixelShuffle_split(
    const CviBackendContext &ctx, uint32_t input_n, uint32_t input_c,
    uint32_t input_h, uint32_t input_w, uint32_t output_c, uint32_t output_h,
    uint32_t output_w, uint32_t factor, cvk_fmt_t fmt, int eu_align, uint32_t &h_step) {
  h_step = input_h;
  for (; h_step > 0; --h_step) {
    cvk_tl_shape_t tiled_ifmap_shape = {factor, factor * NPU_NUM, h_step, input_w};
    uint32_t tiled_ifmap_size =
        ctx.lmem_tensor_to_size(tiled_ifmap_shape, fmt, eu_align);

    cvk_tl_shape_t tiled_ofmap_shape =  {factor, factor * NPU_NUM, h_step, input_w};
    uint32_t tiled_ofmap_size =
        ctx.lmem_tensor_to_size(tiled_ofmap_shape, fmt, eu_align);
    uint32_t total_size = tiled_ifmap_size + tiled_ofmap_size;
    if (total_size <= static_cast<uint32_t>(LOCAL_MEM_SIZE))
      break;
  }

  LLVM_DEBUG(llvm::dbgs() << "  upsample_split:  h_step " << h_step << "\n");

  assert(h_step && "Expect valid upsample tiling");
}

static void pixelShuffle_assign_lmem_layout(
    const CviBackendContext &ctx, uint32_t tiling_c, uint32_t tiling_h, uint32_t input_c,
    uint32_t input_h, uint32_t input_w, uint32_t output_c, uint32_t output_h,
    uint32_t output_w, uint32_t factor, cvk_fmt_t fmt, int eu_align, cvk_tl_t &tl_ifmap, cvk_tl_t &tl_ofmap) {

  uint32_t tl_offset = 0; // begin of local memory
  ctx.lmem_init_tensor(&tl_ifmap, {factor, factor * NPU_NUM, tiling_h, input_w}, fmt, eu_align);
  tl_ifmap.start_address = tl_offset;
  tl_offset += ctx.lmem_tensor_to_size(tl_ifmap.shape, tl_ifmap.fmt,
                                       tl_ifmap.eu_align);

  ctx.lmem_init_tensor(&tl_ofmap, {factor, factor * NPU_NUM, tiling_h, input_w}, fmt,
                       eu_align);
  tl_ofmap.start_address = tl_offset;
}

static void pixelShuffle_tensor_load_nc_transpose(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    cvk_tg_stride_t &ifmap_gstride, int n_pos, int c_pos, int h_pos, cvk_fmt_t fmt, cvk_tl_shape_t tileShape) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos + ifmap_gstride.c * c_pos
                                                           + ifmap_gstride.h * h_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {
      tileShape.n, tileShape.c, tileShape.h,
      tileShape.w};
  tg_ifmap.stride =  ifmap_gstride;
  tg_ifmap.stride.n = tg_ifmap.stride.c * tileShape.c;

  cvk_tl_t tl_dst;
  tl_dst.start_address = 0;  // Same as ifmap = 0
  tl_dst.fmt = fmt;
  tl_dst.shape = {tileShape.c, tileShape.n, tileShape.h, tileShape.w};
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/0);

  cvk_tdma_g2l_tensor_copy_nc_transposed_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = &tl_dst;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  pixelShuffle_tensor_load_nc_transpose\n"
      << "    tg offset " << ga_ifmap_offset
      << ", shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h << ")\n"
      << "    tl shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h
      << ", " << param.dst->stride.w << ")\n");

  ctx.tdma_g2l_tensor_copy_nc_transposed(&param);
}

static void pixelShuffle_tiu_copy(
    const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tl_ifmap, cvk_tl_t *tl_ofmap, int factor) {

  cvk_tl_t tl_dst;
  tl_dst.start_address = tl_ofmap->start_address;  // start of lmem
  tl_dst.fmt = tl_ofmap->fmt;
  tl_dst.shape = tl_ofmap->shape;
  int bytesize = tl_ofmap->stride.w;
  tl_dst.stride = {
      (uint32_t)(factor * tl_ofmap->shape.w * bytesize),
      (uint32_t)bytesize,
      (uint32_t)(factor * factor * tl_ofmap->shape.w * bytesize),
      (uint32_t)(factor * bytesize)
  };

  cvk_tiu_copy_param_t p2 = {0};
  p2.src = tl_ifmap;
  p2.dst = &tl_dst;
  p2.layer_id = layer_id;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "    [%d] L2L Reshape:\n"
                 "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                 "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                 p2.src->start_address, p2.src->shape.n,
                 p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                 p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                 p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                 p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
  ctx.tiu_copy(&p2);
}

static void pixelShuffle_tensor_store(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ofmap,
    cvk_tl_t *tl_ofmap, cvk_tg_stride_t &ofmap_gstride, int n_pos, int oc_pos, int h_pos, cvk_tl_shape_t tileShape) {
  uint64_t ga_ofmap_offset = ofmap_gstride.n * n_pos + ofmap_gstride.c * oc_pos
                                                            + ofmap_gstride.h * h_pos;

  cvk_tg_t tg_ofmap = {0};
  tg_ofmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  tg_ofmap.fmt = tl_ofmap->fmt;
  tg_ofmap.start_address = ga_ofmap + ga_ofmap_offset;
  tg_ofmap.shape = {
      tileShape.n, tileShape.c, tileShape.h,
      tileShape.w};
  tg_ofmap.stride = ofmap_gstride;

  cvk_tl_t tl_dst;
  tl_dst.start_address = tl_ofmap->start_address;  // start of lmem
  tl_dst.fmt = tl_ofmap->fmt;
  tl_dst.shape = tileShape;
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, tl_ofmap->fmt, /*eu_align=*/0);

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = &tl_dst;
  param.dst = &tg_ofmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  pixelShuffle_tensor_store\n"
      << "    tl shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << ", " << param.src->shape.w
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h
      << ", " << param.src->stride.w << ")\n"
      << "    tg offset " << ga_ofmap_offset
      << ", shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h << ")\n");

  ctx.tdma_l2g_tensor_copy(&param);
}

static void _pixel_shuffle_fixed_kernel_new(const CviBackendContext &ctx, uint32_t layer_id,
                                     gaddr_t ga_ifmap, gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c,
                                     uint32_t input_h, uint32_t input_w, uint32_t factor, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "pixel_shuffle_fixed_bmkernel:\n"
                 "  ga_ifmap 0x%lx, ga_ofmap 0x%lx, shape(%d, %d, %d, %d)\n"
                 "  h_factor %d, w_factor %d\n",
                 ga_ifmap, ga_ofmap, input_n, input_c, input_h, input_w, factor, factor));

  uint32_t in = input_n;
  uint32_t ic = input_c;
  uint32_t ih = input_h;
  uint32_t iw = input_w;
  uint32_t oc = input_c / (factor * factor);
  uint32_t oh = input_h * factor;
  uint32_t ow = input_w * factor;

  int eu_align = 0; // no need to align eu

  uint32_t n_step = 1;
  uint32_t oc_step = (oc >= (uint32_t)NPU_NUM) ? NPU_NUM : oc;
  uint32_t c_step = oc_step * factor * factor;
  uint32_t h_step = 0;
  pixelShuffle_split(
    ctx, in, ic, ih, iw,
    oc,  oh, ow,
    factor, fmt, eu_align, h_step);
  if (!h_step)
    return;

  // TODO: support other fmt
  // 2 means bf16 takes 2 bytes
  int unit_sz = fmt == CVK_FMT_BF16 ? 2 : 1;
  cvk_tg_stride_t ifmap_gstride = {
      input_c * input_h * input_w * unit_sz,
      input_h * input_w * unit_sz,
      input_w * unit_sz};
  cvk_tg_stride_t ofmap_gstride = {
    oc * oh * ow * unit_sz,
    oh * ow * unit_sz,
    ow * unit_sz};

  for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step) {
    for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
      uint32_t tiling_c = std::min(input_c - c_pos, c_step);
      for (uint32_t h_pos = 0; h_pos < input_h; h_pos += h_step) {
        uint32_t tiling_h = std::min(input_h - h_pos, h_step);
        // 1. Assign local memory layout
        cvk_tl_t tl_ifmap, tl_ofmap;
        pixelShuffle_assign_lmem_layout(ctx, tiling_c, tiling_h, ic, ih, iw, oc,
                                    oh, ow, factor, fmt, eu_align, tl_ifmap, tl_ofmap);
        // 2. tensor load
        cvk_tl_shape_t tileShape = {oc_step, factor * factor, tiling_h, iw};
        pixelShuffle_tensor_load_nc_transpose(ctx, layer_id, ga_ifmap, ifmap_gstride, n_pos, c_pos, h_pos,
                            fmt, tileShape);

        // 3. tiu copy
        pixelShuffle_tiu_copy(ctx, layer_id, &tl_ifmap, &tl_ofmap, factor);

        // 4. tensor store
        uint32_t oc_pos = c_pos / (factor * factor);
        uint32_t oh_pos = h_pos * factor;
        cvk_tl_shape_t outputTileShape = {1, tiling_c / (factor * factor), tiling_h * factor, ow};
        pixelShuffle_tensor_store(ctx, layer_id, ga_ofmap, &tl_ofmap, ofmap_gstride,
                              n_pos, oc_pos, oh_pos, outputTileShape);
      }
    } //for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
  } // for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step)
}

void cvi_backend_tg_fixed_pixel_shuffle_kernel(const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
                             uint32_t layer_id, const uint32_t *depends, uint32_t depends_len, gaddr_t ga_ifmap,
                             gaddr_t ga_ofmap, int input_n, int input_c, int input_h, int input_w, int factor) {

  // For tdma
  ctx.set_layer_id(layer_id);

  _pixel_shuffle_fixed_kernel_new(
      ctx, layer_id, ga_ifmap, ga_ofmap,
      (uint32_t)input_n, (uint32_t)input_c,
      (uint32_t)input_h, (uint32_t)input_w,
      (uint32_t)factor, CVK_FMT_I8);
}

void cvi_backend_tg_bf16_pixel_shuffle_kernel(const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
                                  uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
                                  gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c,
                                  int input_h, int input_w, int factor) {
  // For tdma
  ctx.set_layer_id(layer_id);

  _pixel_shuffle_fixed_kernel_new(
      ctx, layer_id, ga_ifmap, ga_ofmap,
      (uint32_t)input_n, (uint32_t)input_c,
      (uint32_t)input_h, (uint32_t)input_w,
      (uint32_t)factor, CVK_FMT_BF16);
}
