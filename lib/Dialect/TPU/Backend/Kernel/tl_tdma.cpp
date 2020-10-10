/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_tdma.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_tdma"

void cvi_backend_tl_to_tensor(
    const CviBackendContext &ctx,
    cvk_tl_t *tensor,
    laddr_t la,
    uint32_t tensor_n, uint32_t tensor_c, uint32_t tensor_h, uint32_t tensor_w,
    cvk_fmt_t fmt, uint8_t eu_align) {

  tensor->start_address = la;
  tensor->fmt = fmt;
  tensor->shape = ctx.shape_t4(tensor_n, tensor_c, tensor_h, tensor_w);

  if (fmt == CVK_FMT_I8 || fmt == CVK_FMT_BF16) {
    tensor->stride = ctx.tl_default_stride(tensor->shape, fmt, eu_align);
  }
  else {
    assert(0 && "not support");
  }
}

void cvi_backend_tl_load_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to) {

  cvi_backend_tl_load_stride(ctx,
                             layer_id,
                             ga_src,
                             la_dst,
                             Local_N,
                             Local_C,
                             Local_H,
                             Local_W,
                             Global_C,
                             Global_H,
                             Global_W,
                             DoTranspose,
                             DoAligned,
                             isNeuron,
                             from,
                             to,
                             false  // DoDecompress
                             );

}


void cvi_backend_tl_load_compressed(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to,
    int h_step, int step_size) {

  // Global shape is used for stride - global memory layout
  assert(from == to && "Expect same data type");

  int eu_align = DoAligned ? 1 : 0;

  cvk_tl_stride_t tl_stride =
    ctx.tl_default_stride(ctx.shape_t4(Local_N, Local_C, Local_H, Local_W),
                          to, eu_align);

  for (int i = 0; i < Local_H; i+=h_step) {
    int cur_h = std::min(h_step, Local_H - i);

    cvk_cmpr_tg_t tg_cmpr_src = {0};
    ctx.gmem_init_tensor(&tg_cmpr_src.t,
                         {(uint32_t)Local_N, (uint32_t)Global_C, (uint32_t)cur_h,
                          (uint32_t)Global_W},
                         from);
    tg_cmpr_src.t.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_src);
    tg_cmpr_src.t.start_address = ga_src + step_size * (i / h_step);

    // HxW in each lane is contiguous
    cvk_tl_t tl_dst;
    ctx.lmem_init_tensor(&tl_dst, ctx.shape_t4(Local_N, Local_C, cur_h, Local_W),
                         to, eu_align);
    tl_dst.stride = tl_stride;
    tl_dst.start_address = la_dst + i * Local_W * tl_stride.w;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &tg_cmpr_src;
    param.dst = &tl_dst;
    param.layer_id = layer_id;
    ctx.tdma_g2l_tensor_copy_decompressed(&param);
  }
}

void cvi_backend_tl_load_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_src, laddr_t la_dst,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to,
    bool DoDecompress) {
  LLVM_DEBUG(
    llvm::errs() << llvm::format("cvi_backend_tl_load_stride:\n"
                                  "    layer_id %d\n"
                                  "    src (, %d, %d, %d), dst (%d, %d, %d, %d)\n"
                                  "    src 0x%lx, dst 0x%lx\n"
                                  "    DoTranspose %d, DoAligned %d, isNeuron %d\n"
                                  "    from %d to %d, Compressed %d\n",
                                  layer_id, Global_C, Global_H, Global_W, Local_N, Local_C,
                                  Local_H, Local_W, ga_src, la_dst, DoTranspose, DoAligned,
                                  isNeuron, from, to, DoDecompress));
  // tensor in local memory
  cvk_tl_shape_t tl_shape;
  tl_shape.n = DoTranspose ? Local_C : Local_N;
  tl_shape.c = DoTranspose ? Local_N : Local_C;
  tl_shape.h = Local_H;
  tl_shape.w = Local_W;

  // quant emit once data type change(e.g int8->bf16 or bf16->int8)
  int is_quant = 0;
  int is_bf16 = 0;

  if (from == CVK_FMT_BF16 || to == CVK_FMT_BF16) {
    // TODO: support other format
    assert(!DoDecompress && "not support bf16 + compress yet");
  }

  if ((from == CVK_FMT_BF16 && to == CVK_FMT_I8) &&
      from == CVK_FMT_I8 && to == CVK_FMT_BF16) {
    // TODO: support U8/BF16 quant
    is_quant = 1;
  }

  if (from == CVK_FMT_BF16 && to == CVK_FMT_BF16) {
    is_bf16 = 1;
  }

  cvk_tl_t tl_data;
  tl_data.start_address = la_dst;
  tl_data.fmt = from;
  tl_data.shape = tl_shape;
  tl_data.stride = ctx.tl_default_stride(tl_shape, tl_data.fmt, DoAligned ? 1 : 0);

  // Global shape used for stride calculation
  cvk_tg_stride_t ga_stride =
      ctx.tg_default_stride(
          {(uint32_t)Local_N, (uint32_t)Global_C,
          (uint32_t)Global_H, (uint32_t)Global_W},
      from);

  cvk_tg_t ga_data = {0};
  ga_data.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_src);
  ga_data.start_address = ga_src;
  ga_data.fmt = to;
  ga_data.shape = {tl_data.shape.n, tl_data.shape.c,
                   tl_data.shape.h, tl_data.shape.w};
  ga_data.stride = ga_stride;

  if (!DoDecompress) {
    // normal data
    cvk_tdma_g2l_tensor_copy_param_t param = {0};
    param.src = &ga_data;
    param.dst = &tl_data;
    if (is_quant || is_bf16) {
      ctx.tdma_g2l_bf16_tensor_copy(&param);
    }
    else {
      ctx.tdma_g2l_tensor_copy(&param);
    }
  } else {
    // Compressed data
    cvk_cmpr_tg_t cmpr_ga_data = {0};
    cmpr_ga_data.t = ga_data;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &cmpr_ga_data;
    param.dst = &tl_data;
    ctx.tdma_g2l_tensor_copy_decompressed(&param);
  }
}

void cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap, cvk_fmt_t fmt,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw, bool do_decompress) {
  cvk_tl_t tl_ifmap;
  tl_ifmap.start_address = la_ifmap;
  tl_ifmap.fmt = fmt;
  tl_ifmap.shape = ctx.shape_t4(n, ic, ih, iw);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, /*eu_align=*/1);
  ctx.set_layer_id(layer_id);

  if (fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8) {
    cvk_tg_stride_t ifmap_gstride = {ic * ih * iw, ih * iw, iw};
    ctx.tdma_load_stride(&tl_ifmap, ga_ifmap, ifmap_gstride,
                       /*do_transpose=*/false, do_decompress);
  } else if (fmt == CVK_FMT_BF16) {
    ctx.tdma_load_bf16(&tl_ifmap, ga_ifmap);
  } else {
    assert(0);
  }
}

void cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap, cvk_fmt_t fmt,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw) {
  cvi_backend_tl_load(ctx, layer_id, la_ifmap, ga_ifmap, fmt, n, ic, ih, iw,
                      /*do_decompress=*/false);
}

void cvi_backend_tl_store(const CviBackendContext &ctx, uint32_t layer_id,
                          laddr_t la_ofmap, gaddr_t ga_ofmap, cvk_fmt_t fmt,
                          uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow) {
  cvk_tl_t tl_ofmap;
  tl_ofmap.start_address = la_ofmap;
  tl_ofmap.fmt = fmt;
  tl_ofmap.shape = ctx.shape_t4(n, oc, oh, ow);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, /*eu_align=*/1);

  ctx.set_layer_id(layer_id);
  if (fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8) {
    ctx.tdma_store(&tl_ofmap, ga_ofmap);
  } else if (fmt == CVK_FMT_BF16) {
    ctx.tdma_store_bf16(&tl_ofmap, ga_ofmap);
  } else {
    assert(0);
  }
}

void cvi_backend_tl_store_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to, bool DoCompress) {
  LLVM_DEBUG(
    llvm::errs() << llvm::format("cvi_backend_tl_store_stride:\n"
                                  "    layer_id %d\n"
                                  "    src (%d, %d, %d, %d), dst (, %d, %d, %d)\n"
                                  "    src 0x%lx, dst 0x%lx\n"
                                  "    DoTranspose %d, DoAligned %d, isNeuron %d"
                                  "    from %d to %d \n",
                                  layer_id, Local_N, Local_C, Local_H, Local_W, Global_C,
                                  Global_H, Global_W, la_src, ga_dst, DoTranspose, DoAligned,
                                  isNeuron, from, to));
  // tensor in local memory
  cvk_tl_shape_t tl_shape;
  tl_shape.n = DoTranspose ? Local_C : Local_N;
  tl_shape.c = DoTranspose ? Local_N : Local_C;
  tl_shape.h = Local_H;
  tl_shape.w = Local_W;

  // quant emit once data type change(e.g int8->bf16 or bf16->int8)
  int is_quant = 0;
  int is_bf16 = 0;

  if ((from == CVK_FMT_BF16 && to == CVK_FMT_I8) &&
      from == CVK_FMT_I8 && to == CVK_FMT_BF16) {
    // TODO: support U8/BF16 quant
    is_quant = 1;
  }

  if (from == CVK_FMT_BF16 && to == CVK_FMT_BF16) {
    is_bf16 = 1;
  }

  cvk_tl_t tl_data;
  tl_data.start_address = la_src;
  tl_data.fmt = from;
  tl_data.shape = tl_shape;
  tl_data.stride = ctx.tl_default_stride(tl_shape, tl_data.fmt, DoAligned ? 1 : 0);

  cvk_tg_stride_t ts_stride =
      ctx.tg_default_stride({(uint32_t)Local_N, (uint32_t)Global_C,
          (uint32_t)Global_H, (uint32_t)Global_W}, to);

  // We need another API to pass memory region from TPU dialect codegen.
  if (is_quant || is_bf16) {
    ctx.tdma_store_stride_bf16(&tl_data, ga_dst, ts_stride,
                              /*do_transpose=*/false, DoCompress);
  } else {
    ctx.tdma_store_stride(&tl_data, ga_dst, ts_stride, /*do_transpose=*/false,
                          DoCompress);
  }
}

void cvi_backend_tl_store_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to) {

  cvi_backend_tl_store_stride(ctx, layer_id, ga_dst, la_src, Local_N, Local_C,
                              Local_H, Local_W, Global_C, Global_H, Global_W,
                              DoTranspose, DoAligned, isNeuron, from, to,
                              false);
}

// Tiled compressed activation split as (n, c, h_step, w)
// Global memory layout: (h/h_step, n, c, h_step, w)
//
// output shape       (1, 64, 35, 112)
// tiled output shape (1, 64,  1, 112)
//
// tiled TDMA store
//   (1, 64, 1, 112)
//   (1, 64, 1, 112)
//   ...
//   (1, 64, 1, 112)
//
//  n   h  c  w
//  0   0  0  0       | header | compressed shape (1, h=1, c=64, w=112) |
//  0   1  0  0
//
//  0  34  0  0
//
void cvi_backend_tl_store_compressed(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to, int h_step, int step_size) {

  // Global shape is used for stride - global memory layout
  assert(from == to && "Expect same data type");

  int eu_align = DoAligned ? 1 : 0;
  cvk_tl_stride_t tl_stride =
    ctx.tl_default_stride(ctx.shape_t4(Local_N, Local_C, Local_H, Local_W),
                          from, eu_align);

  for (int i = 0; i < Local_H; i+=h_step) {
    int cur_h = std::min(h_step, Local_H - i);

  // HxW in each lane is contiguous
    cvk_tl_t tl_src;
    ctx.lmem_init_tensor(&tl_src, ctx.shape_t4(Local_N, Local_C, cur_h, Local_W),
                         from, eu_align);
    tl_src.stride = tl_stride;
    tl_src.start_address = la_src + i * Local_W * tl_stride.w;

    cvk_cmpr_tg_t tg_cmpr_dst = {0};
    ctx.gmem_init_tensor(&tg_cmpr_dst.t,
                         {(uint32_t)Local_N, (uint32_t)Global_C, (uint32_t)cur_h,
                          (uint32_t)Global_W},
                         to);
    tg_cmpr_dst.t.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_dst);
    tg_cmpr_dst.t.start_address = ga_dst + step_size * (i / h_step);

    cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
    param.src = &tl_src;
    param.dst = &tg_cmpr_dst;

    ctx.tdma_l2g_tensor_copy_compressed(&param);
  }
}

void cvi_backend_tl_copy(
  const CviBackendContext &ctx, uint32_t layer_id,
  int la_src, int la_dst,
  int n, int c, int h, int w,
  bool align) {

  ctx.set_layer_id(layer_id);

  cvk_tl_t tl_dst;
  cvk_tl_t tl_src;

  tl_src.start_address = la_src;
  tl_src.fmt = CVK_FMT_I8;
  tl_src.shape = ctx.shape_t4(n, c, h, w);
  tl_src.stride = ctx.tl_default_stride(tl_src.shape, CVK_FMT_I8, align);

  tl_dst.start_address = la_dst;
  tl_dst.fmt = CVK_FMT_I8;
  tl_dst.shape = ctx.shape_t4(n, c, h, w);
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, CVK_FMT_I8, align);

  // tl_copy.dst = &tl_dst;
  // tl_copy.src = &tl_src;
  // ctx.tdma_l2l_tensor_copy(&tl_copy);

  cvk_tiu_mul_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = &tl_dst;
  p2.a = &tl_src;
  p2.b_const.val = 1;
  p2.b_const.is_signed = true;
  p2.b_is_const = 1;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_mul(&p2);
}

void cvi_backend_tl_bf16_ps32_to_fp32(const CviBackendContext &ctx,
                                      uint32_t layer_id, laddr_t la_addr,
                                      int n, int c, int h, int w) {
  assert((n > 1) && ((n % 2) == 0) && "Expect ps32 shape");
  assert((h == 1) && (w == 1) && "Only support h=1, w=1");
  n /= 2; // Exclude lower part

  int eu_align = 1; // the result of tiu operation always align eu
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t shape = ctx.shape_t4(n, c, h, w);
  cvk_tl_stride_t stride = ctx.tl_default_stride(shape, fmt, eu_align);

  uint32_t la_high = la_addr;
  uint32_t la_low = la_addr + stride.n * n;

  cvk_tl_t tl_src;
  ctx.lmem_init_tensor(&tl_src, shape, fmt, eu_align);
  tl_src.start_address = la_high;
  tl_src.shape = shape;
  tl_src.stride = {stride.n, (uint32_t)EU_NUM, stride.h, stride.w};

  cvk_tl_t tl_dst;
  ctx.lmem_init_tensor(&tl_dst, shape, fmt, eu_align);
  tl_dst.start_address = la_low + sizeof(uint16_t); // concat higher part
  tl_dst.shape = shape;
  tl_dst.stride = {stride.n, (uint32_t)EU_NUM, stride.h, stride.w};

  cvk_tdma_l2l_tensor_copy_param_t param = {0};
  param.src = &tl_src;
  param.dst = &tl_dst;
  param.layer_id = layer_id;
  ctx.tdma_l2l_bf16_tensor_copy(&param);
}

void cvi_backend_tl_store_fp32(const CviBackendContext &ctx,
                               uint32_t layer_id, gaddr_t ga_dst,
                               laddr_t la_src,
                               int n, int c, int h, int w)
{
  n /= 2; // Exclude lower part

  int eu_align = 1; // the result of tiu operation always align eu
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t shape = ctx.shape_t4(n, c, h, w);
  cvk_tl_stride_t stride = ctx.tl_default_stride(shape, fmt, eu_align);

  uint32_t la_low = la_src + stride.n * n;

  cvk_tl_t tl_src = {0};
  tl_src.start_address = la_low;
  tl_src.fmt = fmt;
  tl_src.shape = ctx.shape_t4(n, c, h, (sizeof(uint32_t)/sizeof(uint16_t)) * w);
  tl_src.stride = {stride.n, (uint32_t)EU_NUM, stride.h, stride.w};

  cvk_tg_shape_t tg_shape = {(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t) (2 * w)};
  cvk_tg_t tg_dst;
  ctx.gmem_init_tensor(&tg_dst, tg_shape, fmt);
  tg_dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_dst);
  tg_dst.start_address = ga_dst;
  //tg_dst.stride.c = 2;

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = &tl_src;
  param.dst = &tg_dst;
  ctx.tdma_l2g_bf16_tensor_copy(&param);
}
