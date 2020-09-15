/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: src_tl_api.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_api"


void _cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id, laddr_t la_ifmap,
    gaddr_t ga_ifmap, uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw,
    cvk_fmt_t fmt, uint8_t eu_align) {

  // default load from NEURON_MEMORY
  cvk_tl_t tl_ifmap;
  tl_ifmap.start_address = la_ifmap;
  tl_ifmap.fmt = fmt;
  tl_ifmap.shape = ctx.shape_t4(n, ic, ih, iw);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, eu_align);
  cvk_tg_stride_t ifmap_gstride = {ic * ih * iw, ih * iw, iw};

  // TODO: check fmt in ctx.tdma_load
  if (fmt == CVK_FMT_I8) {
    ctx.tdma_load_stride(&tl_ifmap, ga_ifmap, ifmap_gstride);
  }
  else if (fmt == CVK_FMT_BF16) {
    ifmap_gstride = ctx.tg_default_stride({n, ic, ih, iw}, fmt);
    ctx.tdma_load_stride_bf16(&tl_ifmap, ga_ifmap, ifmap_gstride);
  }
  else {
    assert(0 && "not support");
  }
}

void _cvi_backend_tl_store(
    const CviBackendContext &ctx, uint32_t layer_id, laddr_t la_ofmap,
    gaddr_t ga_ofmap, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow,
    cvk_fmt_t fmt, uint8_t eu_align) {
  cvk_tl_t tl_ofmap;
  tl_ofmap.start_address = la_ofmap;
  tl_ofmap.fmt = fmt;
  tl_ofmap.shape = ctx.shape_t4(n, oc, oh, ow);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, eu_align);
  cvk_tg_stride_t ofmap_gstride = {oc * oh * ow, oh * ow, ow};

  if (fmt == CVK_FMT_I8) {
    ctx.tdma_store_stride(&tl_ofmap, ga_ofmap, ofmap_gstride);
  }
  else if (fmt == CVK_FMT_BF16) {
    // TODO bf16 should be prefix rather than shuffix
    ofmap_gstride = ctx.tg_default_stride({n, oc, oh, ow}, fmt);
    ctx.tdma_store_stride_bf16(&tl_ofmap, ga_ofmap, ofmap_gstride);
  }
  else {
    assert(0 && "not support");
  }
}

void cvi_backend_tl_load_tensor(
    const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tensor,
    gaddr_t ga_ifmap, uint8_t eu_align) {

  // TODO: save eu_align info in kernel
  _cvi_backend_tl_load(ctx, layer_id,
      tensor->start_address, ga_ifmap,
      tensor->shape.n,
      tensor->shape.c,
      tensor->shape.h,
      tensor->shape.w,
      tensor->fmt,
      eu_align);
}

void cvi_backend_tl_store_tensor(
    const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tensor,
    gaddr_t ga_ofmap, uint8_t eu_align) {

  _cvi_backend_tl_store(ctx, layer_id,
      tensor->start_address, ga_ofmap,
      tensor->shape.n,
      tensor->shape.c,
      tensor->shape.h,
      tensor->shape.w,
      tensor->fmt,
      eu_align);
}

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
    cvk_fmt_t from, cvk_fmt_t to,
    bool bCompressed) {
  LLVM_DEBUG(
    llvm::errs() << llvm::format("cvi_backend_tl_load_stride:\n"
                                  "    layer_id %d\n"
                                  "    src (, %d, %d, %d), dst (%d, %d, %d, %d)\n"
                                  "    src 0x%lx, dst 0x%lx\n"
                                  "    DoTranspose %d, DoAligned %d, isNeuron %d\n"
                                  "    from %d to %d, Compressed %d\n",
                                  layer_id, Global_C, Global_H, Global_W, Local_N, Local_C,
                                  Local_H, Local_W, ga_src, la_dst, DoTranspose, DoAligned,
                                  isNeuron, from, to, bCompressed));
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
    assert(!bCompressed && "not support bf16 + compress yet");
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

  // Glbal shape used for stride calculation
  cvk_tg_stride_t ga_stride =
      ctx.tg_default_stride(
          {(uint32_t)Local_N, (uint32_t)Global_C,
          (uint32_t)Global_H, (uint32_t)Global_W},
      from);

  if (!bCompressed) {
    // normal data
    cvk_tg_t ga_data = {0};
    ga_data.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_src);
    ga_data.start_address = ga_src;
    ga_data.fmt = to;
    ga_data.shape = {tl_data.shape.n, tl_data.shape.c,
                      tl_data.shape.h, tl_data.shape.w};
    ga_data.stride = ga_stride;
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
    cvk_cmpr_tg_t ga_data = {0};
    ga_data.t.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_src);
    ga_data.t.start_address = ga_src;
    ga_data.t.fmt = to;
    ga_data.t.shape = {tl_data.shape.n, tl_data.shape.c,
                      tl_data.shape.h, tl_data.shape.w};
    ga_data.t.stride = ga_stride;
    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &ga_data;
    param.dst = &tl_data;
    ctx.tdma_g2l_tensor_copy_decompressed(&param);
  }
}


void cvi_backend_tl_store_stride(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_dst, laddr_t la_src,
    int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W,
    bool DoTranspose, bool DoAligned, bool isNeuron,
    cvk_fmt_t from, cvk_fmt_t to) {
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
    ctx.tdma_store_stride_bf16(&tl_data, ga_dst, ts_stride);
  } else {
    ctx.tdma_store_stride(&tl_data, ga_dst, ts_stride);
  }
}
