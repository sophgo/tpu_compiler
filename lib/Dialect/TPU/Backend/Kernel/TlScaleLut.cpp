/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_scale_lut.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_scale_lut"

void cvi_backend_tl_scale_lut(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, laddr_t table_laddr,
    int input_n, int input_c, int input_h, int input_w) {

  cvk_tl_shape_t lshape = ctx.tl_shape_t4(input_n,input_c,input_h,input_w);
  cvk_tl_shape_t table_shape = ctx.tl_shape_t4(1, NPU_NUM, 16, 16);

  cvk_tl_t bottom = {0};
  bottom.start_address = ifmap_laddr;
  bottom.fmt = CVK_FMT_U8;
  bottom.shape = lshape;
  bottom.stride = ctx.tl_default_stride(lshape, CVK_FMT_U8, 1);

  cvk_tl_t top = {0};
  top.start_address = ofmap_laddr;
  top.fmt = CVK_FMT_I8;
  top.shape = lshape;
  top.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);

  cvk_tl_t table = {0};
  table.start_address = table_laddr;
  table.fmt = CVK_FMT_I8;
  table.shape = table_shape;
  table.stride = ctx.tl_default_stride(table_shape, CVK_FMT_I8, 1);

  cvk_tiu_lookup_table_param_t p = {0};
  p.ofmap = &top;
  p.ifmap = &bottom;
  p.table = &table;
  p.layer_id = layer_id;
  ctx.tiu_lookup_table(&p);
}