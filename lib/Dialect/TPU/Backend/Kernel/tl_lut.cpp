/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_lut.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_lut"

void cvi_backend_tl_lut_LA(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    gaddr_t ga_input, gaddr_t ga_output, gaddr_t sg_lut_gaddr,
    int n, int c, int h, int w,
    bool do_load, bool do_store) {

  ctx.set_layer_id(layer_id);
  int oh = h;
  int ow = w;
  uint32_t input_size = ctx.lmem_tensor_to_size({(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w}, CVK_FMT_I8, /*eu_align=*/1);
  uint32_t output_size = ctx.lmem_tensor_to_size({(uint32_t)n, (uint32_t)c, (uint32_t)oh, (uint32_t)ow}, CVK_FMT_I8, /*eu_align=*/1);
  uint32_t working_size = LOCAL_MEM_SIZE - input_size - output_size;
  bool isEnoughSize = (working_size  < 256) ? false : true;
  assert(isEnoughSize);

  LLVM_DEBUG(
    llvm::errs() << "    Lut    : nchw = ("
                 << n << "," << c << "," << h << "," << w << ")"
                 << ", working_size " << working_size
                 << "\n                     "
                 << "la_i = " << la_input
                 << ", la_o = " << la_output
                 << ", la_w = " << la_working
                 << "\n                     "
                 << "ga_i = " << ga_input
                 << ", ga_o = " << ga_output
                 << ", ga_lut = " << sg_lut_gaddr
                 << "\n";
  );

  int table_w, table_h;
  table_w = 16;
  table_h = 16;

  bool prealloc = false;
  if (la_input != LA_INVALID) { //some checks
    prealloc = true;
    assert(la_output != LA_INVALID && la_working != LA_INVALID);
    laddr_t la_max = std::max(la_input, la_output);
    assert(la_working >= input_size);
    assert(la_max + output_size <= (uint32_t)LOCAL_MEM_SIZE);
  }

  cvk_tl_t *tl_input = nullptr;
  cvk_tl_t *tl_output = nullptr;
  cvk_tl_t *sg_lut_table = nullptr;

  if(prealloc) {
      tl_input = new cvk_tl_t;
      tl_output = new cvk_tl_t;
      sg_lut_table = new cvk_tl_t;

      tl_input->start_address = la_input;
      tl_input->fmt = CVK_FMT_I8;
      tl_input->shape = ctx.shape_t4(n, c, h, w);
      tl_input->stride = ctx.tl_default_stride(tl_input->shape, CVK_FMT_I8, /*eu_align=*/1);

      tl_output->start_address = la_output;
      tl_output->fmt = CVK_FMT_I8;
      tl_output->shape = ctx.shape_t4(n, c, oh, ow);
      tl_output->stride = ctx.tl_default_stride(tl_output->shape, CVK_FMT_I8, /*eu_align=*/1);

      sg_lut_table->start_address = la_working;
  } else {
      tl_input = ctx.lmem_alloc_tensor(ctx.shape_t4(n, c, h, w), CVK_FMT_I8, /*eu_align=*/1);
      tl_output = ctx.lmem_alloc_tensor(ctx.shape_t4(n, c, oh, ow), CVK_FMT_I8, /*eu_align=*/1);
      sg_lut_table = ctx.lmem_alloc_tensor(ctx.shape_t4(1, NPU_NUM, 16, 16), CVK_FMT_I8, /*eu_align=*/1);
      assert(sg_lut_table && tl_input && tl_output);
  }


  // load lut table
  sg_lut_table->fmt = CVK_FMT_I8;
  sg_lut_table->shape = ctx.shape_t4(1, NPU_NUM, table_h, table_w);
  sg_lut_table->stride = ctx.tl_default_stride(sg_lut_table->shape, CVK_FMT_I8, /*eu_align=*/1);
  ctx.tdma_load(sg_lut_table, sg_lut_gaddr);

  // global memory stride from global memory shape
  cvk_tg_stride_t gstride = {(uint32_t)(c * h * w), (uint32_t)(h * w), (uint32_t)w}; //inputStride
  cvk_tg_stride_t gstride_output = {(uint32_t)(c * oh * ow), (uint32_t)(oh * ow), (uint32_t)ow}; //outputStride
  // load input
  if (do_load) {
    ctx.tdma_load_stride(tl_input, ga_input, gstride);
  }

  //compute
  cvk_tiu_lookup_table_param_t p12 = {0};
   p12.ifmap = tl_input;
   p12.ofmap = tl_output;
   p12.table = sg_lut_table;
   p12.layer_id = layer_id;
   ctx.tiu_lookup_table(&p12);

  // store output
  if (do_store) {
    ctx.tdma_store_stride(tl_output, ga_output, gstride_output);
  }

  //
  // Release resource in reverse order
  //
  if (prealloc) {
    delete sg_lut_table;
    delete tl_output;
    delete tl_input;
  } else {
    ctx.lmem_free_tensor(sg_lut_table);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
}


// for layer group

void cvi_backend_tl_lut(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_y_table, laddr_t la_slope_lut,
    int thresh_min, int thresh_max, int n, int c, int h, int w) {

  ctx.set_layer_id(layer_id);
  LLVM_DEBUG(
    llvm::errs() << "    Lut    : nchw = ("
                 << n << "," << c << "," << h << "," << w << ")"
                 << "\n                     "
                 << "la_i = " << la_input
                 << ", la_o = " << la_output
                 << ", la_w = " << la_working
                 << ", la_lut = " << la_y_table
                 << "\n";
  );

  if (la_working == 0 && la_slope_lut == 0) {
    // assign in MixNet.cpp::_add_tl_activation_op
    // int8 lut

    int table_w, table_h;
    table_w = 16;
    table_h = 16;

    cvk_tl_t *tl_input = new cvk_tl_t;
    cvk_tl_t *tl_output = new cvk_tl_t;
    cvk_tl_t *tl_y_table = new cvk_tl_t;

    tl_input->start_address = la_input;
    tl_input->fmt = CVK_FMT_I8;
    tl_input->shape = ctx.shape_t4(n, c, h, w);
    tl_input->stride = ctx.tl_default_stride(tl_input->shape, CVK_FMT_I8, /*eu_align=*/1);

    tl_output->start_address = la_output;
    tl_output->fmt = CVK_FMT_I8;
    tl_output->shape = ctx.shape_t4(n, c, h, w);
    tl_output->stride = ctx.tl_default_stride(tl_output->shape, CVK_FMT_I8, /*eu_align=*/1);

    tl_y_table->start_address = la_y_table;
    tl_y_table->fmt = CVK_FMT_I8;
    tl_y_table->shape = ctx.shape_t4(1, NPU_NUM, table_h, table_w);
    tl_y_table->stride = ctx.tl_default_stride(tl_y_table->shape, CVK_FMT_I8, /*eu_align=*/1);

    //compute
    cvk_tiu_lookup_table_param_t p12 = {0};
    p12.ifmap = tl_input;
    p12.ofmap = tl_output;
    p12.table = tl_y_table;
    p12.layer_id = layer_id;
    ctx.tiu_lookup_table(&p12);

    delete tl_y_table;
    delete tl_output;
    delete tl_input;
  }
  else {
    // bf16 lut
    // FIXME: support sqrt/reciprocal
    // FIXME: copy from lut.cpp, need to replace ctx.tiu_bf16_lookup_interp_table
    // init
    bool isSync = (abs(thresh_min) == thresh_max);
    float offset = (float)(thresh_max + thresh_min) / 2;
    int range = (thresh_max - thresh_min);
    const int lut_index_num = 256;
    int input_n, input_c, input_h, input_w;
    input_n = n;
    input_c = c;
    input_h = h;
    input_w = w;
    cvk_tl_t _tl_ifmap, _tl_ofmap_slope, _tl_ofmap_y0, _tl_table_answer, _tl_table_answer_slope;
    cvk_tl_t *tl_ifmap, *tl_ofmap_slope, *tl_ofmap_y0, *tl_table_answer, *tl_table_answer_slope;
    tl_ifmap = &_tl_ifmap;
    tl_ofmap_slope = &_tl_ofmap_slope;
    tl_ofmap_y0 = &_tl_ofmap_y0;
    tl_table_answer = &_tl_table_answer;
    tl_table_answer_slope = &_tl_table_answer_slope;

    // input
    tl_ifmap->start_address = la_input;
    tl_ifmap->fmt = CVK_FMT_BF16;
    tl_ifmap->shape = ctx.shape_t4(n, c, h, w);
    tl_ifmap->stride = ctx.tl_default_stride(tl_ifmap->shape, tl_ifmap->fmt, /*eu_align=*/1);
    // output
    tl_ofmap_y0->start_address = la_output;
    tl_ofmap_y0->fmt = tl_ifmap->fmt;
    tl_ofmap_y0->shape = tl_ifmap->shape;
    tl_ofmap_y0->stride = tl_ifmap->stride;
    // working
    tl_ofmap_slope->start_address = la_working;
    tl_ofmap_slope->fmt = tl_ifmap->fmt;
    tl_ofmap_slope->shape = tl_ifmap->shape;
    tl_ofmap_slope->stride = tl_ifmap->stride;

    // 1880v2 hw setting
    int const table_n = 1;
    int const table_h = 32;
    int const table_w = 8;

    // y0
    tl_table_answer->start_address = la_y_table;
    tl_table_answer->fmt = CVK_FMT_BF16;
    tl_table_answer->shape = ctx.shape_t4(table_n, NPU_NUM, table_h, table_w);
    tl_table_answer->stride = ctx.tl_default_stride(tl_table_answer->shape, CVK_FMT_BF16, /*eu_align=*/1);

    // slope
    tl_table_answer_slope->start_address = la_slope_lut;
    tl_table_answer_slope->fmt = tl_table_answer->fmt;
    tl_table_answer_slope->shape = tl_table_answer->shape;
    tl_table_answer_slope->stride = tl_table_answer->stride;

    cvk_tl_shape_t tl_ofmap_x0_int8_shape = {
        (uint32_t)input_n, (uint32_t)input_c,
        (uint32_t)(input_h * input_w), 1
    };

    if(!isSync){
      cvk_tiu_add_param_t p90 = {0};
      p90.res_high = nullptr;
      p90.res_low = tl_ifmap;
      p90.a_high = nullptr;
      p90.a_low = tl_ifmap;
      p90.b_is_const = true;
      p90.b_const.val = ctx.convert_fp32_to_bf16((-1.0 * offset));
      p90.rshift_bits = 0;
      p90.layer_id = layer_id;
      p90.relu_enable = 0;
      ctx.tiu_add(&p90);
    }

    cvk_tdma_l2l_tensor_copy_param_t p3 = {0};
    // scale input for remap its idx(-x~x) to (-127~127), dirty tl_ifmap
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = NULL;
    p4.res_low = tl_ifmap;
    p4.a = tl_ifmap;
    p4.b_is_const = 1;
    p4.b_const.val = ctx.convert_fp32_to_bf16((float)lut_index_num / (float)range);
    p4.rshift_bits = 0;
    p4.relu_enable = 0;
    p4.layer_id = layer_id;
    ctx.tiu_mul(&p4);

    // <! get idx from bf16->int8
    memset(&p3, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
    cvk_tl_t dst;
    memcpy(&dst, tl_ofmap_y0, sizeof(cvk_tl_t));

    // fill low 8 bit for lut index
    dst.shape = tl_ofmap_x0_int8_shape;
    dst.fmt = CVK_FMT_I8;
    cvk_tl_shape_t _tl_ofmap_x0_int8_shape = {
      (uint32_t)input_n, (uint32_t)input_c, (uint32_t)(input_h * input_w), 2};
    int int8_eu_align = 1;
    dst.stride =
      ctx.tl_default_stride(_tl_ofmap_x0_int8_shape, CVK_FMT_I8, /*eu_align=*/int8_eu_align);
    dst.int8_rnd_mode = 1;
    p3.dst = &dst;
    p3.src = tl_ifmap;
    ctx.tdma_l2l_bf16_tensor_copy(&p3);
    dst.int8_rnd_mode = 0; // reset

    // <! int8 to bf16 format cus for sub use, sub MUST in the same format
    memset(&p3, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
    p3.dst = tl_ofmap_slope; //<! bf16
    p3.src = &dst;
    ctx.tdma_l2l_bf16_tensor_copy(&p3);

    // <! sub, diff base , a - b
    // (x - x0)
    cvk_tiu_sub_param_t p5 = {0};
    p5.res_high = 0;
    p5.res_low = tl_ifmap;
    p5.a_high = 0;
    p5.a_low = tl_ifmap;
    p5.b_high = 0;
    p5.b_low = tl_ofmap_slope;
    p5.rshift_bits = 0;
    p5.layer_id = layer_id;
    ctx.tiu_sub(&p5);

    // get f(x0) and slope(x)
    // reshape, 16->16
    dst.fmt = CVK_FMT_BF16;
    dst.shape = tl_ofmap_slope->shape;
    dst.stride = tl_ofmap_slope->stride;

    // <! get slope by index
    cvk_tiu_lookup_table_param_t p6 = {0};
    memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
    p6.ofmap = tl_ofmap_slope;
    p6.ifmap = &dst;
    p6.table = tl_table_answer_slope;
    p6.layer_id = layer_id;
    ctx.tiu_lookup_table(&p6);

    // base f(x0)
    memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
    p6.ofmap = tl_ofmap_y0;
    p6.ifmap = &dst;
    p6.table = tl_table_answer;
    p6.layer_id = layer_id;
    ctx.tiu_lookup_table(&p6);

    // <! mac
    // <! part A + part B, a * b + res = res
    cvk_tiu_mac_param_t p7 = {0};
    p7.res_high = 0;
    p7.res_low = tl_ofmap_y0;
    p7.res_is_int8 = 0;
    p7.a = tl_ifmap;
    p7.b_is_const = 0;
    p7.b = tl_ofmap_slope;
    p7.lshift_bits = 0; // lshift_bits;
    p7.rshift_bits = 0; // rshift_bits;
    p7.relu_enable = 0;
    p7.layer_id = layer_id;
    ctx.tiu_mac(&p7);
  }
}

void cvi_backend_tl_lut_exponential_mul_mantissa(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_exponential_table, laddr_t la_mantissa_lut, int n, int c, int h, int w) {

  ctx.set_layer_id(layer_id);
  LLVM_DEBUG(
    llvm::errs() << "    Lut    : nchw = ("
                 << n << "," << c << "," << h << "," << w << ")"
                 << "\n                     "
                 << "la_i = " << la_input
                 << ", la_o = " << la_output
                 << ", la_w = " << la_working
                 << ", la_lut = " << la_exponential_table
                 << ", la_lut = " << la_mantissa_lut
                 << "\n";
  );

  cvk_tl_t _tl_ifmap, _tl_ofmap_slope, _tl_ofmap_y0, _tl_table_answer, _tl_table_answer_mantissa;
  cvk_tl_t *tl_ifmap, *tl_ofmap_slope, *tl_ofmap_y0, *tl_table_answer, *tl_table_answer_mantissa;
  tl_ifmap = &_tl_ifmap;
  tl_ofmap_slope = &_tl_ofmap_slope;
  tl_ofmap_y0 = &_tl_ofmap_y0;
  tl_table_answer = &_tl_table_answer;
  tl_table_answer_mantissa = &_tl_table_answer_mantissa;

  // input
  tl_ifmap->start_address = la_input;
  tl_ifmap->fmt = CVK_FMT_BF16;
  tl_ifmap->shape = ctx.shape_t4(n, c, h, w);
  tl_ifmap->stride = ctx.tl_default_stride(tl_ifmap->shape, tl_ifmap->fmt, /*eu_align=*/1);
  // output
  tl_ofmap_y0->start_address = la_output;
  tl_ofmap_y0->fmt = tl_ifmap->fmt;
  tl_ofmap_y0->shape = tl_ifmap->shape;
  tl_ofmap_y0->stride = tl_ifmap->stride;
  // working
  tl_ofmap_slope->start_address = la_working;
  tl_ofmap_slope->fmt = tl_ifmap->fmt;
  tl_ofmap_slope->shape = tl_ifmap->shape;
  tl_ofmap_slope->stride = tl_ifmap->stride;

  // 1880v2 hw setting
  int const table_n = 1;
  int const table_h = 32;
  int const table_w = 8;

  // y0
  tl_table_answer->start_address = la_exponential_table;
  tl_table_answer->fmt = CVK_FMT_BF16;
  tl_table_answer->shape = ctx.shape_t4(table_n, NPU_NUM, table_h, table_w);
  tl_table_answer->stride = ctx.tl_default_stride(tl_table_answer->shape, CVK_FMT_BF16, /*eu_align=*/1);

  // mantissa
  tl_table_answer_mantissa->start_address = la_mantissa_lut;
  tl_table_answer_mantissa->fmt = tl_table_answer->fmt;
  tl_table_answer_mantissa->shape = tl_table_answer->shape;
  tl_table_answer_mantissa->stride = tl_table_answer->stride;

  // issue lut cmd
  cvk_tdma_l2l_tensor_copy_param_t p10;
  // remove low 8 bits by int8 copy with stride
  // <! get index(pow)
  memset(&p10, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
  p10.dst = tl_ofmap_y0;
  p10.src = tl_ifmap;
  p10.mv_lut_base = false; // MUST init by ifself in soc
  p10.mv_lut_idx = true;
  p10.layer_id = layer_id;
  ctx.tdma_l2l_bf16_tensor_copy(&p10);

  // <! get f(x0) = 2^(x0*-0.5)
  cvk_tiu_lookup_table_param_t p12={0};
  p12.layer_id = layer_id;
  p12.ofmap = tl_ofmap_y0;
  p12.ifmap = tl_ofmap_y0;
  p12.table = tl_table_answer;
  ctx.tiu_lookup_table(&p12);

  // <! get mantissa value
  p12.ofmap = tl_ifmap;
  p12.ifmap = tl_ifmap;
  p12.table = tl_table_answer_mantissa;
  ctx.tiu_lookup_table(&p12);

  // sqrt = (2^exp) * mantissa
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = tl_ofmap_y0;
  p.a =tl_ofmap_y0;
  p.b = tl_ifmap;
  p.b_is_const = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = false;
  ctx.tiu_mul(&p);
}