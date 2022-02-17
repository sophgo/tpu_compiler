/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgBf16MatchTemplateKenrel.cpp
 * Description:
 */
#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_match_template_kernel"

#define ASSERT(x) assert(x)

static cvk_tl_t *load(const CviBackendContext &ctx, cvk_tl_shape_t &tl_shape,
                            cvk_tg_stride_t &gstride, uint64_t ga_src, cvk_fmt_t ifmt,
                            cvk_fmt_t fmt, uint32_t layer_id, float_t mean) {
  cvk_tl_t *tl_ifmap = ctx.lmem_alloc_tensor(tl_shape, fmt, 1);
  ASSERT(tl_ifmap);
  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  cvk_tg_t tg_src;
  tg_src.start_address = ga_src;
  tg_src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(tg_src.start_address);
  tg_src.int8_rnd_mode = 0;
  tg_src.fmt = ifmt;
  tg_src.shape = ctx.tg_shape_t4(tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w);
  tg_src.stride = gstride;
  p.src = &tg_src;
  p.dst = tl_ifmap;
  ctx.tdma_g2l_tensor_copy(&p);
  // temporary fix overflow issue, for some extream case it still don't work
  if (mean != 0){
      cvk_tiu_add_param_t p1 = {0};
      p1.res_high = nullptr;
      p1.res_low = tl_ifmap;
      p1.a_high = nullptr;
      p1.a_low = tl_ifmap;
      p1.b_is_const = true;
      p1.b_const.val = ctx.convert_fp32_to_bf16(-mean);
      p1.b_const.is_signed = 1;
      p1.rshift_bits = 0;
      p1.layer_id = layer_id;
      p1.relu_enable = false;
      ctx.tiu_add(&p1);
  }

  return tl_ifmap;
}

static cvk_tl_t *load_template(const CviBackendContext &ctx, int64_t ga_src,
                               int32_t c_step, int32_t h, int32_t w,
                               cvk_fmt_t ifmt, cvk_fmt_t fmt,
                               bool boradcast, uint32_t layer_id, float_t &mean) {
  cvk_tl_shape_t tl_tshape;
  if (boradcast)
    // load and broadcast template to lanes
    tl_tshape = ctx.tl_shape_t4(1, NPU_NUM, h, w);
  else
    tl_tshape = ctx.tl_shape_t4(1, c_step, h, w);
  cvk_tg_stride_t tg_tstride = ctx.tg_default_stride({1, 1, tl_tshape.h, tl_tshape.w}, ifmt);
  tg_tstride.n = 0;
  tg_tstride.c = 0;
  cvk_tl_t *tl_template = load(ctx, tl_tshape, tg_tstride, ga_src, ifmt, fmt, layer_id, mean);
  return tl_template;
}

// _____________         _____________
// ||slide |    |        |            |      output
// ||window|th  |        |     _______|     _____
// ||__tw__|   ih ...... |     |      | --> |    | n
// |            |        |     |      |     |____|
// |______iw____|        |_____|______|       c

void cvi_backend_tg_bf16_match_template_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_input, gaddr_t ga_template,
    gaddr_t ga_table, gaddr_t ga_mantissa_table,
    gaddr_t ga_output, int ih, int iw,
    int th, int tw, const char* mode) {
  int32_t n, c, h, w, stride, outer_size, reduce_size;
  uint32_t lmem_used = 0;
  cvk_fmt_t fmt = CVK_FMT_BF16;
  // according to which set in Quantization.cpp:145
  cvk_fmt_t ifmt = CVK_FMT_U8;
  float_t mean = 0.;
  int32_t g_elt_size = ctx.bytesize_of_fmt(ifmt);
  bool boradcast = false;

  // reshape input
  n = ih - th + 1;
  c = iw - tw + 1;
  h = th;
  w = tw;
  stride = iw;
  outer_size = n * c;
  reduce_size = h * w;

  if (h >= MAX_WIDTH || w >= MAX_WIDTH) {
    llvm::errs() << llvm::format("Template size[%d] is too large\n", reduce_size);
    assert(0);
  }
  // load table
  cvk_tl_shape_t table_shape = ctx.lut_table_shape(fmt);
  cvk_tl_t *tl_lut = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *tl_lut_mantissa = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  ctx.tdma_load(tl_lut, ga_table);
  ctx.tdma_load(tl_lut_mantissa, ga_mantissa_table);
  lmem_used += 2 * ctx.lmem_tensor_to_size(table_shape, fmt, 1);
  // tile policy
  int c_step = std::min(outer_size, MAX_CHANNEL);
  while (c_step > 0) {
    // for table
    uint32_t mem_need = lmem_used;
    // for input
    mem_need += ctx.lmem_tensor_to_size(1, c_step, h, w, fmt, 1);
    // for intermidate value and output
    uint32_t out_mem_need = ctx.lmem_tensor_to_size(1, c_step, 1, 1, fmt, 1);
    if (!strcmp(mode, "TM_CCOEFF_NORMED")) {
      // for template. boradcast defult is false
      mean = 128.;
      if (boradcast)
        mem_need += ctx.lmem_tensor_to_size(1, NPU_NUM, h, w, fmt, 1); // for test
      else
        mem_need += ctx.lmem_tensor_to_size(1, c_step, h, w, fmt, 1);
      // 4 means tl_out, tl_inter_res, tl_buf, tl_lut_out
      mem_need +=  4 * out_mem_need;
    }
    else if(!strcmp(mode, "TM_SQDIFF")){
      // broadcast template,
      mem_need += ctx.lmem_tensor_to_size(1, NPU_NUM, h, w, fmt, 1);
      // for output, tl_buf, tl_lut_out
      mem_need += 3 * out_mem_need;
      boradcast = true;
    }
    else {
      llvm::errs() << llvm::format("Match template not support [%s] method.\n", mode);
      assert(0);
    }
    if (mem_need <= (uint32_t) LOCAL_MEM_SIZE){
      break;
    }
    if(c_step % NPU_NUM != 0){
      c_step -= c_step % NPU_NUM;
    } else {
      c_step -= NPU_NUM;
    }
  }

  if (c_step <= 0){
    llvm::errs() << llvm::format("Tilling Match Template failed, src shape:[1,%d,%d,%d]\n",
                                outer_size, h, w);
    assert(0);
  }

  cvk_tl_t *tl_template = load_template(ctx, ga_template, c_step, h, w,
                                        ifmt, fmt, boradcast, layer_id, mean);

  cvk_tg_stride_t tg_istride = {
      (uint32_t)(stride * g_elt_size), (uint32_t)g_elt_size,
      (uint32_t)(stride * g_elt_size), (uint32_t)g_elt_size
      };

  if (!strcmp(mode, "TM_CCOEFF_NORMED")) {
    ctx.parallel_disable();
    for (int c_pos = 0; c_pos < outer_size;){
      int _c = std::min(c_step, outer_size - c_pos);
      uint64_t in_offset = (c_pos / c * stride + c_pos % c) * g_elt_size;
      uint64_t out_offset = c_pos * ctx.bytesize_of_fmt(fmt);
      if (c_pos / c != (c_pos + _c - 1) / c){
        // if end idx in next row , cut off rest and next loop will cal from next row.
        _c -= (c_pos + _c) % c;  // num of windows keep
        c_pos += _c;
      }
      else{
        c_pos += c_step;
      }
      // load input. For now input assume always be uint8
      cvk_tl_shape_t tl_ishape = ctx.tl_shape_t4(1, _c, h, w);
      cvk_tl_shape_t tl_oshape = ctx.tl_shape_t4(1, _c, 1, 1);
      cvk_tl_t *tl_input = load(ctx, tl_ishape, tg_istride, ga_input + in_offset,
                                ifmt, fmt, layer_id, mean);
      cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_oshape, fmt, 1);
      cvk_tl_t *tl_inter_res = ctx.lmem_alloc_tensor(tl_oshape, fmt, 1);
      // cal reduce_sum(pow(input, 2))
      cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
      p2.ofmap = tl_inter_res;
      p2.ifmap = tl_input;
      p2.weight = tl_input;
      p2.bias = nullptr;
      p2.ins_h = 0;
      p2.ins_w = 0;
      p2.ins_last_h = 0;
      p2.ins_last_w = 0;
      p2.pad_bottom = 0;
      p2.pad_top= 0;
      p2.pad_left = 0;
      p2.pad_right = 0;
      p2.stride_h = 1;
      p2.stride_w = 1;
      p2.dilation_h = 1;
      p2.dilation_w = 1;
      p2.rshift_bits = 0;
      p2.relu_enable = 0;
      p2.layer_id = layer_id;
      ctx.tiu_pt_depthwise_convolution(&p2);
      // cal reduce_sum(mul(input, tmplate))
      cvk_tl_t _tl_template;
      _tl_template.start_address = tl_template->start_address;
      _tl_template.shape = tl_input->shape;
      _tl_template.stride = tl_input->stride;
      _tl_template.fmt = tl_template->fmt;
      if (boradcast){
        _tl_template.stride.n = 0;
        _tl_template.stride.c = 0;
        cvk_tiu_mul_param_t p0 = {0};
        p0.res_high = nullptr;
        p0.res_low = tl_input;
        p0.a = tl_input;
        p0.b = &_tl_template;
        p0.b_is_const = 0;
        p0.rshift_bits = 0;
        p0.layer_id = layer_id;
        p0.relu_enable = 0;
        ctx.tiu_mul(&p0);
        cvk_tiu_average_pooling_param_t param = {0};
        param.ofmap = tl_output;
        param.ifmap = tl_input;
        param.kh = h;
        param.kw = w;
        param.pad_top = 0;
        param.pad_bottom = 0;
        param.pad_left = 0;
        param.pad_right = 0;
        param.stride_h = 1;
        param.stride_w = 1;
        param.avg_pooling_const = ctx.convert_fp32_to_bf16(h * w);
        param.layer_id = layer_id;
        param.ins_val = 0;
        param.ins_fp = param.avg_pooling_const;
        ctx.tiu_average_pooling(&param);
      } else{
        // it seems more efficient in cuerrent case.
        cvk_tiu_depthwise_pt_convolution_param_t p1 = {0};
        p1.ofmap = tl_output;
        p1.ifmap = tl_input;
        p1.weight = tl_template;
        p1.bias = nullptr;
        p1.ins_h = 0;
        p1.ins_w = 0;
        p1.ins_last_h = 0;
        p1.ins_last_w = 0;
        p1.pad_bottom = 0;
        p1.pad_top= 0;
        p1.pad_left = 0;
        p1.pad_right = 0;
        p1.stride_h = 1;
        p1.stride_w = 1;
        p1.dilation_h = 1;
        p1.dilation_w = 1;
        p1.rshift_bits = 0;
        p1.relu_enable = 0;
        p1.layer_id = layer_id;
        ctx.tiu_pt_depthwise_convolution(&p1);
      }
      // lut reduce)sum(sqrt(power(input, 2)))
      cvk_tl_t *tl_buf = ctx.lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
      cvk_tl_t *tl_lut_out = ctx.lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
      cvk_tiu_bf16_lookup_interp_table_param_t p3 = {0};
      p3.ifmap = tl_inter_res;
      p3.buf = tl_buf;
      p3.tbl_answer = tl_lut;
      p3.tbl_answer_mantissa = tl_lut_mantissa;
      p3.ofmap = tl_lut_out;
      p3.is_scientific = 1;
      ctx.tiu_bf16_lookup_interp_table(&p3);
      // mul numerator and denominator
      cvk_tiu_mul_param_t p4 = {0};
      p4.res_high = nullptr;
      p4.res_low = tl_output;
      p4.a = tl_output;
      p4.b = tl_lut_out;
      p4.b_is_const = 0;
      p4.rshift_bits = 0;
      p4.layer_id = layer_id;
      p4.relu_enable = 0;
      ctx.tiu_mul(&p4);

      ctx.tdma_store(tl_output, ga_output + out_offset);
      ctx.lmem_free_tensor(tl_lut_out);
      ctx.lmem_free_tensor(tl_buf);
      ctx.lmem_free_tensor(tl_inter_res);
      ctx.lmem_free_tensor(tl_output);
      ctx.lmem_free_tensor(tl_input);
    } // end for
  } // end if
  else {
    // param: prevent overflow
    float_t scale = 1. / (h * w * 100);
    float_t esp = 1e-5;
    ctx.parallel_disable();
    for (int c_pos = 0; c_pos < outer_size;){
      int _c = std::min(c_step, outer_size - c_pos);
      uint64_t in_offset = (c_pos / c * stride + c_pos % c) * g_elt_size;
      uint64_t out_offset = c_pos * ctx.bytesize_of_fmt(fmt);
      if (c_pos / c != (c_pos + _c - 1) / c){
        _c -= (c_pos + _c) % c;
        c_pos += _c;
      }
      else{
        c_pos += c_step;
      }
      cvk_tl_shape_t tl_ishape = ctx.tl_shape_t4(1, _c, h, w);
      cvk_tl_shape_t tl_oshape = ctx.tl_shape_t4(1, _c, 1, 1);
      cvk_tl_t *tl_input = load(ctx, tl_ishape, tg_istride, ga_input + in_offset,
                                ifmt, fmt, layer_id, mean);
      cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_oshape, fmt, 1);
      cvk_tl_t _tl_template;
      _tl_template.start_address = tl_template->start_address;
      _tl_template.shape = tl_input->shape;
      _tl_template.stride = tl_input->stride;
      _tl_template.stride.n = 0;
      _tl_template.stride.c = 0;
      _tl_template.fmt = tl_template->fmt;
      // cal input - tmplate
      cvk_tiu_sub_param_t p1 = {0};
      p1.res_high = 0;
      p1.res_low = tl_input;
      p1.a_high = 0;
      p1.a_low = tl_input;
      p1.b_high = 0;
      p1.b_low = &_tl_template;
      p1.rshift_bits = 0;
      p1.layer_id = layer_id;
      ctx.tiu_sub(&p1);
      // cal reduce_sum(pow((input - tmplate), 2))
      cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
      p2.ofmap = tl_output;
      p2.ifmap = tl_input;
      p2.weight = tl_input;
      p2.bias = nullptr;
      p2.ins_h = 0;
      p2.ins_w = 0;
      p2.ins_last_h = 0;
      p2.ins_last_w = 0;
      p2.pad_bottom = 0;
      p2.pad_top= 0;
      p2.pad_left = 0;
      p2.pad_right = 0;
      p2.stride_h = 1;
      p2.stride_w = 1;
      p2.dilation_h = 1;
      p2.dilation_w = 1;
      p2.rshift_bits = 0;
      p2.relu_enable = 0;
      p2.layer_id = layer_id;
      ctx.tiu_pt_depthwise_convolution(&p2);
      // diff from ccoeff sqdiff need min value
      cvk_tiu_mul_param_t p3 = {0};   // prevent precision truncation
      p3.res_high = nullptr;
      p3.res_low = tl_output;
      p3.a = tl_output;
      p3.b_const.val = ctx.convert_fp32_to_bf16(scale);
      p3.b_const.is_signed = 1;
      p3.b_is_const = true;
      p3.rshift_bits = 0;
      p3.layer_id = layer_id;
      p3.relu_enable = false;
      ctx.tiu_mul(&p3);
      cvk_tiu_add_param_t p4 = {0};  // prevent overflow
      p4.res_high = nullptr;
      p4.res_low = tl_output;
      p4.a_high = nullptr;
      p4.a_low = tl_output;
      p4.b_is_const = true;
      p4.b_const.val = ctx.convert_fp32_to_bf16(esp);
      p4.b_const.is_signed = 1;
      p4.rshift_bits = 0;
      p4.layer_id = layer_id;
      p4.relu_enable = false;
      ctx.tiu_add(&p4);
      // convert max to min
      cvk_tl_t *tl_buf = ctx.lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
      cvk_tl_t *tl_lut_out = ctx.lmem_alloc_tensor(tl_output->shape, tl_output->fmt, 1);
      cvk_tiu_bf16_lookup_interp_table_param_t p5 = {0};
      p5.ifmap = tl_output;
      p5.buf = tl_buf;
      p5.tbl_answer = tl_lut;
      p5.tbl_answer_mantissa = tl_lut_mantissa;
      p5.ofmap = tl_lut_out;
      p5.is_scientific = 1;
      ctx.tiu_bf16_lookup_interp_table(&p5);

      ctx.tdma_store(tl_lut_out, ga_output + out_offset);
      ctx.lmem_free_tensor(tl_lut_out);
      ctx.lmem_free_tensor(tl_buf);
      ctx.lmem_free_tensor(tl_output);
      ctx.lmem_free_tensor(tl_input);
    } // end for
  } // end else
  // free table and template
  ctx.lmem_free_tensor(tl_template);
  ctx.lmem_free_tensor(tl_lut_mantissa);
  ctx.lmem_free_tensor(tl_lut);
} // end fun