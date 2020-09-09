/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_eltwise.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_eltwise_add"

void cvi_backend_tl_eltwise_op(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output, laddr_t la_working,
    gaddr_t ga_input, gaddr_t ga_output, gaddr_t ga_addend,
    int op_code, int n, int c, int h, int w, bool do_relu,
    bool do_early_stride, int stride_h, int stride_w,
    int8_t rshift, int8_t m_i8_input, int8_t m_i8_addend, int32_t i32Multiplier,
    bool do_load, bool do_store) {
  // work space size
  // 2 for 16bit working_l and working_h
  // 2 for load addend double buffer
  #define WORKING_NUM (4)
  int oh = (do_early_stride) ? h / stride_h : h;
  int ow = (do_early_stride) ? w / stride_w : w;
  uint32_t input_size = ctx.lmem_tensor_to_size({(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w}, CVK_FMT_I8, /*eu_align=*/1);
  uint32_t output_size = ctx.lmem_tensor_to_size({(uint32_t)n, (uint32_t)c, (uint32_t)oh, (uint32_t)ow}, CVK_FMT_I8, /*eu_align=*/1);
  uint32_t working_size = LOCAL_MEM_SIZE - input_size - output_size;
  uint32_t inputTileShapeC, inputTileShapeH, inputTileShapeW, reshapeInputHSize;

  // Setup tiling shape
  if (do_early_stride) {
    assert(h % stride_h == 0);
    assert(w % stride_w == 0);

    inputTileShapeH = h;
    reshapeInputHSize = h;
    inputTileShapeW = w;
    inputTileShapeC = NPU_NUM * (uint32_t)(working_size / WORKING_NUM / align_up(h * w, EU_NUM));
  } else {
    assert(stride_h == 1);
    assert(stride_w == 1);

    if (align_up(h * w, EU_NUM) * WORKING_NUM > working_size) {
      //cut tileShape (1, npu_num, X, eu_num)
      //Regard (1, c, h, w) -> (1, c, ceil(h*w/eu_number), eu_num)
      inputTileShapeC = NPU_NUM;
      // note inputTileShapeH is not spatial h anymore, it is against w = EU_NUM
      // i.e. each step is inputTileShapeH * EU_NUM
      inputTileShapeH = working_size / WORKING_NUM / EU_NUM;
      reshapeInputHSize = (h * w + EU_NUM - 1) / EU_NUM;
      inputTileShapeW = EU_NUM;
    } else {
      //split h*w (whole 2d shape)
      inputTileShapeH = h;
      reshapeInputHSize = h;
      inputTileShapeW = w;
      inputTileShapeC = NPU_NUM * (uint32_t)(working_size / WORKING_NUM / align_up(h * w, EU_NUM));
    }
  }
  if (inputTileShapeC > MAX_TIU_CHL) {
    inputTileShapeC = align_up(MAX_TIU_CHL / 2, NPU_NUM);
  }

  LLVM_DEBUG(
    llvm::errs() << "    eltwise    : nchw = ("
                 << n << "," << c << "," << h << "," << w << ")"
                 << ", working_size " << working_size
                 << "\n                     "
                 << "inputTileShapeC " << inputTileShapeC
                 << ", inputTileShapeH " << inputTileShapeH
                 << ", reshapeInputHSize " << reshapeInputHSize
                 << ", inputTileShapeW " << inputTileShapeW
                 << "\n                     "
                 << ", EARLY_STRIDE " << do_early_stride
                 << ", LD " << do_load
                 << ", ST " << do_store
                 << "\n                     "
                 << "la_i = " << la_input
                 << ", la_o = " << la_output
                 << ", la_w = " << la_working
                 << "\n                     "
                 << "ga_i = " << ga_input
                 << ", ga_o = " << ga_output
                 << ", ga_a = " << ga_addend
                 << "\n                     "
                 << ", i32Multiplier = " << i32Multiplier
                 << "\n"
                 ;
  );

  bool prealloc = false;
  laddr_t la_working_h, la_working_l, la_addend[2];
  if (la_input != LA_INVALID) {
    prealloc = true;
    assert(la_output != LA_INVALID && la_working != LA_INVALID);
    laddr_t la_max = std::max(la_input, la_output);
    assert(la_working >= input_size);
    assert(la_max + output_size <= (uint32_t)LOCAL_MEM_SIZE);
    uint32_t one_working_size
        = (inputTileShapeC / NPU_NUM) * align_up((inputTileShapeH / stride_h) * (inputTileShapeW / stride_w), EU_NUM);
    uint32_t input_working_size
        = (inputTileShapeC / NPU_NUM) * align_up((inputTileShapeH)* (inputTileShapeW), EU_NUM);
    assert(one_working_size * WORKING_NUM <= (la_max - la_working));
    la_working_l = la_working;
    la_working_h = la_working_l + one_working_size;
    la_addend[0] = la_working_h + one_working_size;
    la_addend[1] = la_addend[0] + input_working_size;
  }

  // alloc lmem_t
  cvk_tl_t *tl_input = nullptr;
  cvk_tl_t *tl_output = nullptr;
  cvk_tl_t *tl_working_h = nullptr;
  cvk_tl_t *tl_working_l = nullptr;
  cvk_tl_t *tl_addend[2] = {nullptr, nullptr};

  if (prealloc) {
    tl_input = new cvk_tl_t;
    tl_output = new cvk_tl_t;
    tl_working_l = new cvk_tl_t;
    tl_working_h = new cvk_tl_t;
    tl_addend[0] = new cvk_tl_t;
    tl_addend[1] = new cvk_tl_t;

    tl_input->start_address = la_input;
    tl_output->start_address = la_output;
    tl_working_l->start_address = la_working_l;
    tl_working_h->start_address = la_working_h;
    tl_addend[0]->start_address = la_addend[0];
    tl_addend[1]->start_address = la_addend[1];
  } else {
    tl_input = ctx.lmem_alloc_tensor(ctx.shape_t4(n, c, h, w), CVK_FMT_I8, /*eu_align=*/1);
    tl_output = ctx.lmem_alloc_tensor(ctx.shape_t4(n, c, oh, ow), CVK_FMT_I8, /*eu_align=*/1);
    tl_working_l = ctx.lmem_alloc_tensor(ctx.shape_t4(1, inputTileShapeC, inputTileShapeH / stride_h, inputTileShapeW / stride_w), CVK_FMT_I8, /*eu_align=*/1);
    tl_working_h = ctx.lmem_alloc_tensor(ctx.shape_t4(1, inputTileShapeC, inputTileShapeH / stride_h, inputTileShapeW / stride_w), CVK_FMT_I8, /*eu_align=*/1);
    tl_addend[0] = ctx.lmem_alloc_tensor(ctx.shape_t4(1, inputTileShapeC, inputTileShapeH, inputTileShapeW), CVK_FMT_I8, /*eu_align=*/1);
    tl_addend[1] = ctx.lmem_alloc_tensor(ctx.shape_t4(1, inputTileShapeC, inputTileShapeH, inputTileShapeW), CVK_FMT_I8, /*eu_align=*/1);
    assert(tl_addend[0] && tl_addend[1] && tl_input && tl_output);
  }

  // global memory stride from global memory shape
  cvk_tg_stride_t gstride = {(uint32_t)(c * h * w), (uint32_t)(h * w), (uint32_t)w}; //inputStride
  cvk_tg_stride_t gstride_output = {(uint32_t)(c * oh * ow), (uint32_t)(oh * ow), (uint32_t)ow}; //outputStride
  cvk_tl_stride_t default_stride = ctx.tl_default_stride(ctx.shape_t4(n, c, h, w), CVK_FMT_I8, 1);
  uint32_t n_stride = default_stride.n;

  ctx.parallel_disable();
  ctx.set_layer_id(layer_id);

  if (do_load) {
    // load input
    tl_input->fmt = CVK_FMT_I8;
    tl_input->shape = ctx.shape_t4(n, c, h, w);
    tl_input->stride = ctx.tl_default_stride(tl_input->shape, CVK_FMT_I8, /*eu_align=*/1);
    ctx.tdma_load_stride(tl_input, ga_input, gstride);
  }

  int flip = 0;
  for (uint32_t n_pos = 0; n_pos < (uint32_t)n; n_pos++) {
    for (uint32_t c_pos = 0; c_pos < (uint32_t)c; c_pos += inputTileShapeC) {
      uint32_t cur_c = std::min(c - c_pos, inputTileShapeC);
      for (uint32_t h_pos = 0; h_pos < reshapeInputHSize; h_pos += inputTileShapeH) {
        uint32_t cur_h = std::min(reshapeInputHSize - h_pos, inputTileShapeH);

        // load addend
        // assert(c_pos * h_pos == 0);  // at least one is zero
        uint64_t ga_addend_pos = ga_addend + n_pos * c * h * w
                            + c_pos * h * w + h_pos * inputTileShapeW;
        cvk_tl_t tl_addend_tmp;
        tl_addend_tmp.start_address = tl_addend[flip]->start_address;
        tl_addend_tmp.fmt = CVK_FMT_I8;
        tl_addend_tmp.shape = ctx.shape_t4(1, cur_c, cur_h, inputTileShapeW);
        tl_addend_tmp.stride = ctx.tl_default_stride(tl_addend_tmp.shape, CVK_FMT_I8, /*eu_align=*/1);
        gstride = {(uint32_t)(c * h * w), (uint32_t)(h * w), (uint32_t)inputTileShapeW}; //cstride/nstride is same as original
        ctx.tdma_load_stride(&tl_addend_tmp, ga_addend_pos, gstride);

        ctx.parallel_disable();
        ctx.parallel_enable();

        // form input tl tensor on position
        cvk_tl_t tl_input_pos;
        tl_input_pos.start_address = tl_input->start_address
                                  + n_pos * n_stride
                                  + (c_pos / NPU_NUM) * align_up(h * w, EU_NUM)
                                  + h_pos * inputTileShapeW;
        tl_input_pos.fmt = CVK_FMT_I8;
        tl_input_pos.shape = ctx.shape_t4(1, cur_c, cur_h, inputTileShapeW);
        tl_input_pos.stride = ctx.tl_default_stride(tl_input_pos.shape, CVK_FMT_I8, /*eu_align=*/1);

        // form working tl tensor with shape
        cvk_tl_t tl_working_h_tmp;
        tl_working_h_tmp.start_address = tl_working_h->start_address;
        tl_working_h_tmp.fmt = CVK_FMT_I8;
        tl_working_h_tmp.shape = ctx.shape_t4(1, cur_c, cur_h / stride_h, inputTileShapeW / stride_w);
        tl_working_h_tmp.stride = ctx.tl_default_stride(tl_working_h_tmp.shape, CVK_FMT_I8, /*eu_align=*/1);
        cvk_tl_t tl_working_l_tmp;
        tl_working_l_tmp.start_address = tl_working_l->start_address;
        tl_working_l_tmp.fmt = CVK_FMT_I8;
        tl_working_l_tmp.shape = ctx.shape_t4(1, cur_c, cur_h / stride_h, inputTileShapeW / stride_w);
        tl_working_l_tmp.stride = ctx.tl_default_stride(tl_working_l_tmp.shape, CVK_FMT_I8, /*eu_align=*/1);

        // form wl with stride
        cvk_tl_t tl_input_pos_with_stride;
        tl_input_pos_with_stride.fmt = CVK_FMT_I8;
        tl_input_pos_with_stride.start_address = tl_input_pos.start_address;
        tl_input_pos_with_stride.shape = ctx.shape_t4(1, cur_c, cur_h / stride_h, inputTileShapeW / stride_w);
        // stride
        tl_input_pos_with_stride.stride = {
            tl_input_pos.stride.n,
            tl_input_pos.stride.c,
            (uint32_t)(stride_h * inputTileShapeW),
            (uint32_t)stride_w
        };

        cvk_tl_t tl_addend_tmp_with_stride;
        tl_addend_tmp_with_stride.fmt = CVK_FMT_I8;
        tl_addend_tmp_with_stride.start_address = tl_addend_tmp.start_address;
        tl_addend_tmp_with_stride.shape = ctx.shape_t4(1, cur_c, cur_h / stride_h, inputTileShapeW / stride_w);
        tl_addend_tmp_with_stride.stride = {
            tl_addend_tmp.stride.n,
            tl_addend_tmp.stride.c,
            (uint32_t)(stride_h * inputTileShapeW),
            (uint32_t)stride_w
        };

        switch (op_code) {
          case 0: {  // production
            cvk_tiu_mul_qm_param_t p1 = {0};
            p1.res_high = nullptr;
            p1.res_low = &tl_working_l_tmp;
            p1.a = &tl_input_pos_with_stride;
            p1.b_is_const = 0;
            p1.b = &tl_addend_tmp_with_stride;
            p1.rshift_bits = rshift;
            p1.relu_enable = do_relu;
            p1.layer_id = layer_id;
            p1.multiplier = i32Multiplier;
            ctx.tiu_mul_qm(&p1);
            break;
          }
          case 1: {  //Add
            // do mul on input, read with stride if stride is present
            cvk_tiu_mul_param_t p1 = {0};
            p1.res_high = &tl_working_h_tmp;
            p1.res_low = &tl_working_l_tmp;
            p1.a = &tl_input_pos_with_stride;
            p1.b_const.val = m_i8_input;
            p1.b_const.is_signed = true;
            p1.b_is_const = true;
            p1.rshift_bits = 0;
            p1.layer_id = layer_id;
            p1.relu_enable = 0;
            ctx.tiu_mul(&p1);

            // do mac (mul on addend and add previous result)
            cvk_tiu_mac_param_t p2 = {0};
            p2.res_high = &tl_working_h_tmp;
            p2.res_low = &tl_working_l_tmp;
            p2.a = &tl_addend_tmp_with_stride;
            p2.res_is_int8 = true;
            p2.b_const.val = m_i8_addend;
            p2.b_is_const = 1;
            p2.b_const.is_signed = true;
            p2.lshift_bits = 0;
            p2.rshift_bits = rshift;
            p2.layer_id = layer_id;
            p2.relu_enable = do_relu;
            ctx.tiu_mac(&p2);
            break;
          }
          case 2: { //max
            //todo
            assert(0);
            break;
          }
          default :{//max
            //not support
            assert(0);
            break;
          }
        }

        // form output tl tensor on position
        cvk_tl_t tl_output_pos;
        tl_output_pos.start_address = tl_output->start_address
                                  + n_pos * n_stride
                                  + (c_pos / NPU_NUM) * align_up(oh * ow, EU_NUM)
                                  + h_pos * inputTileShapeW / stride_w;
        tl_output_pos.fmt = CVK_FMT_I8;
        tl_output_pos.shape = ctx.shape_t4(1, cur_c, cur_h/stride_h, inputTileShapeW/stride_w);
        tl_output_pos.stride = ctx.tl_default_stride(tl_output_pos.shape, CVK_FMT_I8, /*eu_align=*/1);

        // copy to output
        // IF runtime can optimize, this op can be removed
        // Just set mac res = output_addr
        // But output_addr need low + high part now
        cvk_tiu_copy_param_t p3 = {0};
        p3.dst = &tl_output_pos;
        p3.src = &tl_working_l_tmp;
        p3.layer_id = layer_id;
        ctx.tiu_copy(&p3);

        flip = 1 - flip;
      }
    }
  }

  ctx.parallel_disable();

  if (do_store) {
    // store output
    tl_output->fmt = CVK_FMT_I8;
    tl_output->shape = ctx.shape_t4(n, c, oh, ow);
    tl_output->stride = ctx.tl_default_stride(tl_output->shape, CVK_FMT_I8, /*eu_align=*/1);
    ctx.tdma_store_stride(tl_output, ga_output, gstride_output);
  }

  //
  // Release resource in reverse order
  //
  if (prealloc) {
    delete tl_addend[1];
    delete tl_addend[0];
    delete tl_working_h;
    delete tl_working_l;
    delete tl_output;
    delete tl_input;
  } else {
    ctx.lmem_free_tensor(tl_addend[1]);
    ctx.lmem_free_tensor(tl_addend[0]);
    ctx.lmem_free_tensor(tl_working_h);
    ctx.lmem_free_tensor(tl_working_l);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
}


void cvi_backend_tl_eltwise(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t *la_input, laddr_t la_output, laddr_t la_working,
    int input_n, int input_c, int input_h, int input_w, int input_size,
    int op,
    int8_t rshift, const int8_t *m_i8,
    bool use_default_coeff,
    bool do_relu, float relu_slope, const int *coeffs, const int i32Multiplier,
    bool do_early_stride, int stride_h, int stride_w) {

  LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tl_eltwise:\n"
                                "  layer_id %d\n"
                                "  in(%d, %d, %d, %d), intput_size %d\n"
                                "  use_default_coeff %d, op %d, do_relu %d, relu_slop %.2f\n"
                                "  do_early_stride %d, stride_h %d, stride_w %d",
                                layer_id, input_n, input_c, input_h, input_w, input_size,
                                use_default_coeff, op, do_relu, relu_slope, do_early_stride,
                                stride_h, stride_w));

  int oh = (do_early_stride) ? input_h / stride_h : input_h;
  int ow = (do_early_stride) ? input_w / stride_w : input_w;

  cvk_tl_shape_t origin_shape = {
      static_cast<uint32_t>(input_n),
      static_cast<uint32_t>(input_c),
      static_cast<uint32_t>(input_h),
      static_cast<uint32_t>(input_w)};
  cvk_tl_stride_t bottom_stride = ctx.tl_default_stride(origin_shape, CVK_FMT_I8, 1);
  if (do_early_stride) {
    bottom_stride = {bottom_stride.n, bottom_stride.c , (uint32_t)(input_w * stride_h), (uint32_t)stride_w};
  }

  cvk_tl_shape_t shape = {
      static_cast<uint32_t>(input_n),
      static_cast<uint32_t>(input_c),
      static_cast<uint32_t>(oh),
      static_cast<uint32_t>(ow)};
  cvk_tl_stride_t top_stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);

  bool fused_relu = (do_relu && (relu_slope == 0.0f)) ? true : false;

  auto ic_step = input_c;
  if (input_c > MAX_TIU_CHL) {
    int i = 2;
    do {
      ic_step = align_up(ceiling_func(input_c, i++), NPU_NUM * EU_NUM);
    } while (ic_step > MAX_TIU_CHL);

    llvm::errs() << "tl_eltwise input_c(" << input_c
                  << ") is larger than " << MAX_TIU_CHL
                  << ", need to split it with step "
                  << ic_step << "\n";
  }

  for (int32_t ic_pos = 0; ic_pos < input_c; ic_pos += ic_step) {
    auto cur_ic = std::min(ic_step, input_c - ic_pos);
    cvk_tl_t input[2];
    cvk_tl_t output;

    shape.c = cur_ic;
    input[0].start_address = la_input[0] + (ic_pos / NPU_NUM) * bottom_stride.c;
    input[0].fmt = CVK_FMT_I8;
    input[0].shape = shape;
    input[0].stride = bottom_stride;

    output.start_address = la_output + (ic_pos / NPU_NUM) * top_stride.c;
    output.fmt = CVK_FMT_I8;
    output.shape = shape;
    output.stride = top_stride;

    switch (op) {
      case 0: {  // production
        assert(input_size == 2 && "Support only input_size = 2");
        assert(i32Multiplier != 0);
        input[1].start_address = la_input[1] + (ic_pos / NPU_NUM) * bottom_stride.c;
        input[1].fmt = CVK_FMT_I8;
        input[1].shape = shape;
        input[1].stride = bottom_stride;

        cvk_tiu_mul_qm_param_t p1 = {0};
        p1.res_high = nullptr;
        p1.res_low = &output;
        p1.a = &input[0];
        p1.b_is_const = 0;
        p1.b = &input[1];
        p1.rshift_bits = rshift;
        p1.relu_enable = 0;
        p1.layer_id = layer_id;
        p1.multiplier = i32Multiplier;
        ctx.tiu_mul_qm(&p1);
        break;
      }
      case 1: {  // sum
        cvk_tl_t working;
        working.start_address = la_working + (ic_pos / NPU_NUM) * top_stride.c;
        working.fmt = CVK_FMT_I8;
        working.shape = shape;
        working.stride = top_stride;

        cvk_tl_t *res_high = &working;
        cvk_tl_t *res_low = &output;
        bool out_is_higher_addr = false;
        if (la_output > la_working) {
          out_is_higher_addr = true;
          res_high = &output;
          res_low = &working;
        }
        // res_high->stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);

        if (use_default_coeff) {
          cvk_tiu_mul_param_t p = {0};
          p.res_high = res_high;
          p.res_low = res_low;
          p.a = &input[0];
          p.b_const.val = m_i8[0] * coeffs[0];
          p.b_const.is_signed = true;
          p.b_is_const = true;
          p.rshift_bits = 0;
          p.layer_id = layer_id;
          p.relu_enable = 0;
          ctx.tiu_mul(&p);

          for (int i = 1; i < input_size - 1; ++i) {
            input[1].start_address = la_input[i] + (ic_pos / NPU_NUM) * bottom_stride.c;
            input[1].fmt = CVK_FMT_I8;
            input[1].shape = shape;
            input[1].stride = bottom_stride;  // EU-aligned

            cvk_tiu_mac_param_t p3 = {0};
            p3.res_high = res_high;
            p3.res_low = res_low;
            p3.a = &input[1];
            p3.res_is_int8 = false;
            p3.b_const.val = m_i8[i] * coeffs[i];
            p3.b_is_const = 1;
            p3.b_const.is_signed = true;
            p3.lshift_bits = 0;
            p3.rshift_bits = 0;
            p3.layer_id = layer_id;
            p3.relu_enable = 0;
            ctx.tiu_mac(&p3);
          }
          input[1].start_address = la_input[input_size - 1] + (ic_pos / NPU_NUM) * bottom_stride.c;
          input[1].fmt = CVK_FMT_I8;
          input[1].shape = shape;
          input[1].stride = bottom_stride;  // EU-aligned

          cvk_tiu_mac_param_t p3 = {0};
          p3.res_high = res_high;
          p3.res_low = res_low;
          p3.a = &input[1];
          p3.res_is_int8 = true;
          p3.b_const.val = m_i8[input_size - 1] * coeffs[input_size - 1];
          p3.b_is_const = 1;
          p3.b_const.is_signed = true;
          p3.lshift_bits = 0;
          p3.rshift_bits = rshift;
          p3.layer_id = layer_id;
          p3.relu_enable = fused_relu;
          ctx.tiu_mac(&p3);

          if (out_is_higher_addr) {
            ctx.parallel_disable();
            cvk_tdma_l2l_tensor_copy_param_t p10 = {0};
            p10.dst = &output;
            p10.src = res_low;
            p10.layer_id = layer_id;
            ctx.tdma_l2l_tensor_copy(&p10);
          }
        } else {
          // Not support
          assert(0);
        }

        break;
      }
      case 2: {  // max
        // Not support
        assert(0);
        break;
      }
    }
  }
  if (do_relu && !fused_relu) {
    // Not support
    assert(0);
  }
}
