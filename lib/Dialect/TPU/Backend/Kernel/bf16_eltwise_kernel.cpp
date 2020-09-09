/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_elewise_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880v2_eltwise"

#define ASSERT(x) assert(x)
#define ELTWISE_PROD (0)  // production
#define ELTWISE_SUM  (1)  // sum
#define ELTWISE_MAX  (2)  // max
#define ELTWISE_MIN  (3)  // min
#define ELTWISE_MAX_MIN  (4)  // min max for clip

//namespace bmnet {

static void elt_one_step(const CviBackendContext &ctx, uint32_t layer_id, int op, gaddr_t ga_input[],
                             gaddr_t ga_output, int input_size, int input_n, int input_c,
                             int input_h, int input_w, int heightDividedNumber, int widthDividedNumber,
                             const float coeffs[], gaddr_t gaddr_offset, bool fused_relu) {
  cvk_tl_shape_t tl_inputShape = ctx.shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_shape_t tl_outputShape = ctx.shape_t4(input_n, input_c, input_h/heightDividedNumber, input_w/widthDividedNumber);
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_inputShape, CVK_FMT_BF16, /*eu_align=*/1);
  cvk_tl_t *tl_output = NULL;

  if (op == ELTWISE_MAX || op == ELTWISE_MIN || op == ELTWISE_MAX_MIN) {
    tl_output = tl_input;
  }
  else {
    tl_output = ctx.lmem_alloc_tensor(tl_outputShape, CVK_FMT_BF16, /*eu_align=*/1);
  }

  cvk_tl_t tl_working_input;
  tl_working_input.start_address = tl_input->start_address;
  tl_working_input.shape = tl_output->shape;
  tl_working_input.stride = {
      ALIGN((uint32_t)(input_h * input_w * sizeof(uint16_t)), 16),
      ALIGN((uint32_t)(input_h * input_w * sizeof(uint16_t)), 16),
      (uint32_t)(heightDividedNumber * input_w * sizeof(uint16_t)),
      (uint32_t)(widthDividedNumber * sizeof(uint16_t))
  };
  tl_working_input.fmt = tl_input->fmt;

  bool isStoreSameShape = (heightDividedNumber == 1) && (widthDividedNumber == 1);

  LLVM_DEBUG(llvm::errs() << "      elt_one_step: (" << input_n << ", " << input_c
                           << ", " << input_h << ", " << input_w << ") ->"
                           << "(" << input_n << ", " << input_c
                           << ", " << input_h/heightDividedNumber << ", " << input_w/widthDividedNumber << ")" << "\n";);
  if (tl_input == nullptr) {
    LLVM_DEBUG(llvm::errs() << "      unable to alloc tl_input\n";);
    ASSERT(tl_input);
  }


  if (op == ELTWISE_SUM) {
    if(!isStoreSameShape) { // Pack input number
      cvk_tl_t tl_temp0 = *tl_input;
      tl_temp0.start_address = tl_output->start_address;
      ctx.tdma_load_bf16(
          &tl_temp0,
          ga_input[0] + gaddr_offset * heightDividedNumber * widthDividedNumber);

      tl_temp0.start_address = tl_output->start_address;
      tl_temp0.shape = tl_outputShape;
      tl_temp0.stride = tl_working_input.stride;
      tl_temp0.fmt = tl_output->fmt;

      cvk_tl_t tl_temp1;
      tl_temp1.start_address = tl_output->start_address;
      tl_temp1.shape = tl_outputShape;
      tl_temp1.stride = tl_output->stride;
      tl_temp1.fmt = tl_output->fmt;

      cvk_tiu_copy_param_t p3 = {0};
      p3.src = &tl_temp0;
      p3.dst = &tl_temp1;
      p3.layer_id = layer_id;
      ctx.tiu_copy(&p3);
    } else {
      ctx.tdma_load_bf16(tl_output, ga_input[0] + gaddr_offset);
    }
    for (int i = 1; i < input_size; ++i) {
      ctx.tdma_load_bf16(
          tl_input,
          ga_input[i] + gaddr_offset * heightDividedNumber * widthDividedNumber);

      cvk_tiu_add_param_t p3 = {0};
      // p3.res_high = tl_output_h;
      // p3.res_low = tl_output;
      // p3.a = tl_input;
      p3.res_high = nullptr;
      p3.res_low = tl_output;
      p3.a_high = nullptr;
      p3.a_low = tl_output;
      p3.b_is_const = false;
      p3.b.high = nullptr;
      p3.b.low = &tl_working_input;
      p3.rshift_bits = 0;
      p3.layer_id = layer_id;
      p3.relu_enable = (i != input_size - 1) ? 0 : fused_relu;
      ctx.tiu_add(&p3);
    }
  }
  else if (op == ELTWISE_PROD) {
    ctx.tdma_load_bf16(tl_output, ga_input[1] + gaddr_offset);
    ctx.tdma_load_bf16(tl_input, ga_input[0] + gaddr_offset);

    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = tl_output;
    p.a = tl_input;
    p.b = tl_output;
    p.b_is_const = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = fused_relu;
    ctx.tiu_mul(&p);
  }
  else if (op == ELTWISE_MAX) {
    ctx.tdma_load_bf16(tl_input, ga_input[0] + gaddr_offset);
    cvk_tiu_min_param_t p7 = {0};
    p7.min = tl_input;
    p7.a = tl_input;
    p7.b_is_const = 1;
    p7.b_const.val = ctx.convert_fp32_to_bf16(coeffs[0]);
    p7.b_const.is_signed = 1;
    p7.layer_id = layer_id;

    ctx.tiu_min(&p7);
  }
  else if (op == ELTWISE_MIN) {
    ctx.tdma_load_bf16(tl_input, ga_input[0] + gaddr_offset);
    cvk_tiu_max_param_t p = {0};
    p.max = tl_input;
    p.a = tl_input;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(coeffs[0]);
    p.b_const.is_signed = 1;
    p.layer_id = layer_id;

    ctx.tiu_max(&p);
  }
  else if (op == ELTWISE_MAX_MIN) {
    ctx.tdma_load_bf16(tl_input, ga_input[0] + gaddr_offset);

    // ELTWISE_MAX
    cvk_tiu_min_param_t p7 = {0};
    p7.min = tl_input;
    p7.a = tl_input;
    p7.b_is_const = 1;
    p7.b_const.val = ctx.convert_fp32_to_bf16(coeffs[0]);
    p7.b_const.is_signed = 1;
    p7.layer_id = layer_id;

    ctx.tiu_min(&p7);

    // ELTWISE_MIN
    cvk_tiu_max_param_t p = {0};
    p.max = tl_input;
    p.a = tl_input;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(coeffs[1]);
    p.b_const.is_signed = 1;
    p.layer_id = layer_id;

    ctx.tiu_max(&p);
  }
  else {
    ASSERT(0 && "not support");
  }

  ctx.tdma_store_bf16(tl_output, ga_output + gaddr_offset);

  // Reverse order
  if (op == ELTWISE_MIN || op == ELTWISE_MAX || op == ELTWISE_MAX_MIN) {
    tl_output = NULL;
  }
  else {
    ctx.lmem_free_tensor(tl_output);
  }

  ctx.lmem_free_tensor(tl_input);
}

// Adopt the tiling style used in bmnet_arithmetic_fixed_forward_bmkernel.
// We should unify the tensor arithemetics.
static void _bf16_eltwise_forward_kernel(const CviBackendContext &ctx, uint32_t layer_id, int op, gaddr_t ga_input[],
                                            gaddr_t ga_output, int input_size, int input_n, int input_c,
                                            int input_h, int input_w,
                                            bool do_relu, float relu_slope,
                                            bool do_early_stride, int stride_h, int stride_w,
                                            const float coeffs[]) {
  int blob_num = 3;
  if (op == ELTWISE_PROD) {
    // 2 means input[0] input[1]
    // input[0] will overwrite by mul and we store back to input[0]
    blob_num = 2;
  }

  if (op == ELTWISE_MAX || op == ELTWISE_MIN || op == ELTWISE_MAX_MIN) {
    // we could overwrite myself
    blob_num = 1;
  }

  int output_h = input_h;
  int output_w = input_w;
  if (do_early_stride) {
    assert(input_h % stride_h == 0);
    assert(input_w % stride_w == 0);
    output_h = input_h / stride_h;
    output_w = input_w / stride_w;
  }

  int tensor_size = input_n * input_c * input_h * input_w;
  int total_lmem_size = LOCAL_MEM_SIZE * NPU_NUM;

  cvk_tl_shape_t tl_shape = ctx.shape_t4(input_n, input_c, input_h, input_w);
  int total_tiu_lmem_size = blob_num * ctx.lmem_tensor_to_size(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);

  LLVM_DEBUG(llvm::errs() << llvm::format("_bf16_eltwise_forward_kernel:\n"
                               "Input    shape (%d, %d, %d, %d)\n"
                               "Output   shape (%d, %d, %d, %d)\n"
                               "    input_size %d, do_relu %d, relu_slope %f\n"
                               "    total_lmem_size %d, total_tiu_lmem_size %d\n",
                               input_n, input_c, input_h, input_w,
                               input_n, input_c, output_h, output_w,
                               input_size, do_relu, relu_slope,
                               total_lmem_size, total_tiu_lmem_size););

  gaddr_t gaddr_offset = 0;
  int remain_size = tensor_size;

  bool fused_relu = (do_relu && (relu_slope == 0.0f)) ? true : false;

  bool isStoreSameShape = (input_h == output_h) && (input_w == output_w);
  int heightDividedNumber = input_h / output_h;
  int widthDividedNumber = input_w / output_w;

  if(isStoreSameShape) {
    // Use all lanes
    if (remain_size >= (NPU_NUM * EU_NUM)) {
      do {
        // Find height
        int height = remain_size / (NPU_NUM * EU_NUM);
        do {
          cvk_tl_shape_t tmp_shape = ctx.shape_t4(1, NPU_NUM, height, EU_NUM);
          int required_size = blob_num * ctx.lmem_tensor_to_size(tmp_shape, CVK_FMT_BF16, /*eu_align=*/1);

          if (required_size <= LOCAL_MEM_SIZE)
            break;
        } while (--height);

        int step_size = height * NPU_NUM * EU_NUM;

        LLVM_DEBUG(llvm::errs() << llvm::format("    step_size %d, remain_size %d, gaddr_offset 0x%lx\n",
                                              step_size, remain_size, gaddr_offset););

        elt_one_step(ctx, layer_id, op, ga_input, ga_output, input_size, 1, NPU_NUM, height, EU_NUM,
                        1, 1, coeffs, gaddr_offset, fused_relu);

        // Update step
        remain_size -= step_size;
        gaddr_offset += step_size * sizeof(uint16_t);
      } while (remain_size >= ((NPU_NUM * EU_NUM)));
    }

    // Use one lane to handle remaining
    if (remain_size) {
      int step_size = remain_size;

      elt_one_step(ctx, layer_id, op, ga_input, ga_output, input_size, 1, 1, 1, step_size,
                      1, 1, coeffs, gaddr_offset, fused_relu);

      LLVM_DEBUG(llvm::errs() << llvm::format("    step_size %d, remain_size %d, gaddr_offset 0x%lx\n",
                                            step_size, remain_size, gaddr_offset););
      remain_size -= step_size;
      gaddr_offset += step_size * sizeof(uint16_t);
    }
  } else {
    if (remain_size >= (NPU_NUM * input_w * heightDividedNumber)) {
      do{
        // Find height
        int step_w = input_w;
        int step_h = heightDividedNumber;
        int step_c = NPU_NUM;

        do {
          cvk_tl_shape_t tmp_shape = ctx.shape_t4(1, step_c, step_h, step_w);
          int required_size = blob_num * ctx.lmem_tensor_to_size(tmp_shape, CVK_FMT_BF16, /*eu_align=*/1);

          // LLVM_DEBUG(llvm::errs() << llvm::format(" LOCAL_MEM_SIZE %d, required_size %d, gaddr_offset 0x%lx\n, "
          //                                       " step_c = %d, step_h = %d, step_w = %d, \n   ",
          //                                     LOCAL_MEM_SIZE, required_size, gaddr_offset, step_c, step_h, step_w));
          if(step_h == input_h) {
            break;
          }
          if (required_size >= LOCAL_MEM_SIZE) {
            step_h -= heightDividedNumber;
            break;
          }
          step_h += heightDividedNumber;
        } while (true);

        do {
          cvk_tl_shape_t tmp_shape = ctx.shape_t4(1, step_c, step_h, step_w);
          int required_size = blob_num * ctx.lmem_tensor_to_size(tmp_shape, CVK_FMT_BF16, /*eu_align=*/1);

          // LLVM_DEBUG(llvm::errs() << llvm::format(" LOCAL_MEM_SIZE %d, required_size %d, gaddr_offset 0x%lx\n, "
          //                                       " step_c = %d, step_h = %d, step_w = %d, \n   ",
          //                                     LOCAL_MEM_SIZE, required_size, gaddr_offset, step_c, step_h, step_w));
          if (required_size >= LOCAL_MEM_SIZE) {
            step_c-= NPU_NUM;
            break;
          }
          if(step_c == input_c) {
            break;
          }
          step_c += NPU_NUM;
          if(remain_size <= step_c * step_h * step_w) {
            break;
          }
        } while (true);


        assert(step_c != 0);
        assert(step_h != 0);
        int input_step_size = step_c * step_h * step_w;
        int output_step_size = step_c * (step_h / heightDividedNumber) * (step_w / widthDividedNumber);

        LLVM_DEBUG(llvm::errs() << llvm::format(" input_step_size %d, remain_size %d, gaddr_offset 0x%lx\n, "
                                                " step_c = %d, step_h = %d, step_w = %d, \n   ",
                                              input_step_size, remain_size, gaddr_offset, step_c, step_h, step_w));

          elt_one_step(ctx, layer_id, op, ga_input, ga_output, input_size, 1, step_c, step_h, step_w,
                          heightDividedNumber, widthDividedNumber, coeffs, gaddr_offset, fused_relu);

          // Update step
          remain_size -= input_step_size; //calculate with input
          gaddr_offset += output_step_size * sizeof(uint16_t);  //calculate with output
        } while (remain_size >= (NPU_NUM * input_w * heightDividedNumber));
    }

      // Use one lane to handle remaining
    if (remain_size) {
      int step_size = remain_size;
      int remain_height = ceil(remain_size / (NPU_NUM * input_w));


      elt_one_step(ctx, layer_id, op, ga_input, ga_output, input_size, 1, NPU_NUM, remain_height, input_w,
                      heightDividedNumber, widthDividedNumber, coeffs, gaddr_offset, fused_relu);

      LLVM_DEBUG(llvm::errs() << llvm::format("    step_size %d, remain_size %d, gaddr_offset 0x%lx\n",
                                            step_size, remain_size, gaddr_offset););
      remain_size -= step_size;
      gaddr_offset += step_size * sizeof(uint16_t);
    }
  }
}

void bf16_eltwise_forward_kernel(const CviBackendContext &ctx,
                                 uint32_t layer_id, gaddr_t ga_input[], gaddr_t ga_output,
                                 int input_size, int op, int input_n, int input_c,
                                 int input_h, int input_w,
                                 bool do_relu, float relu_slope,
                                 bool do_early_stride, int stride_h, int stride_w,
                                 const float coeffs[]) {
  LLVM_DEBUG(
      llvm::errs() << llvm::format("bf16_eltwise_forward_kernel\n"
                               "    shape (%d, %d, %d, %d)\n"
                               "    input_size %d, do_relu %d, relu_slope %f\n"
                               "    ga_input ",
                               input_n, input_c, input_h, input_w,
                               input_size, do_relu, relu_slope);
      for (int i = 0; i < input_size; ++i) {
        llvm::errs() << llvm::format("0x%lx, ", ga_input[i]);
      }
      llvm::errs() << llvm::format("ga_output 0x%lx\n", ga_output);
  );
  // Keep tiu/tdma in order
  ctx.parallel_disable();

  switch (op) {
    case ELTWISE_MAX:  // max
    case ELTWISE_MIN:  // min
    case ELTWISE_MAX_MIN:
      // FIXME: support tensor compare
      assert(input_size == 1 && "currently ONLY support max/min with const");
      assert(do_relu == 0 && "currenlly not support relu");
    case ELTWISE_PROD:  // production
    case ELTWISE_SUM:  // sum
      _bf16_eltwise_forward_kernel(ctx, layer_id, op, ga_input, ga_output, input_size, input_n, input_c,
                                   input_h, input_w, do_relu, relu_slope,
                                   do_early_stride, stride_h, stride_w, coeffs);
      break;
  }

}
