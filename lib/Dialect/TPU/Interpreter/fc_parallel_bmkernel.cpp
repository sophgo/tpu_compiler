/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
/*
 * Copyright Bitmain Technofcies Inc.
 * Written by:
 *   Mingkang Qin <mingkang.qin@bitmain.com>
 */
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>
//#include <support/Debug.h>
//#include <support/Format.h>
//#include <targets/plat-bm188x/bmkernel/bmkernel_api.h>
//#include <builder/CommandCategory.hpp>
//#include <targets/Target.hpp>
//#include <targets/plat-bm188x/BM1880v2BackendContext.hpp>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_fc"

#define DEBUG_BMNET(x) LLVM_DEBUG(x)
#define LOG(x) llvm::errs()
#define ASSERT(x) assert(x)

// TODO(wwcai): to remove
#if 1
#define CHIP_VERSION (ctx.hw.chip_version)
#define NODECHIP_SHIFT (ctx.hw.nodechip_shift)
#define NPU_SHIFT (ctx.hw.npu_shift)
#define EU_SHIFT (ctx.hw.eu_shift)
#define LOCAL_MEM_ADDRWIDTH (ctx.hw.local_mem_shift)
#define LOCAL_MEM_BANKS (ctx.hw.local_mem_banks)
#define GLOBAL_MEM_SIZE (ctx.hw.global_mem_size)
#define CHIP_IS_BM1680 (CHIP_VERSION == BM_CHIP_BM1680)
#define CHIP_IS_BM1682 (CHIP_VERSION == BM_CHIP_BM1682)
#define NODECHIP_NUM (1 << NODECHIP_SHIFT)
#define NODECHIP_MASK (NODECHIP_NUM - 1)
#define NPU_NUM (1 << NPU_SHIFT)
#define NPU_MASK (NPU_NUM - 1)
#define EU_NUM (1 << EU_SHIFT)
#define EU_MASK (EU_NUM - 1)
#define LOCAL_MEM_SIZE (1 << LOCAL_MEM_ADDRWIDTH)
#endif

#define RELU   1
#define PRELU  2

static int get_csize_local(const BackendContext &ctx, int h, int w)
{
  int unit_size = ctx.hw.unit_size;
  ASSERT(unit_size > 0);
  return ALIGN(h * w, ctx.hw.eu_num) * unit_size;
}

#if 0
#include "llvm/Support/CommandLine.h"
namespace cl = llvm::cl;
static cl::opt<bool> OptUseLessLossFC("enable-less-loss-fc", cl::desc("enable less-loss FC"),
                                      cl::init(false), cl::cat(CatBM188X));

static cl::opt<bool> OptForceLessLossFC("force-less-loss-fc", cl::desc("force less-loss FC"),
                                        cl::init(false), cl::cat(CatBM188X));
#endif
//#define USE_MATRIX_W16

//namespace bmnet {

static inline int get_max(int a, int b) { return a > b ? a : b; }

static inline int idiv_round(int pNumerator, int pDenominator) {
  return (pNumerator + pDenominator - 1) / pDenominator;
}

static gaddr_t get_slice_global_offset(gaddr_t global_offset, int row_pos, int col_pos, int row_num,
                                       int col_num, int row_slice_num, int col_slice_num) {
  gaddr_t slice_offset_row = 0;
  if (row_pos < (row_num % row_slice_num)) {
    slice_offset_row = row_pos * (row_num / row_slice_num + 1);
  } else {
    slice_offset_row = (row_num % row_slice_num) * (row_num / row_slice_num + 1) +
                       (row_pos - (row_num % row_slice_num)) * (row_num / row_slice_num);
  }

  gaddr_t slice_offset_col = 0;
  if (col_pos < (col_num % col_slice_num)) {
    slice_offset_col = col_pos * (col_num / col_slice_num + 1);
  } else {
    slice_offset_col = (col_num % col_slice_num) * (col_num / col_slice_num + 1) +
                       (col_pos - (col_num % col_slice_num)) * (col_num / col_slice_num);
  }

  gaddr_t slice_offset = (slice_offset_col + slice_offset_row * col_num) * INT8_SIZE;

  return (global_offset + slice_offset);
}

//
// Shape/stride used in TDMA may not the same as in TIU.
// Adjust shape/stride for TIU.
//
// E.g.
//   Y(0, 4) = L(1, 256) * R(256, 4) + B(1, 4)
//
//   TDMA:
//      L(0, 16, 1, 16)
//      R(255, 1, 1, 4)
//      B(0, 1, 1, 4)
//
//   TIU:
//       Y res0(1, 1, 1, 16)
//       L opd0(1, 16, 1, 16)
//       R opd1(256, 1, 1, 16)
//       B opd2(1, 1, 1, 16)
//
static void matrix_multiplication(const BM1880v2BackendContext &ctx,
                                  bmk1880v2_tiu_matrix_multiplication_param_t &p) {
  // No need to adjust shape/stride
  if (p.res->shape.w >= ctx.hw.eu_num) {
    DEBUG_BMNET(llvm::errs() << llvm::format("  L(%d, %d), R(%d, %d)\n", p.left->shape.n,
                                          p.left->shape.col, p.right->shape.n, p.right->shape.col));
    ctx.tiu_matrix_multiplication(&p);

    return;
  }

  //
  // New shape/stride to align EU_NUM
  // adjust w as EU_NUM
  //
  bmk1880v2_matrix_lmem_t tl_res;
  tl_res.start_address = p.res->start_address;
  tl_res.fmt = p.res->fmt;
  tl_res.shape = {p.res->shape.n, p.res->shape.c, static_cast<u32>(ctx.hw.eu_num),
                  p.res->shape.col};
  tl_res.stride = ctx.matrix_lmem_default_stride(tl_res.shape);

  bmk1880v2_matrix_lmem_t tl_right;
  tl_right.start_address = p.right->start_address;
  tl_right.fmt = p.right->fmt;
  tl_right.shape = {p.right->shape.n, p.right->shape.c, static_cast<u32>(ctx.hw.eu_num),
                    p.right->shape.col};
  tl_right.stride = ctx.matrix_lmem_default_stride(tl_right.shape);

  bmk1880v2_matrix_lmem_t tl_bias = {0};
  if (p.bias) {
    tl_bias.start_address = p.bias->start_address;
    tl_bias.fmt = p.bias->fmt;
    tl_bias.shape = {p.bias->shape.n, p.bias->shape.c, static_cast<u32>(ctx.hw.eu_num),
                     p.bias->shape.col};
    tl_bias.stride = ctx.matrix_lmem_default_stride(tl_bias.shape);
  }

  bmk1880v2_tiu_matrix_multiplication_param_t p2 = p;
  p2.res = &tl_res;
  p2.left = p.left;
  p2.right = &tl_right;
  p2.bias = p.bias ? &tl_bias : nullptr;

  DEBUG_BMNET(llvm::errs() << llvm::format("  Modified L(%d, %d), R(%d, %d)\n", p2.left->shape.n,
                                        p2.left->shape.col, p2.right->shape.n,
                                        p2.right->shape.col));

  ctx.tiu_matrix_multiplication(&p2);
}

static void tg_prelu(const BM1880v2BackendContext &ctx, gaddr_t global_offset_top_data,
                     gaddr_t activation_ga_slope, int input_row_num, int weight_col_num,
                     int activation_channel_shared, int activation_gt_scale,
                     int activation_gt_rshift, int activation_le_scale, int activation_le_rshift) {
  ASSERT(0);
#if 0
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

  shape_t prelu_shape = shape_t4(1, weight_col_num, 1, 1);
  shape_t data_shape = shape_t4(input_row_num, weight_col_num, 1, 1);

  tensor_lmem *tl_fc_data = ctx.tl_prealloc_align(3 * bank_size, data_shape, FMT_I8);
  tensor_lmem *tl_slope = ctx.tl_prealloc_align(4 * bank_size, prelu_shape, FMT_I8);

  // load fc_reslut from global memory
  ctx.gdma_load(tl_fc_data, global_offset_top_data, CTRL_NEURON);
  ctx.gdma_load(tl_slope, activation_ga_slope, CTRL_WEIGHT);

  tensor_lmem *relu = ctx.tl_prealloc_align(5 * bank_size, data_shape, FMT_I8);
  tensor_lmem *neg = ctx.tl_prealloc_align(6 * bank_size, data_shape, FMT_I8);
  tensor_lmem *zero = ctx.tl_prealloc_align(7 * bank_size, data_shape, FMT_I8);
  ctx.tpu_zero(zero);
  // 0. relu = relu(tl_fc_data_int8)
  // 1. relu = (relu * gt_scale) >> gt_rshift
  // 2. neg = neg(0, botom)
  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
  // 4. tl_fc_data = or relu, neg
  bmk1880v2_relu_param_t p13;
  p13.ofmap = relu;
  p13.ifmap = tl_fc_data;
  ctx.tpu_relu(&p13);
  bmk1880v2_mul_const_param_t p;
  p.res_high = nullptr;
  p.res_low = relu;
  p.a = relu;
  p.b = activation_gt_scale;
  p.b_is_signed = true;
  p.rshift_width = activation_gt_rshift;
  ctx.tpu_mul_const(&p);

  bmk1880v2_min_param_t p6;
  p6.min = neg;
  p6.a = zero;
  p6.b = tl_fc_data;
  ctx.tpu_min(&p6);
  bmk1880v2_depthwise_param_t param;
  param.ofmap = neg;
  param.ifmap = neg;
  param.weight = tl_slope;
  param.bias = nullptr;
  param.ins_h = 0;
  param.ins_last_h = 0;
  param.ins_w = 0;
  param.ins_last_w = 0;
  param.pad_top = 0;
  param.pad_bottom = 0;
  param.pad_left = 0;
  param.pad_right = 0;
  param.stride_h = 1;
  param.stride_w = 1;
  param.rshift_width = activation_le_rshift;
  ctx.tpu_depthwise(&param);
  bmk1880v2_or_int8_param_t p9;
  p9.res = tl_fc_data;
  p9.a = relu;
  p9.b = neg;
  ctx.tpu_or_int8(&p9);
  ctx.gdma_store(tl_fc_data, global_offset_top_data, CTRL_NEURON);
  // free
  ctx.tl_free(zero);
  ctx.tl_free(neg);
  ctx.tl_free(relu);
  ctx.tl_free(tl_slope);
  ctx.tl_free(tl_fc_data);
#endif
}

static int method_selection_fc_forward_parallel(const BM1880v2BackendContext &ctx,
                                                // input parameters
                                                int input_row_num, int input_col_num,
                                                int weight_col_num, int W_param,
                                                int weight_bank_num,
                                                // output parameters
                                                int *slice_num) {
  int channel_size_local = get_csize_local(ctx, 1, W_param);
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  slice_num[0] = slice_num[1] = slice_num[2] = 1;

  DEBUG_BMNET(llvm::errs() << llvm::format("method_selection_fc_forward_parallel: \n"
                                        "    in(%d, %d) out(,%d), W_param %d, weight_bank_num %d\n",
                                        input_row_num, input_col_num, weight_col_num, W_param,
                                        weight_bank_num));

  int slicing_index = 2;
  int threshold_size[2];
  int SLICE_SIZE_THRESHOLD = (1 << ctx.hw.local_mem_shift) / (ctx.hw.local_mem_banks);
  threshold_size[0] = SLICE_SIZE_THRESHOLD;
  threshold_size[1] = bank_size;
  while (slicing_index-- && slice_num[0] + slice_num[1] + slice_num[2] == 3) {
    // input blob
    int C_param = (input_col_num + W_param - 1) / W_param;
    int input_local_size =
        input_row_num * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
    int input_slice_num =
        (input_local_size + threshold_size[slicing_index] - 1) / threshold_size[slicing_index];

    int slice_num_1_input = (input_slice_num > input_col_num) ? input_col_num : input_slice_num;

    // to slice_num_0_input, input_slice_num > input_row_num, should slice row, because maybe it is
    // not enough to just slice col to slice_num_0_input, input_slice_num < input_row_num, don't
    // need to slice, since it is enough to just slice col to slice_num_1_input, input_slice_num >
    // input_col_num, must slice part, and the celling is inpu_col_num
    int slice_num_0_input = (input_slice_num > input_row_num)
                                ? (input_slice_num + input_col_num - 1) / input_col_num
                                : 1;

    DEBUG_BMNET(llvm::errs() << "    slicing_index=" << slicing_index
                          << ", --slice_num_0_input=" << slice_num_0_input
                          << ", --slice_num_1_input=" << slice_num_1_input << "\n");

    // output blob
    C_param = (weight_col_num + W_param - 1) / W_param;

    int bias_local_size = 2 * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
    int output_local_size =
        2 * input_row_num * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
    int output_slice_num =
        (output_local_size + threshold_size[slicing_index] - 1) / threshold_size[slicing_index];

    int slice_num_0_output = (output_slice_num > input_row_num) ? input_row_num : output_slice_num;
    int slice_num_2_output = (output_slice_num > input_row_num)
                                 ? (output_slice_num + input_row_num - 1) / input_row_num
                                 : 1;

    DEBUG_BMNET(llvm::errs() << "    slicing_index=" << slicing_index
                          << ", --slice_num_0_output=" << slice_num_0_output
                          << ", --slice_num_2_output=" << slice_num_2_output << "\n");

    // weight blob and bias blob
    C_param = (weight_col_num + W_param - 1) / W_param;
    int weight_local_size =
        input_col_num * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
    int weight_threshold_size = (weight_bank_num * threshold_size[slicing_index] - bias_local_size);
    int weight_bias_slice_num =
        (weight_local_size + weight_threshold_size - 1) / weight_threshold_size;

    int slice_num_1_weight =
        (weight_bias_slice_num > input_col_num) ? input_col_num : weight_bias_slice_num;
    int slice_num_2_weight = (weight_bias_slice_num > input_col_num)
                                 ? (weight_bias_slice_num + input_col_num - 1) / input_col_num
                                 : 1;
    DEBUG_BMNET(llvm::errs() << "    slicing_index=" << slicing_index
                          << ", --slice_num_1_weight=" << slice_num_1_weight
                          << ", --slice_num_2_weight=" << slice_num_2_weight << "\n");

    // initialize slice numbers
    slice_num[0] = get_max(slice_num_0_output, slice_num_0_input);
    slice_num[1] = get_max(slice_num_1_weight, slice_num_1_input);
    slice_num[2] = get_max(slice_num_2_weight, slice_num_2_output);

    // actively slicing blobs to support parallelism
    if (!slicing_index) {
      if (slice_num[1] >= slice_num[2] && slice_num[1] >= slice_num[0]) {
        return 0;
      } else if (slice_num[0] >= slice_num[2] && slice_num[0] >= slice_num[1]) {
        return 1;
      } else {  // if(slice_num[2] >= slice_num[0] && slice_num[2] >= slice_num[1])
        return 2;
      }
    }
  }

  // fine-tuning
  int matrix_shape[3] = {1, 1, 1};
  while (true) {
    matrix_shape[0] = (input_row_num + slice_num[0] - 1) / slice_num[0];
    matrix_shape[1] = (input_col_num + slice_num[1] - 1) / slice_num[1];
    matrix_shape[2] = (weight_col_num + slice_num[2] - 1) / slice_num[2];

    int C_param_input_col = (matrix_shape[1] + W_param - 1) / W_param;
    int C_param_weight_col = (matrix_shape[2] + W_param - 1) / W_param;

    int weight_local_size =
        matrix_shape[1] * (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) * channel_size_local;
    int output_local_size = 2 * matrix_shape[0] *
                            (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) *
                            channel_size_local;
    int input_local_size =
        matrix_shape[0] * (ceiling_func_shift(C_param_input_col, NPU_SHIFT)) * channel_size_local;
    int bias_local_size =
        2 * (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) * channel_size_local;

    bool slicing_success = (input_local_size <= bank_size) && (output_local_size <= bank_size) &&
                           (weight_local_size + bias_local_size <= 5 * bank_size);
    if (slicing_success) {
      if (slice_num[0] + slice_num[1] == 2 || slice_num[0] + slice_num[2] == 2 ||
          slice_num[2] + slice_num[1] == 2) {
        DEBUG_BMNET(llvm::errs() << llvm::format("    weight_local_size: %d(%d * %d * %d)\n",
                                              weight_local_size, matrix_shape[1],
                                              (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)),
                                              channel_size_local));
        DEBUG_BMNET(llvm::errs() << llvm::format("    output_local_size: %d(%d * %d * %d)\n",
                                              output_local_size, 2 * matrix_shape[0],
                                              (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)),
                                              channel_size_local));
        DEBUG_BMNET(llvm::errs() << llvm::format("    input_local_size: %d(%d * %d * %d)\n",
                                              input_local_size, matrix_shape[0],
                                              (ceiling_func_shift(C_param_input_col, NPU_SHIFT)),
                                              channel_size_local));

        int tmp = (ceiling_func_shift(C_param_weight_col, NPU_SHIFT));
        DEBUG_BMNET(llvm::errs() << llvm::format("    bias_local_size: %d(%d * %d * %d)\n",
                                              bias_local_size, 2, tmp, channel_size_local));
      }
      break;
    } else if (slice_num[1] < input_col_num) {
      slice_num[1]++;
    } else if (slice_num[2] < weight_col_num) {
      slice_num[2]++;
    } else {
      slice_num[0]++;
    }
  }

  if (slice_num[0] == 1 && slice_num[2] == 1) {
    return 0;
  } else if (slice_num[1] == 1 && slice_num[2] == 1) {
    return 1;
  } else if (slice_num[0] == 1 && slice_num[1] == 1) {
    return 2;
  } else {
    return 3;
  }
}

u32 bmk1880v2_get_matrix_lmem_size(const BM1880v2BackendContext &ctx,
                                   bmk1880v2_matrix_lmem_shape_t s) {
  // w could be <= EN_NUM
  // E.g (152, 2)
  return s.n * (ceiling_func_shift(s.c, NPU_SHIFT) * ALIGN(s.w, EU_NUM));
}

// split input col
// output: bottom_data
// input:  top_data + weight_data + bias_data
// calculate:
//   top_data = bottom_data * weight_data + bias_data
// memory layout:
//   bank 0: bottom_data 0
//   bank 1: bottom_data 1
//   bank 2: weight_data + bias_diff
//   bank 3: top_data
static void fc_forward_parallel_slice_input_col_num(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation, int activation_method, gaddr_t activation_ga_slope,
    int activation_channel_shared, int activation_gt_scale, int activation_gt_rshift,
    int activation_le_scale, int activation_le_rshift, int W_param, int slice_num, bool weight_tp,
    int left_shift_width, int right_shift_width) {
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  int input_col_num_slice = input_col_num / slice_num + (0 < input_col_num % slice_num);

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "fc_forward_parallel_slice_input_col_num:\n"
                  "   in(%d, %d), out(,%d), have_bias %d, do_activation %d, activation_method %d\n"
                  "   W_param %d, slice_num %d, weight_tp %d\n"
                  "   bank_size %d, input_col_num_slice %d\n",
                  input_row_num, input_col_num, weight_col_num, have_bias, do_activation,
                  activation_method, W_param, slice_num, weight_tp, bank_size,
                  input_col_num_slice));

  bmk1880v2_matrix_tgmem_shape_t ts_top_ps_shape = {static_cast<u32>(2) * input_row_num,
                                                    static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_shape_t ts_top_shape = {static_cast<u32>(input_row_num),
                                                 static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_shape_t ts_slice_bottom_shape = {static_cast<u32>(input_row_num),
                                                          static_cast<u32>(input_col_num_slice)};
  bmk1880v2_matrix_tgmem_shape_t ts_slice_weight_shape = {static_cast<u32>(input_col_num_slice),
                                                          static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_shape_t ts_bias_shape = {static_cast<u32>(2),
                                                  static_cast<u32>(weight_col_num)};

  bmk1880v2_matrix_lmem_shape_t tl_top_ps_shape =
      ctx.matrix_lmem_default_shape(ts_top_ps_shape.row, ts_top_ps_shape.col);
  bmk1880v2_matrix_lmem_shape_t tl_slice_bottom_shape =
      ctx.matrix_lmem_default_shape(ts_slice_bottom_shape.row, ts_slice_bottom_shape.col);
  bmk1880v2_matrix_lmem_shape_t tl_slice_weight_shape =
      ctx.matrix_lmem_default_shape(ts_slice_weight_shape.row, ts_slice_weight_shape.col);
  bmk1880v2_matrix_lmem_shape_t tl_bias_shape =
      ctx.matrix_lmem_default_shape(ts_bias_shape.row, ts_bias_shape.col);
  bmk1880v2_matrix_lmem_shape_t tl_top_shape =
      ctx.matrix_lmem_default_shape(ts_top_shape.row, ts_top_shape.col);

  bmk1880v2_matrix_tgmem_stride_t ts_top_ps_stride = {ts_top_ps_shape.col};
  bmk1880v2_matrix_tgmem_stride_t ts_top_stride = {ts_top_shape.col};
  bmk1880v2_matrix_tgmem_stride_t ts_slice_bottom_stride = {static_cast<u32>(input_col_num)};
  bmk1880v2_matrix_tgmem_stride_t ts_slice_weight_stride = {static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_stride_t ts_bias_stride = {ts_bias_shape.col};

  bmk1880v2_matrix_lmem_t tl_bias;

  if (weight_tp) {
    ts_slice_weight_stride.row = input_col_num;
  }

  bmk1880v2_matrix_lmem_t tl_slice_bottom[2];
  bmk1880v2_tensor_lmem_t *tl_slice_bias = nullptr;

  tl_slice_bottom[0].start_address = 0;
  tl_slice_bottom[0].shape = tl_slice_bottom_shape;
  tl_slice_bottom[0].stride = ctx.matrix_lmem_default_stride(tl_slice_bottom[0].shape);
  tl_slice_bottom[0].fmt = FMT_I8;

  tl_slice_bottom[1].start_address = bank_size;
  tl_slice_bottom[1].shape = tl_slice_bottom_shape;
  tl_slice_bottom[1].stride = ctx.matrix_lmem_default_stride(tl_slice_bottom[1].shape);
  tl_slice_bottom[1].fmt = FMT_I8;

  bmk1880v2_matrix_lmem_t tl_slice_weight;
  tl_slice_weight.start_address = 2 * bank_size;
  tl_slice_weight.shape = tl_slice_weight_shape;
  tl_slice_weight.stride = ctx.matrix_lmem_default_stride(tl_slice_weight.shape);
  tl_slice_weight.fmt = FMT_I8;

  if (have_bias) {
    tl_bias.start_address =
        2 * bank_size + bmk1880v2_get_matrix_lmem_size(ctx, tl_slice_weight.shape);
    tl_bias.shape = tl_bias_shape;
    tl_bias.stride = ctx.matrix_lmem_default_stride(tl_bias.shape);
    tl_bias.fmt = FMT_I8;
  }

  bmk1880v2_matrix_lmem_t tl_top_ps;
  tl_top_ps.shape = tl_top_ps_shape;
  tl_top_ps.stride = ctx.matrix_lmem_default_stride(tl_top_ps.shape);
  tl_top_ps.fmt = FMT_I8;
  if (do_activation && activation_method == PRELU) {
    tl_top_ps.start_address = 3 * bank_size;
  } else {
    tl_top_ps.start_address = 7 * bank_size;
  }

  ctx.tdma_load_stride(&tl_slice_bottom[0], global_offset_bottom_data, ts_slice_bottom_stride,
                       CTRL_NEURON);

  if (weight_tp) {
    DEBUG_BMNET(llvm::errs() << llvm::format("  tdma_load_stride: tl_slice_weight\n"
                                          "      dst (%d, %d)\n",
                                          tl_slice_weight.shape.n, tl_slice_weight.shape.col));

    ctx.tdma_load_stride(&tl_slice_weight, global_offset_weight_data, ts_slice_weight_stride,
                         CTRL_TP | CTRL_WEIGHT);
  } else {
    ctx.tdma_load_stride(&tl_slice_weight, global_offset_weight_data, ts_slice_weight_stride,
                         CTRL_WEIGHT);
  }

  if (have_bias) {
    ctx.tdma_load_stride(&tl_bias, global_offset_bias_data, ts_bias_stride, CTRL_WEIGHT);
  }

  for (int slice_idx = 0; slice_idx < slice_num - 1; slice_idx++) {
    int cur = (slice_idx) % 2;
    int next = (slice_idx + 1) % 2;
    ctx.parallel_enable();

    {
      bmk1880v2_tiu_matrix_multiplication_param_t p;
      p.res = &tl_top_ps;  // TODO(wwcai): 16bit is not enough, to test
      p.left = &tl_slice_bottom[cur];
      p.right = &tl_slice_weight;
      p.bias = (slice_idx == 0 && have_bias) ? &tl_bias : nullptr;
      p.lshift_bits = left_shift_width;
      p.rshift_bits = left_shift_width;
      p.res_is_int8 = 0;
      p.add_result = (slice_idx > 0) ? 1 : 0;
      p.relu_enable = 0;

      DEBUG_BMNET(llvm::errs() << llvm::format("  [%d] L(%d, %d), R(%d, %d)\n", slice_idx,
                                            p.left->shape.n, p.left->shape.col, p.right->shape.n,
                                            p.right->shape.col));

      matrix_multiplication(ctx, p);
    }

    input_col_num_slice = input_col_num / slice_num + ((slice_idx + 1) < input_col_num % slice_num);
    gaddr_t slice_global_offset_bottom_data = get_slice_global_offset(
        global_offset_bottom_data, 0, slice_idx + 1, input_row_num, input_col_num, 1, slice_num);

    gaddr_t slice_global_offset_weight_data;
    if (weight_tp) {
      slice_global_offset_weight_data = get_slice_global_offset(
          global_offset_weight_data, 0, slice_idx + 1, weight_col_num, input_col_num, 1, slice_num);
    } else {
      slice_global_offset_weight_data = get_slice_global_offset(
          global_offset_weight_data, slice_idx + 1, 0, input_col_num, weight_col_num, slice_num, 1);
    }

    // Reshape
    // bmk does not keep eu-align info, user need to update stride if shape changed
    tl_slice_bottom[next].shape = ctx.matrix_lmem_default_shape(input_row_num, input_col_num_slice);
    tl_slice_bottom[next].stride = ctx.matrix_lmem_default_stride(tl_slice_bottom[next].shape);

    ctx.tdma_load_stride(&tl_slice_bottom[next], slice_global_offset_bottom_data,
                         ts_slice_bottom_stride, CTRL_NEURON);

    ctx.parallel_disable();

    // Reshape
    // bmk does not keep eu-align info, user need to update stride if shape changed
    tl_slice_weight.shape = ctx.matrix_lmem_default_shape(input_col_num_slice, weight_col_num);
    tl_slice_weight.stride = ctx.matrix_lmem_default_stride(tl_slice_weight.shape);

    if (weight_tp) {
      ctx.tdma_load_stride(&tl_slice_weight, slice_global_offset_weight_data,
                           ts_slice_weight_stride, CTRL_TP | CTRL_WEIGHT);
    } else {
      ctx.tdma_load_stride(&tl_slice_weight, slice_global_offset_weight_data,
                           ts_slice_weight_stride, CTRL_WEIGHT);
    }
  }

  bmk1880v2_matrix_lmem_t tl_top;
  tl_top = tl_top_ps;  // shape {2 * input_row_num, weight_col_num}
  if (slice_num == 1) {
    // Reshape
    // bmk does not keep eu-align info, user need to update stride if shape changed
    tl_top.shape = tl_top_shape;  // shape {input_row_num, weight_col_num}
    tl_top.stride = ctx.matrix_lmem_default_stride(tl_top.shape);
  }

  {
    bmk1880v2_tiu_matrix_multiplication_param_t p;
    p.res = &tl_top;
    p.left = &tl_slice_bottom[(slice_num - 1) % 2];
    p.right = &tl_slice_weight;
    p.bias = (slice_num == 1) ? &tl_bias : nullptr;
    p.lshift_bits = left_shift_width;
    p.rshift_bits = right_shift_width;
    p.res_is_int8 = 1;
    p.add_result = (slice_num > 1) ? 1 : 0;
    p.relu_enable = 0;

    DEBUG_BMNET(llvm::errs() << llvm::format("  L(%d, %d), R(%d, %d)\n", p.left->shape.n,
                                          p.left->shape.col, p.right->shape.n, p.right->shape.col));

    matrix_multiplication(ctx, p);
  }

  // Reshape
  // bmk does not keep eu-align info, user need to update stride if shape changed
  tl_top.shape = tl_top_ps_shape;  // shape {2 * input_row_num, weight_col_num}
  tl_top.stride = ctx.matrix_lmem_default_stride(tl_top.shape);

  if (do_activation && activation_method == RELU) {
    // shape of tl_top change dynamically,
    // tl_top_shape is not consitent with shape of tl_top
    // Initialize shape from input_row_num, weight_col_num
    bmk1880v2_matrix_lmem_t tl_top_data_int8;
    tl_top_data_int8.start_address = 7 * bank_size;
    tl_top_data_int8.fmt = FMT_I8;
    tl_top_data_int8.shape = ctx.matrix_lmem_default_shape(input_row_num, weight_col_num);
    tl_top_data_int8.stride = ctx.matrix_lmem_default_stride(tl_top_data_int8.shape);

    // Convert to tensor format
    bmk1880v2_tensor_lmem_t tl_top_tensor_result;
    tl_top_tensor_result.start_address = tl_top_data_int8.start_address;
    tl_top_tensor_result.shape = {tl_top_data_int8.shape.n, tl_top_data_int8.shape.c, 1,
                                  tl_top_data_int8.shape.w};
    tl_top_tensor_result.stride = {tl_top_data_int8.stride.n, tl_top_data_int8.stride.c,
                                   tl_top_data_int8.stride.h, 1};
    tl_top_tensor_result.fmt = FMT_I8;

    bmk1880v2_tiu_element_wise_max_param_t p_relu;
    p_relu.a = &tl_top_tensor_result;
    p_relu.max = &tl_top_tensor_result;
    p_relu.b_val = 0;
    p_relu.b_is_const = 1;
    p_relu.b_is_signed = 1;
    p_relu.layer_id = layer_id;
    ctx.tiu_element_wise_max(&p_relu);

    ctx.tdma_store_stride(&tl_top, global_offset_top_data, ts_top_stride, CTRL_NEURON);
  } else if (do_activation && activation_method == PRELU) {
    ASSERT(!activation_channel_shared);  // TODO
    /*
     * we need to use depthwise to implement PRELU, so the input should
     * be tensor. Besided Matrix shares the same prelu weight within the
     * same column, so we store the result matrix first and read it back
     * later with the special tensor format.
     */
    // shape of tl_top change dynamically,
    // tl_top_shape is not consitent with shape of tl_top
    // Initialize shape from input_row_num, weight_col_num
    bmk1880v2_matrix_lmem_t tl_top_data_int8;
    tl_top_data_int8.start_address = 3 * bank_size;
    tl_top_data_int8.fmt = FMT_I8;
    tl_top_data_int8.shape = ctx.matrix_lmem_default_shape(input_row_num, weight_col_num);
    tl_top_data_int8.stride = ctx.matrix_lmem_default_stride(tl_top_data_int8.shape);

    DEBUG_BMNET(llvm::errs() << "  PRELU =>" << "\n");

    DEBUG_BMNET(llvm::errs() << llvm::format("    tdma_store: tl_top_data_int8\n"
                                          "       dst (%d, %d), laddr 0x%lx, gaddr 0x%lx\n",
                                          tl_top_data_int8.shape.n, tl_top_data_int8.shape.col,
                                          tl_top_data_int8.start_address, global_offset_top_data));

    ctx.tdma_store_stride(&tl_top_data_int8, global_offset_top_data, ts_top_stride, CTRL_NEURON);

    ASSERT(!activation_channel_shared);  // TODO

    bmk1880v2_tensor_lmem_shape_t tl_ifmap_shape = {static_cast<u32>(input_row_num),
                                                    static_cast<u32>(weight_col_num), 1, 1};
    bmk1880v2_tensor_tgmem_shape_t ts_ifmap_shape = {static_cast<u32>(input_row_num),
                                                     static_cast<u32>(weight_col_num), 1, 1};

    bmk1880v2_tensor_lmem_t tl_fc_data;
    tl_fc_data.start_address = 3 * bank_size + tl_top_data_int8.shape.n * tl_top_data_int8.stride.n;
    tl_fc_data.fmt = FMT_I8;
    tl_fc_data.shape = tl_ifmap_shape;
    tl_fc_data.stride = ctx.tensor_lmem_default_stride(tl_ifmap_shape, 1);

    // load fc_reslut from global memory
    DEBUG_BMNET(llvm::errs() << llvm::format("    tdma_load: tl_fc_data\n"
                                          "       dst (%d, %d, %d, %d), laddr 0x%lx, gaddr 0x%lx\n",
                                          tl_fc_data.shape.n, tl_fc_data.shape.c,
                                          tl_fc_data.shape.h, tl_fc_data.shape.w,
                                          tl_fc_data.start_address, global_offset_top_data));

    ctx.tdma_load(&tl_fc_data, global_offset_top_data, CTRL_NEURON);

    bmk1880v2_tensor_lmem_shape_t tl_prelu_shape = {1, static_cast<u32>(weight_col_num), 1, 1};
    bmk1880v2_tensor_tgmem_shape_t ts_prelu_shape = {1, static_cast<u32>(weight_col_num), 1, 1};
    bmk1880v2_tensor_lmem_t tl_prelu;
    tl_prelu.start_address = 4 * bank_size;
    tl_prelu.shape = tl_prelu_shape;
    tl_prelu.stride = ctx.tensor_lmem_default_stride(tl_prelu.shape, 1);
    tl_prelu.fmt = FMT_I8;

    DEBUG_BMNET(llvm::errs() << llvm::format("    tdma_load: tl_prelu\n"
                                          "       dst (%d, %d, %d, %d), laddr 0x%lx, gaddr 0x%lx\n",
                                          tl_prelu.shape.n, tl_prelu.shape.c, tl_prelu.shape.h,
                                          tl_prelu.shape.w, tl_prelu.start_address,
                                          activation_ga_slope));

    ctx.tdma_load(&tl_prelu, activation_ga_slope, CTRL_WEIGHT);

    bmk1880v2_tensor_lmem_t tl_relu;
    tl_relu = tl_fc_data;
    tl_relu.start_address = 5 * bank_size;

    bmk1880v2_tensor_lmem_t tl_neg;
    tl_neg = tl_fc_data;
    tl_neg.start_address = 6 * bank_size;

    // No need to fill zero for neg(0, bottom)
    // Now, we have bank 7

#if 0
    bmk1880v2_tensor_lmem_t tl_zero;
    tl_zero = tl_fc_data;
    tl_zero.start_address = 7 * bank_size + tl_top_ps.shape.n * tl_top_ps.stride.n;

    {
      bmk1880v2_tdma_tg2l_tensor_fill_constant_param_t p;
      p.constant = 0;
      p.dst = &tl_zero;
      ctx.tdma_tg2l_tensor_fill_constant(&p);
    }
#endif

    // 0. relu = relu(tl_fc_data_int8)
    // 1. relu = (relu * gt_scale) >> gt_rshift
    // 2. neg = neg(0, botom)
    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
    // 4. tl_fc_data = or relu, neg

    // 0. relu = relu(tl_fc_data_int8)
    {
      bmk1880v2_tiu_element_wise_max_param_t p;
      p.max = &tl_relu;
      p.a = &tl_fc_data;
      p.b_is_const = 1;
      p.b_is_signed = 1;
      p.b_val = 0;
      p.layer_id = layer_id;
      ctx.tiu_element_wise_max(&p);
    }

    // 1. relu = (relu * gt_scale) >> gt_rshift
    {
      bmk1880v2_tiu_element_wise_mul_param_t p;
      p.res_high = nullptr;
      p.res_low = &tl_relu;
      p.a = &tl_relu;
      p.b_val = activation_gt_scale;
      p.b_is_signed = true;
      p.b_is_const = 1;
      p.rshift_bits = activation_gt_rshift;
      p.layer_id = layer_id;
      ctx.tiu_element_wise_mul(&p);
    }

    // 2. neg = neg(0, botom)
    {
      bmk1880v2_tiu_element_wise_min_param_t p;
      p.min = &tl_neg;
      p.a = &tl_fc_data;
      p.b_is_const = 1;
      p.b_val = 0;
      p.b_is_signed = 1;
      p.layer_id = layer_id;
      ctx.tiu_element_wise_min(&p);
    }

    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> le_rshift
    {
      bmk1880v2_tiu_depthwise_convolution_param_t p;
      p.ofmap = &tl_neg;
      p.ifmap = &tl_neg;
      p.weight = &tl_prelu;
      p.bias = nullptr;
      p.ins_h = 0;
      p.ins_last_h = 0;
      p.ins_w = 0;
      p.ins_last_w = 0;
      p.pad_top = 0;
      p.pad_bottom = 0;
      p.pad_left = 0;
      p.pad_right = 0;
      p.stride_h = 1;
      p.stride_w = 1;
      p.rshift_bits = activation_le_rshift;
      p.relu_enable = 0;
      p.layer_id = layer_id;
      ctx.tiu_depthwise_convolution(&p);
    }

    // 4. tl_fc_data = or relu, neg
    {
      bmk1880v2_tiu_element_wise_or_int8_param_t p;
      p.res = &tl_fc_data;
      p.a = &tl_relu;
      p.b = &tl_neg;
      p.layer_id = layer_id;
      ctx.tiu_element_wise_or_int8(&p);
    }

    DEBUG_BMNET(llvm::errs() << llvm::format("    tdma_store: tl_fc_data\n"
                                          "       dst (%d, %d, %d, %d), laddr 0x%lx, gaddr 0x%lx\n",
                                          tl_fc_data.shape.n, tl_fc_data.shape.c,
                                          tl_fc_data.shape.h, tl_fc_data.shape.w,
                                          tl_fc_data.start_address, global_offset_top_data));

    ctx.tdma_store(&tl_fc_data, global_offset_top_data, CTRL_NEURON);

    DEBUG_BMNET(llvm::errs() << "  <= PRELU" << "\n");

  } else {
    // shape of tl_top change dynamically,
    // tl_top_shape is not consitent with shape of tl_top
    // Initialize shape from input_row_num, weight_col_num
    bmk1880v2_matrix_lmem_t tl_top_data_int8;
    tl_top_data_int8.start_address = 7 * bank_size;
    tl_top_data_int8.fmt = FMT_I8;
    tl_top_data_int8.shape = ctx.matrix_lmem_default_shape(input_row_num, weight_col_num);
    tl_top_data_int8.stride = ctx.matrix_lmem_default_stride(tl_top_data_int8.shape);
    ctx.tdma_store_stride(&tl_top_data_int8, global_offset_top_data, ts_top_stride, CTRL_NEURON);
  }

  DEBUG_BMNET(llvm::errs() << "<= fc_forward_parallel_slice_input_col_num" << "\n");
}

static bool fc_forward_serial_slice_weight_col_num(
    const BM1880v2BackendContext &ctx, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation,
    // new added
    int activation_method, gaddr_t activation_ga_slope, int activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift,
    // end
    bool weight_tp,  // new added
    int left_shift_width, int right_shift_width) {
  // Assume L Matrix = M x K, R Matrix = K x N, Y Matrix = M x N

  ASSERT(0);
  return true;
#if 0
  int M = input_row_num;
  int K = input_col_num;
  int N = weight_col_num;

  DEBUG_BMNET(llvm::errs() << "M=" << M << " K=" << K << " N=" << N
                        << ", Partial-Sum shift-width = " << left_shift_width << "\n");

  int cur_limit;
  int M_slice = 1;
  cur_limit = 4095;  // HW limitation
  int K_slice = (K >= cur_limit) ? cur_limit : K;
#ifdef USE_MATRIX_W16
  cur_limit = ctx.hw.npu_num * ctx.hw.eu_num;
#else
  cur_limit = ctx.hw.npu_num * 2 * ctx.hw.eu_num;
#endif
  int N_slice = (N >= cur_limit) ? cur_limit : N;  // HW limitation

  // Check Local Memory Space for this algorithm
  // 1. Calculates matrix size
  //
  bool is_align = true;
  shape_t shape_L = shape_t2(M_slice, K_slice);
#ifdef USE_MATRIX_W16
  shape_t shape_R = shape_t2(K_slice, N_slice, ctx.hw.eu_num);
  shape_t shape_Y_I16 = shape_t2(2 * M_slice, N_slice, ctx.hw.eu_num);  // 16-bit output
  shape_t shape_Y_I8 = shape_t2(M_slice, N_slice, ctx.hw.eu_num);
  shape_t shape_B = shape_t2(2, N_slice, ctx.hw.eu_num);
#else
  shape_t shape_R = shape_t2(K_slice, N_slice);
  shape_t shape_Y_I16 = shape_t2(2 * M_slice, N_slice);  // 16-bit output
  shape_t shape_Y_I8 = shape_t2(M_slice, N_slice);
  shape_t shape_B = shape_t2(2, N_slice);
#endif

  int L_size = ctx.tl_shape_to_size(shape_L, is_align, FMT_I8);
  int R_size = ctx.tl_shape_to_size(shape_R, is_align, FMT_I8);
  int Y_size = ctx.tl_shape_to_size(shape_Y_I16, is_align, FMT_I8);
  int B_size = ctx.tl_shape_to_size(shape_B, is_align, FMT_I8);

  if (!have_bias) {
    B_size = 0;
  }

  // 2. Shrink K_slice if memory is not enough (reduces precision)
  while ((B_size + L_size + R_size + Y_size) > LOCAL_MEM_SIZE) {
    if (K_slice == 1) {
      // No way
      DEBUG_BMNET(llvm::errs() << llvm::format(
                      "Serial Weight Slicing Failed: M_slice=%d K_slice=%d N_slice=%d\n", M_slice,
                      K_slice, N_slice));
      DEBUG_BMNET(
          llvm::errs() << llvm::format("L_size=%d R_size=%d Y_size=%d\n", L_size, R_size, Y_size));
      return false;
    }
    K_slice--;
    shape_L = shape_t2(M_slice, K_slice);
#ifdef USE_MATRIX_W16
    shape_R = shape_t2(K_slice, N_slice, ctx.hw.eu_num);
#else
    shape_R = shape_t2(K_slice, N_slice);
#endif
    L_size = ctx.tl_shape_to_size(shape_L, is_align, FMT_I8);
    R_size = ctx.tl_shape_to_size(shape_R, is_align, FMT_I8);
  }

  int N_secs = idiv_round(N, N_slice);
  int K_secs = idiv_round(K, K_slice);

  DEBUG_BMNET(llvm::errs() << "K_secs = " << K_secs << " N_secs = " << N_secs << "\n");
  DEBUG_BMNET(llvm::errs() << "M_slice=" << M_slice << " K_slice=" << K_slice << " N_slice=" << N_slice
                        << "\n");
  DEBUG_BMNET(llvm::errs() << "L_size=" << L_size << " R_size=" << R_size << " Y_size=" << Y_size
                        << " B_size=" << B_size << "\n");

  //
  // End of Check
  //

  ctrl_t load_weight_ctrls = CTRL_WEIGHT;
  if (weight_tp) {
    load_weight_ctrls += CTRL_TP;
  }

  // LMem allocation
  tensor_lmem *tl_Y = ctx.tl_alloc(shape_Y_I16, FMT_I8, CTRL_AL);
  tensor_lmem *tl_L = ctx.tl_alloc(shape_L, FMT_I8, CTRL_AL);
  tensor_lmem *tl_R = ctx.tl_alloc(shape_R, FMT_I8, CTRL_AL);
  tensor_lmem *tl_B = nullptr;

  if (have_bias) {
    tl_B = ctx.tl_alloc(shape_B, FMT_I8, CTRL_AL);
  }

  if (tl_Y == nullptr) {
    assert(false && "fail to allocate tl_Y");
    exit(-1);
  }

  if (tl_L == nullptr) {
    assert(false && "fail to allocate tl_L");
    exit(-1);
  }

  if (tl_R == nullptr) {
    assert(false && "fail to allocate tl_R");
    exit(-1);
  }

  if (have_bias && (tl_B == nullptr)) {
    assert(false && "fail to allocate tl_B");
    exit(-1);
  }

  // Do FC
  int cur_N_slice;
  int cur_K_slice;
  for (int M_i = 0; M_i < M; M_i++) {
    for (int N_i = 0; N_i < N_secs; N_i++) {
      // Reshape for last loop
      if (N_i == (N_secs - 1)) {
        cur_N_slice = N - N_slice * (N_secs - 1);
      } else {
        cur_N_slice = N_slice;
      }

      for (int K_i = 0; K_i < K_secs; K_i++) {
        // Reshape for last loop
        if (K_i == (K_secs - 1)) {
          cur_K_slice = K - K_slice * (K_secs - 1);
        } else {
          cur_K_slice = K_slice;
        }

        tl_reshape(tl_Y, shape_t2(2 * M_slice, cur_N_slice));
        tl_reshape(tl_L, shape_t2(M_slice, cur_K_slice));
        tl_reshape(tl_R, shape_t2(cur_K_slice, cur_N_slice));
        if (have_bias) {
          tl_reshape(tl_B, shape_t2(2, cur_N_slice));
        }

        //
        // Load L Matrix (M x K)
        // 0*K + 0*K_slice, 0*K 1*K_slice, ..., 0*K + (K_sec-1)*slice
        // (Msec-1)*K + (K_i -1)*K_slice, ...
        //
        gaddr_t L_addr = global_offset_bottom_data + M_i * 1 * K + K_i * K_slice;
        ctx.gdma_load(tl_L, L_addr, CTRL_NEURON);

        //
        // Load R Matrix (K x N)
        // 0*N + 0*N_slice, 0*N + 1*N_slice, ...
        // K_i*N + N_i*N_slice, ...
        //
        gaddr_t cur_offset = K_i * K_slice * N + N_i * N_slice;

        gaddr_t R_addr = global_offset_weight_data + cur_offset;
        ctx.gdma_load(tl_R, R_addr, load_weight_ctrls);

        //
        // Load B (1 x N)
        // 0*N_slice, 1*N_slice, N_i * N_slice, ...
        //
        if (have_bias) {
          gaddr_t B_addr = global_offset_bias_data + N_i * N_slice;
          ctx.gdma_load(tl_B, B_addr, CTRL_WEIGHT);
        }

        // Execution
        bmk1880v2_matrix_mac_param_t p;
        p.res = tl_Y;  // 16-bit output space
        p.left = tl_L;
        p.right = tl_R;
        p.bias = (K_i == 0) ? tl_B : nullptr;

        p.lshift_width = left_shift_width;  // TODO(wwcai): compute left shift width

        if (K_i == (K_secs - 1)) {
          p.rshift_width = right_shift_width;
        } else {
          p.rshift_width = left_shift_width;
        }

        p.res_is_int8 = (K_i == (K_secs - 1)) ? 1 : 0;  // use 16-bit output
        p.ctrls = (K_i == 0) ? CTRL_NULL : CTRL_RA;
        ctx.tpu_matrix_mac(&p);
      }

      //
      // Store Y Matrix (M x N)
      // 0*N + 0*N_slice, 0*N + 1*N_slice, ...
      // (M_i)*N + (N_i)*N_slice, ...
      //
      gaddr_t output_addr = global_offset_top_data + M_i * N + N_i * N_slice;
      tl_reshape(tl_Y, shape_t2(M_slice, cur_N_slice));  // use I8 for final output

      if (do_activation) {
        if (activation_method == RELU) {
          // TODO(arcbbb): should be merged to tl_matrix_mac
          bmk1880v2_relu_param_t p13;
          p13.ofmap = tl_Y;
          p13.ifmap = tl_Y;
          ctx.tpu_relu(&p13);
        } else if (do_activation && activation_method == PRELU) {
          // do this later
        } else {
          assert(false && "unknown activation method");
          exit(-1);
        }
      }

      ctx.gdma_store(tl_Y, output_addr, CTRL_NEURON);
      tl_reshape(tl_Y, shape_t2(2 * M_slice, cur_N_slice));  // use I16 for final output
    }
  }

  if (do_activation && activation_method == PRELU) {
    tg_prelu(ctx, global_offset_top_data, activation_ga_slope, input_row_num, weight_col_num,
             activation_channel_shared, activation_gt_scale, activation_gt_rshift,
             activation_le_scale, activation_le_rshift);
  }

  if (have_bias) {
    ctx.tl_free(tl_B);
  }

  ctx.tl_free(tl_R);
  ctx.tl_free(tl_L);
  ctx.tl_free(tl_Y);
  return true;
#endif
}

static void fc_forward_parallel_get_slice_num_multidim(const BM1880v2BackendContext &ctx,
                                                       // input parameters
                                                       int input_row_num, int input_col_num,
                                                       int weight_col_num, int W_param,
                                                       // output parameters
                                                       int *slice_num) {
  int channel_size_local = get_csize_local(ctx, 1, W_param);
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  int bank_size_half = bank_size >> 1;
  slice_num[0] = slice_num[1] = slice_num[2] = 1;

  // input blob
  int C_param = (input_col_num + W_param - 1) / W_param;
  int input_local_size =
      input_row_num * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
  int input_slice_num = (input_local_size + bank_size - 1) / bank_size;

  int slice_num_0_input = (input_slice_num > input_row_num) ? input_row_num : input_slice_num;
  int slice_num_1_input =
      (input_slice_num > input_row_num) ? (input_slice_num + input_col_num - 1) / input_col_num : 1;

  // output blob
  C_param = (weight_col_num + W_param - 1) / W_param;
  int output_local_size =
      2 * (input_row_num + 1) * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
  int output_slice_num = (output_local_size + bank_size - 1) / bank_size;
  int slice_num_0_output = (output_slice_num > input_row_num) ? input_row_num : output_slice_num;
  int slice_num_2_output = (output_slice_num > input_row_num)
                               ? (output_slice_num + input_row_num - 1) / input_row_num
                               : 1;

  // weight blob
  C_param = (weight_col_num + W_param - 1) / W_param;
  int weight_local_size =
      input_col_num * (ceiling_func_shift(C_param, NPU_SHIFT)) * channel_size_local;
  int weight_slice_num = (weight_local_size + bank_size_half - 1) / bank_size_half;
  int slice_num_1_weight = (weight_slice_num > input_col_num) ? input_col_num : weight_slice_num;
  int slice_num_2_weight = (weight_slice_num > input_col_num)
                               ? (weight_slice_num + input_col_num - 1) / input_col_num
                               : 1;

  // initialize slice numbers
  slice_num[0] = get_max(slice_num_0_output, slice_num_0_input);
  slice_num[1] = get_max(slice_num_1_weight, slice_num_1_input);
  slice_num[2] = get_max(slice_num_2_weight, slice_num_2_output);

  // fine-tuning
  int matrix_shape[3] = {1, 1, 1};
  while (true) {
    matrix_shape[0] = (input_row_num + slice_num[0] - 1) / slice_num[0];
    matrix_shape[1] = (input_col_num + slice_num[1] - 1) / slice_num[1];
    matrix_shape[2] = (weight_col_num + slice_num[2] - 1) / slice_num[2];

    int C_param_input_col = (matrix_shape[1] + W_param - 1) / W_param;
    int C_param_weight_col = (matrix_shape[2] + W_param - 1) / W_param;

    int weight_local_size =
        matrix_shape[1] * (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) * channel_size_local;
    int output_local_size = 2 * (matrix_shape[0] + 1) *
                            (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) *
                            channel_size_local;
    int input_local_size =
        matrix_shape[0] * (ceiling_func_shift(C_param_input_col, NPU_SHIFT)) * channel_size_local;
    bool slicing_success = (input_local_size <= bank_size) && (output_local_size <= bank_size) &&
                           (weight_local_size <= bank_size_half);
    if (slicing_success) {
      int bias_local_size =
          2 * (ceiling_func_shift(C_param_weight_col, NPU_SHIFT)) * channel_size_local;
      DEBUG_BMNET(llvm::errs() << "multi-dim slicing:"
                            << "\n");
      DEBUG_BMNET(llvm::errs() << "weight_local_size: " << weight_local_size << "\n");
      DEBUG_BMNET(llvm::errs() << "output_local_size: " << output_local_size << "\n");
      DEBUG_BMNET(llvm::errs() << "input_local_size: " << input_local_size << "\n");
      DEBUG_BMNET(llvm::errs() << "bias_local_size: " << bias_local_size << "\n");
      return;
    } else if (slice_num[1] < input_col_num) {
      slice_num[1]++;
    } else if (slice_num[2] < weight_col_num) {
      slice_num[2]++;
    } else {
      slice_num[0]++;
    }
  }
}

static void fc_forward_parallel_slicing_multidim_init(const BM1880v2BackendContext &ctx,
                                                      gaddr_t *slice_global_offset,
                                                      int *matrix_shape, int *slice_row_stride,
                                                      int W_param) {
  gaddr_t global_offset_bottom_data = slice_global_offset[0];
  gaddr_t global_offset_weight_data = slice_global_offset[1];

  int input_row_num = matrix_shape[0];
  int input_col_num = matrix_shape[1];
  int weight_col_num = matrix_shape[2];
  int st_bot_data = slice_row_stride[0];
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

  bmk1880v2_matrix_tgmem_shape_t s_bottom_data = {static_cast<u32>(input_row_num),
                                                  static_cast<u32>(input_col_num)};
  bmk1880v2_matrix_tgmem_shape_t s_weight_data = {static_cast<u32>(input_col_num),
                                                  static_cast<u32>(weight_col_num)};

  bmk1880v2_matrix_tgmem_stride_t st_bottom_data = {static_cast<u32>(st_bot_data)};
  bmk1880v2_matrix_tgmem_stride_t st_weight_data = {static_cast<u32>(slice_row_stride[3])};

  bmk1880v2_matrix_lmem_t tl_bottom_data;
  tl_bottom_data.start_address = 0;
  tl_bottom_data.fmt = FMT_I8;
  tl_bottom_data.shape = ctx.matrix_lmem_default_shape(s_bottom_data.row, s_bottom_data.col);
  tl_bottom_data.stride = ctx.matrix_lmem_default_stride(tl_bottom_data.shape);

  bmk1880v2_matrix_lmem_t tl_weight_data;
  tl_weight_data.start_address = 2 * bank_size;
  tl_weight_data.fmt = FMT_I8;
  tl_weight_data.shape = ctx.matrix_lmem_default_shape(s_weight_data.row, s_weight_data.col);
  tl_weight_data.stride = ctx.matrix_lmem_default_stride(tl_weight_data.shape);

  ctx.tdma_load_stride(&tl_bottom_data, global_offset_bottom_data, st_bottom_data, CTRL_NEURON);
  ctx.tdma_load_stride(&tl_weight_data, global_offset_weight_data, st_weight_data, CTRL_WEIGHT);
}

// forward multi slice
// output: top_data
// input:  weight_data and bottom_data and bias_data
// calculate:
//   top_data = bottom_data * weight_data + bias_data
// memory layout:
//   bank 0: 1/2 bottom_data + 1/2 bottom_data_next
//   bank 1: weight_data(_next)   one is next stage
//   bank 2: weight_data(_next)
//   bank 3: top_data + bias_data
static void fc_forward_parallel_slicing_multi_dimension_internal(
    const BM1880v2BackendContext &ctx, u32 layer_id, int *slice_idx, int *slice_num,
    gaddr_t *slice_global_offset, int *matrix_shape, gaddr_t *slice_global_offset_next,
    int *matrix_shape_next, int *slice_row_stride, int have_bias, int do_activation,
    int activation_method, int W_param, int left_shift_width, int right_shift_width) {
  gaddr_t global_offset_bias_data = slice_global_offset[2];
  gaddr_t global_offset_top_data = slice_global_offset[3];

  int input_row_num = matrix_shape[0];
  int input_col_num = matrix_shape[1];
  int weight_col_num = matrix_shape[2];

  gaddr_t global_offset_bottom_data_next = slice_global_offset_next[0];
  gaddr_t global_offset_weight_data_next = slice_global_offset_next[1];

  int input_row_num_next = matrix_shape_next[0];
  int input_col_num_next = matrix_shape_next[1];
  int weight_col_num_next = matrix_shape_next[2];

  int row_stride_bottom_data = slice_row_stride[0];
  int row_stride_bias_data = slice_row_stride[2];
  int row_stride_top_data = slice_row_stride[3];

  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  int hf_bsize = bank_size / 2;

  bmk1880v2_matrix_tgmem_shape_t s_bottom_data_next = {static_cast<u32>(input_row_num_next),
                                                       static_cast<u32>(input_col_num_next)};
  bmk1880v2_matrix_tgmem_shape_t s_weight_data_next = {static_cast<u32>(input_col_num_next),
                                                       static_cast<u32>(weight_col_num_next)};
  bmk1880v2_matrix_tgmem_shape_t s_bottom_data = {static_cast<u32>(input_row_num),
                                                  static_cast<u32>(input_col_num)};
  bmk1880v2_matrix_tgmem_shape_t s_weight_data = {static_cast<u32>(input_col_num),
                                                  static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_shape_t s_top_data = {2 * static_cast<u32>(input_row_num),
                                               static_cast<u32>(weight_col_num)};
  bmk1880v2_matrix_tgmem_shape_t s_bias_data = {2, static_cast<u32>(weight_col_num)};

  bmk1880v2_matrix_tgmem_stride_t st_bottom_data = {static_cast<u32>(row_stride_bottom_data)};
  bmk1880v2_matrix_tgmem_stride_t st_bias_data = {static_cast<u32>(row_stride_bias_data)};
  bmk1880v2_matrix_tgmem_stride_t st_top_data = {static_cast<u32>(row_stride_top_data)};
  bmk1880v2_matrix_tgmem_stride_t st_weight_data = {static_cast<u32>(slice_row_stride[3])};

  int cur = slice_idx[1] % 2;
  int next = (slice_idx[1] + 1) % 2;

  bmk1880v2_matrix_lmem_t tl_bias_data = {0};

  bmk1880v2_matrix_lmem_t tl_bottom_data;
  tl_bottom_data.start_address = bank_size * cur;
  tl_bottom_data.fmt = FMT_I8;
  tl_bottom_data.shape = ctx.matrix_lmem_default_shape(s_bottom_data.row, s_bottom_data.col);
  tl_bottom_data.stride = ctx.matrix_lmem_default_stride(tl_bottom_data.shape);

  bmk1880v2_matrix_lmem_t tl_bottom_data_next;
  tl_bottom_data_next.start_address = bank_size * next;
  tl_bottom_data_next.fmt = FMT_I8;
  tl_bottom_data_next.shape =
      ctx.matrix_lmem_default_shape(s_bottom_data_next.row, s_bottom_data_next.col);
  tl_bottom_data_next.stride = ctx.matrix_lmem_default_stride(tl_bottom_data_next.shape);

  bmk1880v2_matrix_lmem_t tl_weight_data;
  tl_weight_data.start_address = 2 * bank_size + hf_bsize * cur;
  tl_weight_data.fmt = FMT_I8;
  tl_weight_data.shape = ctx.matrix_lmem_default_shape(s_weight_data.row, s_weight_data.col);
  tl_weight_data.stride = ctx.matrix_lmem_default_stride(tl_weight_data.shape);

  bmk1880v2_matrix_lmem_t tl_weight_data_next;
  tl_weight_data_next.start_address = 2 * bank_size + hf_bsize * next;
  tl_weight_data_next.fmt = FMT_I8;
  tl_weight_data_next.shape =
      ctx.matrix_lmem_default_shape(s_weight_data_next.row, s_weight_data_next.col);
  tl_weight_data_next.stride = ctx.matrix_lmem_default_stride(tl_weight_data_next.shape);

  bmk1880v2_matrix_lmem_t tl_top_data;
  tl_top_data.start_address = 3 * bank_size;
  tl_top_data.fmt = FMT_I8;
  tl_top_data.shape = ctx.matrix_lmem_default_shape(s_top_data.row, s_top_data.col);
  tl_top_data.stride = ctx.matrix_lmem_default_stride(tl_top_data.shape);

  if (have_bias) {
    tl_bias_data.start_address =
        3 * bank_size + bmk1880v2_get_matrix_lmem_size(ctx, tl_top_data.shape);
    tl_bias_data.fmt = FMT_I8;
    tl_bias_data.shape = ctx.matrix_lmem_default_shape(s_bias_data.row, s_bias_data.col);
    tl_bias_data.stride = ctx.matrix_lmem_default_stride(tl_bias_data.shape);
  }

  ctx.parallel_enable();
  // DMA move matrix bottom_data
  if (slice_num[1] - 1 > slice_idx[1]) {
    ctx.tdma_load_stride(&tl_bottom_data_next, global_offset_bottom_data_next, st_bottom_data,
                         CTRL_NEURON);
    ctx.tdma_load_stride(&tl_weight_data_next, global_offset_weight_data_next, st_weight_data,
                         CTRL_WEIGHT);

    {
      bmk1880v2_tiu_matrix_multiplication_param_t p;
      p.res = &tl_top_data;  // TODO(wwcai): 16bit is not enough, to test
      p.left = &tl_bottom_data;
      p.right = &tl_weight_data;
      p.bias = nullptr;
      p.lshift_bits = left_shift_width;
      p.rshift_bits = left_shift_width;
      p.res_is_int8 = 0;
      p.add_result = (slice_idx[1] > 0) ? 1 : 0;
      p.relu_enable = 0;

      DEBUG_BMNET(llvm::errs() << llvm::format("  L(%d, %d), R(%d, %d)\n", p.left->shape.n,
                                            p.left->shape.col, p.right->shape.n,
                                            p.right->shape.col));

      matrix_multiplication(ctx, p);
    }
  }
  ctx.parallel_disable();

  bmk1880v2_matrix_tgmem_shape_t s_top_data_int8 = {static_cast<u32>(input_row_num),
                                                    static_cast<u32>(weight_col_num)};

  if (slice_idx[1] == slice_num[1] - 1) {
    if (slice_num[1] == 1) {
      // Reshape
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_top_data.shape = ctx.matrix_lmem_default_shape(s_top_data_int8.row, s_top_data_int8.col);
      tl_top_data.stride = ctx.matrix_lmem_default_stride(tl_top_data.shape);
    }

    if (have_bias) {
      ctx.tdma_load_stride(&tl_bias_data, global_offset_bias_data, st_bias_data, CTRL_WEIGHT);
    }

    {
      bmk1880v2_tiu_matrix_multiplication_param_t p;
      p.res = &tl_top_data;  // TODO(wwcai): 16bit is not enough, to test
      p.left = &tl_bottom_data;
      p.right = &tl_weight_data;
      p.bias = &tl_bias_data;
      p.lshift_bits = left_shift_width;
      p.rshift_bits = right_shift_width;
      p.res_is_int8 = 1;
      p.add_result = (slice_num[1] > 1) ? 1 : 0;
      p.relu_enable = 0;

      DEBUG_BMNET(llvm::errs() << llvm::format("  L(%d, %d), R(%d, %d)\n", p.left->shape.n,
                                            p.left->shape.col, p.right->shape.n,
                                            p.right->shape.col));

      matrix_multiplication(ctx, p);
    }
  }

  if (slice_idx[1] == slice_num[1] - 1) {
    bmk1880v2_matrix_lmem_t tl_top_data_int8;
    tl_top_data_int8.start_address = tl_top_data.start_address;
    tl_top_data_int8.fmt = FMT_I8;
    tl_top_data_int8.shape =
        ctx.matrix_lmem_default_shape(s_top_data_int8.row, s_top_data_int8.col);
    tl_top_data_int8.stride = ctx.matrix_lmem_default_stride(tl_top_data_int8.shape);

    if (do_activation && activation_method == RELU) {
      // Convert to tensor format
      bmk1880v2_tensor_lmem_t tl_data;
      tl_data.start_address = tl_top_data_int8.start_address;
      tl_data.fmt = tl_top_data_int8.fmt;
      tl_data.shape = {tl_top_data_int8.shape.n, tl_top_data_int8.shape.c, 1,
                       tl_top_data_int8.shape.w};
      tl_data.stride = {tl_top_data_int8.stride.n, tl_top_data_int8.stride.c,
                        tl_top_data_int8.stride.h, 1};

      bmk1880v2_tiu_element_wise_max_param_t p13;
      p13.a = &tl_data;
      p13.max = &tl_data;
      p13.b_val = 0;
      p13.b_is_const = 1;
      p13.b_is_signed = 1;
      p13.layer_id = layer_id;
      ctx.tiu_element_wise_max(&p13);

    } else if (do_activation && activation_method == PRELU) {
      ASSERT(0);
    }
    ctx.tdma_store_stride(&tl_top_data_int8, global_offset_top_data, st_top_data, CTRL_NEURON);
  }
}

static void fc_forward_parallel_slicing_multi_dimension(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation, int activation_method, int W_param, int *slice_num,
    int left_shift_width, int right_shift_width) {
  int slice_row_stride[4] = {0, 0, 0, 0};
  slice_row_stride[0] = input_col_num;
  slice_row_stride[1] = input_col_num;
  slice_row_stride[2] = weight_col_num;
  slice_row_stride[3] = weight_col_num;

  gaddr_t slice_global_offset[4] = {0, 0, 0, 0};
  gaddr_t slice_global_offset_next[4] = {0, 0, 0, 0};
  int slice_idx[3] = {0, 0, 0};
  int matrix_shape[3] = {0, 0, 0};
  int matrix_shape_next[3] = {0, 0, 0};
  for (slice_idx[0] = 0; slice_idx[0] < slice_num[0]; slice_idx[0]++) {
    matrix_shape[0] = input_row_num / slice_num[0] + (slice_idx[0] < input_row_num % slice_num[0]);
    matrix_shape_next[0] =
        input_row_num / slice_num[0] + (0 + slice_idx[0] < input_row_num % slice_num[0]);

    for (slice_idx[2] = 0; slice_idx[2] < slice_num[2]; slice_idx[2]++) {
      matrix_shape[2] =
          weight_col_num / slice_num[2] + (slice_idx[2] < weight_col_num % slice_num[2]);
      matrix_shape_next[2] =
          weight_col_num / slice_num[2] + (0 + slice_idx[2] < weight_col_num % slice_num[2]);

      for (slice_idx[1] = 0; slice_idx[1] < slice_num[1]; slice_idx[1]++) {
        matrix_shape[1] =
            input_col_num / slice_num[1] + (slice_idx[1] < input_col_num % slice_num[1]);
        matrix_shape_next[1] =
            input_col_num / slice_num[1] + (1 + slice_idx[1] < input_col_num % slice_num[1]);

        slice_global_offset[0] =
            get_slice_global_offset(global_offset_bottom_data, slice_idx[0], slice_idx[1],
                                    input_row_num, input_col_num, slice_num[0], slice_num[1]);
        slice_global_offset[1] =
            get_slice_global_offset(global_offset_weight_data, slice_idx[1], slice_idx[2],
                                    input_col_num, weight_col_num, slice_num[1], slice_num[2]);
        slice_global_offset[2] = get_slice_global_offset(global_offset_bias_data, 0, slice_idx[2],
                                                         1, weight_col_num, 1, slice_num[2]);
        slice_global_offset[3] =
            get_slice_global_offset(global_offset_top_data, slice_idx[0], slice_idx[2],
                                    input_row_num, weight_col_num, slice_num[0], slice_num[2]);
        slice_global_offset_next[0] =
            get_slice_global_offset(global_offset_bottom_data, slice_idx[0], slice_idx[1] + 1,
                                    input_row_num, input_col_num, slice_num[0], slice_num[1]);
        slice_global_offset_next[1] =
            get_slice_global_offset(global_offset_weight_data, slice_idx[1] + 1, slice_idx[2],
                                    input_col_num, weight_col_num, slice_num[1], slice_num[2]);
        if (slice_idx[1] == 0) {
          fc_forward_parallel_slicing_multidim_init(ctx, slice_global_offset, matrix_shape,
                                                    slice_row_stride, W_param);
        }
        fc_forward_parallel_slicing_multi_dimension_internal(
            ctx, layer_id, slice_idx, slice_num, slice_global_offset, matrix_shape,
            slice_global_offset_next, matrix_shape_next, slice_row_stride, have_bias, do_activation,
            activation_method, W_param, left_shift_width, right_shift_width);
      }
    }
  }
}

static void fc_slicing_multi_dimention(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation, int activation_method, gaddr_t activation_ga_slope,
    int activation_channel_shared, int activation_gt_scale, int activation_gt_rshift,
    int activation_le_scale, int activation_le_rshift, bool weight_tp, int left_shift_width,
    int right_shift_width) {
  // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
  u32 M = static_cast<u32>(input_row_num);
  u32 K = static_cast<u32>(input_col_num);
  u32 N = static_cast<u32>(weight_col_num);

  DEBUG_BMNET(llvm::errs() << llvm::format("fc_slicing_multi_dimension\n"
                                        "  Y(%d, %d) = L(%d, %d) * R(%d, %d) + B(%d, %d)\n",
                                        M, N, M, K, K, N, 1, N));

  // Split N <= max total eu number
  u32 total_eu = ctx.hw.npu_num * ctx.hw.eu_num;
  u32 tiled_N = (N >= total_eu) ? total_eu : N;

  // Split K based on lane size
  u32 lane_size = ctx.hw.local_mem_size;
  u32 total_mem_size = ctx.hw.npu_num * ctx.hw.local_mem_size;
  u32 max_k = (1 << 12) - 1;  // 1880v2: 12 bit
  u32 tiled_K = (K >= max_k) ? max_k : K;

  // Tiled Y
  bmk1880v2_matrix_lmem_t tl_tiled_Y = {0};
  tl_tiled_Y.fmt = FMT_I8;

  // Tiled L
  bmk1880v2_matrix_lmem_t tl_tiled_L = {0};
  tl_tiled_L.fmt = FMT_I8;

  // Tiled R
  bmk1880v2_matrix_lmem_t tl_tiled_R = {0};
  tl_tiled_R.fmt = FMT_I8;

  // Tiled B
  u32 bias_size = 0;
  bmk1880v2_matrix_lmem_t tl_tiled_B = {0};
  if (have_bias) {
    // TIU opd2_n = 1, H/W use b_stride to read upper 8-bit
    // But bmk demand n = 2, but assign opd2_n = 1.
    // Let dma load and tiu use different shape.
    tl_tiled_B.fmt = FMT_I8;
    tl_tiled_B.shape = ctx.matrix_lmem_default_shape(2, tiled_N);  // 16bit
    tl_tiled_B.stride = ctx.matrix_lmem_default_stride(tl_tiled_B.shape);
  }

  // Tiled local memory layout:
  //   Y at fixed position since last tiled ones may be smaller
  //
  //   tiled Y, [7:0]
  //   tiled Y, [15:8]
  //   tiled Y, [23:16]
  //   tiled Y, [31:24]
  //   tiled L
  //   tiled R
  //   tiled B, [7:0], if existed
  //   tiled B, [15:8], if existed

  // Find max tiled K
  u32 required_size = 0;
  do {
    required_size = 0;  // Start of LMEM

    // Not split M since we don't want to reload L(weight)
    // or reload partial result of different M.
    //
    // Y(M, N) = L(M, K) * R(K, N) + B(1, N)
    // tiled_Y(M, tiled_N) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) + tiled_B(1, tiled_N)

    // tiled Y, 32bit
    tl_tiled_Y.start_address = required_size;
    tl_tiled_Y.shape = ctx.matrix_lmem_default_shape(M, tiled_N);
    tl_tiled_Y.stride = ctx.matrix_lmem_default_stride(tl_tiled_Y.shape);
    required_size += 4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n;

    // tiled L
    tl_tiled_L.start_address = required_size;
    tl_tiled_L.shape = ctx.matrix_lmem_default_shape(M, tiled_K);
    tl_tiled_L.stride = ctx.matrix_lmem_default_stride(tl_tiled_L.shape);
    required_size += tl_tiled_L.shape.n * tl_tiled_L.stride.n;  // assume n = 2

    // tiled R
    tl_tiled_R.start_address = required_size;
    tl_tiled_R.shape = ctx.matrix_lmem_default_shape(tiled_K, tiled_N);
    tl_tiled_R.stride = ctx.matrix_lmem_default_stride(tl_tiled_R.shape);
    required_size += tl_tiled_R.shape.n * tl_tiled_R.stride.n;

    // tiled B, 16bit
    if (have_bias) {
      tl_tiled_B.start_address = required_size;
      required_size += tl_tiled_B.shape.n * tl_tiled_B.stride.n;
    }

    if (required_size <= lane_size) {
      DEBUG_BMNET(llvm::errs() << llvm::format("  tiled_Y %d, tiled_L %d, tiled_R %d, tiled_B %d\n",
                                            4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n,
                                            tl_tiled_L.shape.n * tl_tiled_L.stride.n,
                                            tl_tiled_R.shape.n * tl_tiled_R.stride.n,
                                            2 * tl_tiled_B.shape.n * tl_tiled_B.stride.n));

      break;
    }

  } while (--tiled_K);

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "  tiled_Y(%d, %d) = tiled_L(%d, %d) * tiled_R(%d, %d) + tiled_B(%d, %d)\n"
                  "  required_size %d kB\n",
                  M, tiled_N, M, tiled_K, tiled_K, tiled_N, 1, tiled_N, required_size / 1024));

  DEBUG_BMNET(llvm::errs() << llvm::format(
                  "  tiled_Y shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_L shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_R shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n"
                  "  tiled_B shape (n=%d, c=%d, w=%d, col=%d), stride(n=%d, c=%d, h=%d)\n",
                  tl_tiled_Y.shape.n, tl_tiled_Y.shape.c, tl_tiled_Y.shape.w, tl_tiled_Y.shape.col,
                  tl_tiled_Y.stride.n, tl_tiled_Y.stride.c, tl_tiled_Y.stride.h, tl_tiled_L.shape.n,
                  tl_tiled_L.shape.c, tl_tiled_L.shape.w, tl_tiled_L.shape.col, tl_tiled_L.stride.n,
                  tl_tiled_L.stride.c, tl_tiled_L.stride.h, tl_tiled_R.shape.n, tl_tiled_R.shape.c,
                  tl_tiled_R.shape.w, tl_tiled_R.shape.col, tl_tiled_R.stride.n,
                  tl_tiled_R.stride.c, tl_tiled_R.stride.h, tl_tiled_B.shape.n, tl_tiled_B.shape.c,
                  tl_tiled_B.shape.w, tl_tiled_B.shape.col, tl_tiled_B.stride.n,
                  tl_tiled_B.stride.c, tl_tiled_B.stride.h));

  ASSERT(tiled_K);
  if (!tiled_K) {
    return;
  }

  // Each tiled_R(weight) is only loaded once.
  // tiled_L(input) reload is reload once tiled_weight moves right.
  //
  // for each tiled N
  for (u32 offset_N = 0; offset_N < N; offset_N += tiled_N) {
    // Y = [Y0, Y1, ... Yn-1]

    // Actual width
    u32 width_N = ((offset_N + tiled_N) <= N) ? tiled_N : (N - offset_N);

    // for each tiled K
    for (u32 offset_K = 0; offset_K < K; offset_K += tiled_K) {
      // Y(M, K) = L(M, K) * R(K, N) + B(1, N)
      // tiled_Y(M, tiled_K) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) + tiled_B(1, tiled_N)
      //
      // L = [L0, L1, ... Lk-1]
      // R = [R0,0,   R0,1,   ..., R0,n-1
      //      R1,0,
      //
      //      Rk-1,0, Rk-1,1, ..., Rk-1,n-1]
      // B = [B0, B1, ... Bn-1]
      //
      // tiled_y,i += L0 * R0,i + L1 * R1,i + ... + Ln-1 * Rk-1,i + Bi

      // Actual width
      u32 width_K = ((offset_K + tiled_K) <= K) ? tiled_K : (K - offset_K);

      required_size = 0;  // Start of LMEM

      // tiled Y, 32bit
      tl_tiled_Y.start_address = required_size;
      tl_tiled_Y.shape = ctx.matrix_lmem_default_shape(M, width_N);  // actual width
      tl_tiled_Y.stride = ctx.matrix_lmem_default_stride(tl_tiled_Y.shape);
      required_size += 4 * tl_tiled_Y.shape.n * tl_tiled_Y.stride.n;

      // Load tiled L from global memory, input
      tl_tiled_L.start_address = required_size;
      tl_tiled_L.shape = ctx.matrix_lmem_default_shape(M, width_K);  // actual width
      tl_tiled_L.stride = ctx.matrix_lmem_default_stride(tl_tiled_L.shape);
      required_size += tl_tiled_L.shape.n * tl_tiled_L.stride.n;
      ctx.tdma_load_stride(&tl_tiled_L, global_offset_bottom_data + offset_K,
                           {K},  // original column width
                           CTRL_NEURON);

      // Load tiled R from global memory, weight
      tl_tiled_R.start_address = required_size;
      tl_tiled_R.shape = ctx.matrix_lmem_default_shape(width_K, width_N);  // actual width
      tl_tiled_R.stride = ctx.matrix_lmem_default_stride(tl_tiled_R.shape);
      required_size += tl_tiled_R.shape.n * tl_tiled_R.stride.n;
      if (weight_tp) {
        ctx.tdma_load_stride(&tl_tiled_R, global_offset_weight_data + offset_N * K + offset_K,
                             {K},  // original transposed column width
                             CTRL_WEIGHT | CTRL_TP);
      } else {
        ctx.tdma_load_stride(&tl_tiled_R, global_offset_weight_data + offset_K * N + offset_N,
                             {N},  // original column width
                             CTRL_WEIGHT);
      }

      // Load tiled B from gobale meory at last time, bias
      // we need tempory shape to load lower 8bit and upper 8bit
      bool is_last_tile = ((offset_K + tiled_K) >= K) ? true : false;
      bool B_needed = (is_last_tile && have_bias) ? true : false;
      if (B_needed) {
        tl_tiled_B.start_address = required_size;

#if 1
        // Follow current bmk restrict
        tl_tiled_B.shape = ctx.matrix_lmem_default_shape(2, width_N);  // actual width
        tl_tiled_B.stride = ctx.matrix_lmem_default_stride(tl_tiled_B.shape);

        ctx.tdma_load_stride(&tl_tiled_B, global_offset_bias_data + offset_N,
                             {N},  // original column width
                             CTRL_WEIGHT);
#else
        // TIU
        tl_tiled_B.shape = ctx.matrix_lmem_default_shape(1, width_N);  // actual width
        bmk1880v2_matrix_lmem_t tl_load_B = tl_tiled_B;
        tl_load_B.shape = ctx.matrix_lmem_default_shape(2, width_N);  // actual width
        tl_load_B.stride = ctx.matrix_lmem_default_stride(tl_load_B.shape);
        ctx.tdma_load_stride(&tl_load_B, global_offset_bias_data + offset_N,
                             {N},  // original column width
                             CTRL_WEIGHT);
#endif
      }

      u32 ps32_mode = 0;    // normal mode
      u32 relu_enable = 0;  // 1880v2 relu can be used in ps32_mode
      if (tiled_K < K) {
        if (offset_K == 0) {        // first tile
          ps32_mode = 2;            // write 32b result at the first time
        } else if (is_last_tile) {  // last tile
          ps32_mode = 1;            // load previous 32-bit result
        } else {
          ps32_mode = 3;  // init & write 32bits partial sum
        }
      }

      // No tiling or last tile
      if ((ps32_mode == 0 || ps32_mode == 1) && do_activation && activation_method == RELU) {
        relu_enable = 1;
      }

      {
        bmk1880v2_tiu_matrix_multiplication_param_t p;
        p.res = &tl_tiled_Y;
        p.left = &tl_tiled_L;
        p.right = &tl_tiled_R;
        p.bias = B_needed ? &tl_tiled_B : nullptr;
        p.lshift_bits = 0;                                     // deprecated
        p.rshift_bits = is_last_tile ? right_shift_width : 0;  // quantization down
        p.res_is_int8 = is_last_tile ? 1 : 0;                  // output 8bit
        p.add_result = 0;                                      // deprecated
        p.relu_enable = relu_enable;
        p.ps32_mode = ps32_mode;
        p.relu_res_is_sign = relu_enable ? 1 : 0;  // [0, 127], 7/29/2019 H/W relu design
        p.layer_id = layer_id;
        p.sw_op_info = offset_N;

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "  offset_N %d, offset_K %d, L(%d, %d), R(%d, %d)\n", offset_N, offset_K,
                        p.left->shape.n, p.left->shape.col, p.right->shape.n, p.right->shape.col));

        matrix_multiplication(ctx, p);
      }

      // Integrate activation method later
      // Store tiled_Y to global memory, handle activation later
      if (is_last_tile) {
        ctx.tdma_store_stride(&tl_tiled_Y, global_offset_top_data + offset_N,
                              {N},  // original column width
                              CTRL_NEURON);
      }

    }  // for (u32 offset_K = 0; offset_K < K; offset_K += tiled_K)
  }    // for (u32 offst_N = 0; offset_N < N; offset_N += tiled_N)

  if (do_activation && activation_method == RELU) {
#if 1
  // do nothing
#else
    ASSERT((M * N) <= total_mem_size);

    // Load Y from global memory
    bmk1880v2_matrix_lmem_t ml_top;
    ml_top.start_address = 0;  // Start of LMEM
    ml_top.fmt = FMT_I8;
    ml_top.shape = ctx.matrix_lmem_default_shape(M, N);
    ml_top.stride = ctx.matrix_lmem_default_stride(ml_top.shape);
    ctx.tdma_load(&ml_top, global_offset_top_data, CTRL_NEURON);

    // Convert to tensor format
    bmk1880v2_tensor_lmem_t tl_top;
    tl_top.start_address = ml_top.start_address;
    tl_top.fmt = FMT_I8;
    tl_top.shape = {ml_top.shape.n, ml_top.shape.c, 1, ml_top.shape.w};
    tl_top.stride = {ml_top.stride.n, ml_top.stride.c, ml_top.stride.h, 1};

    bmk1880v2_tiu_element_wise_max_param_t p;
    p.a = &tl_top;
    p.max = &tl_top;
    p.b_val = 0;
    p.b_is_const = 1;
    p.b_is_signed = 1;
    ctx.tiu_element_wise_max(&p);

    // Store back to global memory
    ctx.tdma_store(&ml_top, global_offset_top_data, CTRL_NEURON);
#endif
  } else if (do_activation && activation_method == PRELU) {
    // TODO
    ASSERT(0);
  }
}

static void fc_forward_bmkernel_in_node(
    const BM1880v2BackendContext &ctx, u32 layer_id, gaddr_t global_offset_bottom_data,
    gaddr_t global_offset_weight_data, gaddr_t global_offset_bias_data,
    gaddr_t global_offset_top_data, int input_row_num, int input_col_num, int weight_col_num,
    int have_bias, int do_activation, int activation_method, gaddr_t activation_ga_slope,
    int activation_channel_shared, int activation_gt_scale, int activation_gt_rshift,
    int activation_le_scale, int activation_le_rshift,
    int W_param,  // 1880v2 use 2*EU_NUM
    bool weight_tp, int left_shift_width, int right_shift_width) {
  int slice_num[3] = {1, 1, 1};
  int weight_bank_num;
  if (do_activation && activation_method == PRELU) {
    weight_bank_num = 1;
  } else {
    weight_bank_num = 5;
  }
    // FIXME:chieh-jay We should enable it later
#if 0
  if (OptUseLessLossFC || OptForceLessLossFC) {
    // Try lossless method
    bool ret = fc_forward_serial_slice_weight_col_num(
        ctx, global_offset_bottom_data, global_offset_weight_data, global_offset_bias_data,
        global_offset_top_data, input_row_num, input_col_num, weight_col_num, have_bias,
        do_activation, activation_method, activation_ga_slope, activation_channel_shared,
        activation_gt_scale, activation_gt_rshift, activation_le_scale, activation_le_rshift,
        weight_tp, left_shift_width, right_shift_width);

    if (ret) {
      return;
    }
    if (OptForceLessLossFC) {
      assert(false && "Not support by Less-loss FC");
      exit(-1);
    }
  }
#endif
  int method_selection = method_selection_fc_forward_parallel(
      ctx, input_row_num, input_col_num, weight_col_num, W_param, weight_bank_num, slice_num);
  DEBUG_BMNET(llvm::errs() << "    fc_fwd_parallel slice_num_0: " << slice_num[0]
                        << ", slice_num_1: " << slice_num[1] << ", slice_num_2: " << slice_num[2]
                        << ", W_param: " << W_param << "\n");

  ASSERT(slice_num[0] <= input_row_num && slice_num[0] >= 1);
  ASSERT(slice_num[1] <= input_col_num && slice_num[1] >= 1);
  ASSERT(slice_num[2] <= weight_col_num && slice_num[2] >= 1);

  DEBUG_BMNET(llvm::errs() << llvm::format("    method_selection %d\n", method_selection));
  LOG(INFO) << "method selected: " << method_selection << "\n";
  switch (method_selection) {
    case 0:  // one-way slicing on input_col_num
    {
      DEBUG_BMNET(llvm::errs() << "    0: slice input col slice_num_0: " << slice_num[0]
                            << ", slice_num_1: " << slice_num[1]
                            << ", slice_num_2: " << slice_num[2] << "\n");
      fc_forward_parallel_slice_input_col_num(
          ctx, layer_id, global_offset_bottom_data, global_offset_weight_data,
          global_offset_bias_data, global_offset_top_data, input_row_num, input_col_num,
          weight_col_num, have_bias, do_activation, activation_method, activation_ga_slope,
          activation_channel_shared, activation_gt_scale, activation_gt_rshift, activation_le_scale,
          activation_le_rshift, W_param, slice_num[1], weight_tp, left_shift_width,
          right_shift_width);
    } break;

    case 1:  // one-way slicing on input_row_num
    {
      DEBUG_BMNET(llvm::errs() << "    1: slice input row slice_num_0: " << slice_num[0]
                            << ", slice_num_1: " << slice_num[1]
                            << ", slice_num_2: " << slice_num[2] << "\n");
      ASSERT(weight_tp);  // TODO(wwcai)
      ASSERT(0);          // TODO(wwcai)
    } break;

    case 2:  // one-way slicing on weight_col_num
    {
      DEBUG_BMNET(llvm::errs() << "    2: slice input row slice_num_0: " << slice_num[0]
                            << ", slice_num_1: " << slice_num[1]
                            << ", slice_num_2: " << slice_num[2] << "\n");
      ASSERT(weight_tp);  // TODO(wwcai)
      ASSERT(0);          // TODO(wwcai)
    } break;

    case 3:  // multi-dimension slicing
    {
      fc_forward_parallel_get_slice_num_multidim(ctx, input_row_num, input_col_num, weight_col_num,
                                                 W_param, slice_num);
      DEBUG_BMNET(llvm::errs() << "    3: multi slice slice_num_0: " << slice_num[0]
                            << ", slice_num_1: " << slice_num[1]
                            << ", slice_num_2: " << slice_num[2] << "\n");
      ASSERT(weight_tp == 0);  // TODO(wwcai)
      fc_forward_parallel_slicing_multi_dimension(
          ctx, layer_id, global_offset_bottom_data, global_offset_weight_data,
          global_offset_bias_data, global_offset_top_data, input_row_num, input_col_num,
          weight_col_num, have_bias, do_activation, activation_method, W_param, slice_num,
          left_shift_width, right_shift_width);
    } break;

    default:
      break;
  }
}

void bmnet_fc_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx, u32 stream_id, u32 inst_id, u32 layer_id, const u32 *depends,
    u32 depends_len, gaddr_t bottom_data_gaddr, gaddr_t weight_data_gaddr, gaddr_t bias_data_gaddr,
    gaddr_t top_data_gaddr, int in_row, int in_col, int out_col, int have_bias, int do_activation,
    int activation_method, gaddr_t activation_ga_slope, int activation_channel_shared,
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, bool weight_tp, int left_shift_width, int right_shift_width) {
  DEBUG_BMNET(llvm::errs() << llvm::format("bmnet_fc_fixed_forward_bmkernel\n"
                                        "    in (%d, %d), out (%d), has_bias %d, do_activation %d, "
                                        "activation_method %d, weight_tp %d\n",
                                        in_row, in_col, out_col, have_bias, do_activation,
                                        activation_method, weight_tp));

  fc_slicing_multi_dimention(ctx, layer_id, bottom_data_gaddr, weight_data_gaddr, bias_data_gaddr,
                             top_data_gaddr, in_row, in_col, out_col, have_bias, do_activation,
                             activation_method, activation_ga_slope, activation_channel_shared,
                             activation_gt_scale, activation_gt_rshift, activation_le_scale,
                             activation_le_rshift, weight_tp, left_shift_width, right_shift_width);
#if 0
  // choose node_id as current active node
  fc_forward_bmkernel_in_node(ctx, bottom_data_gaddr, weight_data_gaddr, bias_data_gaddr,
                              top_data_gaddr, in_row, in_col, out_col, have_bias, do_activation,
                              activation_method, activation_ga_slope, activation_channel_shared,
                              activation_gt_scale, activation_gt_rshift, activation_le_scale,
                              activation_le_rshift, EU_NUM * 2, weight_tp,
                              left_shift_width,  // TODO(wwcai)
                              right_shift_width);
#endif
}

//}  // namespace bmnet
