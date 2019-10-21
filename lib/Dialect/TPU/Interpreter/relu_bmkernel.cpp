/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <map>
#include <vector>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_relu"

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

//namespace {

//using bmnet::BM1880v2BackendContext;

class Bank {
 public:
  int bank_id;
  int bank_num;
  bmk1880v2_matrix_lmem_t lmem;
  bool is_master;

  Bank(const BM1880v2BackendContext &ctx, int id, bmk1880v2_matrix_lmem_shape_t shape, fmt_t fmt,
       int number, bool master = true)
      : ctx_(ctx), bank_id(id), bank_num(number), is_master(master) {
    // lmem = ctx_.tl_alloc_bank(bank_id % bank_num, shape, fmt, CTRL_AL);

    ASSERT(number <= ctx.hw.local_mem_banks);

    int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    lmem.start_address = bank_size * (bank_id % bank_num);
    lmem.fmt = fmt;
    lmem.shape = shape;
    lmem.stride = ctx.matrix_lmem_default_stride(lmem.shape);
  }

  ~Bank() = default;

 private:
  const BM1880v2BackendContext &ctx_;
};

class ParallelBanks {
 public:
  int bank_num;

  ParallelBanks(const BM1880v2BackendContext &ctx, int number) : ctx_(ctx), bank_num(number) {
    for (int i = 0; i < bank_num; i++) {
      free_banks.push_back(i);
    }
  }

  Bank *alloc(bmk1880v2_matrix_lmem_shape_t shape, fmt_t fmt) {
    ASSERT(!this->free_banks.empty());
    int bank_id = this->free_banks.back();
    Bank *bk = new Bank(ctx_, bank_id, shape, fmt, bank_num);
    this->free_banks.pop_back();
    return bk;
  }

  Bank *alloc(Bank *master, bmk1880v2_matrix_lmem_shape_t shape, fmt_t fmt) {
    int bank_id = master->bank_id;
    Bank *bk = new Bank(ctx_, bank_id, shape, fmt, false);
    return bk;
  }

  void free(Bank *bk) {
    if (!bk) {
      return;
    }

    if (bk->is_master) {
      free_banks.push_back(bk->bank_id);
    }
    delete bk;
  }

 private:
  std::vector<int> free_banks;
  const BM1880v2BackendContext &ctx_;
};

//}  // namespace

//namespace bmnet {

static int __get_lmem_usage(const BM1880v2BackendContext &ctx, int count) {
  int w = EU_NUM;
  int c = ceiling_func(count, w);
  int aligned_csize = get_csize_local(ctx, 1, w);
  int c_stride = ceiling_func_shift(c, NPU_SHIFT);
  int lmem_usage = c_stride * aligned_csize;
  return lmem_usage;
}

static int _find_best_slice(const BM1880v2BackendContext &ctx, int blob_num, int count_per_blob) {
  int slice_num;
  int avail_bank_size;
  int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
  int lmem_needed;
  int count_per_slice;

  if (blob_num == 2) {
    avail_bank_size = bank_size;
  } else {
    avail_bank_size = bank_size / 4;
  }

  lmem_needed = __get_lmem_usage(ctx, count_per_blob);
  slice_num = ceiling_func(lmem_needed, avail_bank_size);
  count_per_slice = ALIGN(ceiling_func(count_per_blob, slice_num), EU_NUM * NPU_NUM);

  while (__get_lmem_usage(ctx, count_per_slice) > avail_bank_size || count_per_slice > 65535) {
    slice_num++;
    count_per_slice = ALIGN(ceiling_func(count_per_blob, slice_num), EU_NUM * NPU_NUM);
  }

  return slice_num;
}

/*
  relu_forward_32bit implementation
*/
static void _forward_internel(const BM1880v2BackendContext &ctx, u32 layer_id, u64 bottom_gaddr,
                              u64 top_gaddr, float negative_slope, int input_n, int input_c,
                              int input_h, int input_w, int threshold_x_quantized_len,
                              const int *threshold_x_quantized, const int *right_shift_array) {
  int count = input_n * input_c * input_h * input_w;
  int slice_num = _find_best_slice(ctx, 4, count);

  u64 slice_bottom_gaddr = bottom_gaddr;
  u64 slice_top_gaddr = top_gaddr;

  ParallelBanks *pbanks = new ParallelBanks(ctx, ctx.hw.local_mem_banks);

  int count_per_slice, prev_count_per_slice;

  if (slice_num == 1) {
    count_per_slice = count;
    prev_count_per_slice = count;
  } else {
    count_per_slice = ALIGN(ceiling_func(count, slice_num), EU_NUM * NPU_NUM);
    prev_count_per_slice = count_per_slice;
  }

  LLVM_DEBUG(
    llvm::errs() << llvm::format(
        "forward, count:%d, <%d, %d, %d, %d>, slice_num:%d, count_per_slice:%d\n",
        count, input_n, input_c, input_h, input_w, slice_num, count_per_slice);
  );

  // create constants
  // tensor_lmem *float_zero = bmk1682_tl_alloc_const(0);
  // tensor_lmem *slope = bmk1682_tl_alloc_const(negative_slope);

  bmk1880v2_matrix_lmem_shape_t l_shape = ctx.matrix_lmem_shape_t1(count_per_slice);

  Bank *bottom_preload = pbanks->alloc(l_shape, FMT_I8);
  Bank *top_prev = nullptr;

  DEBUG_BMNET(llvm::errs() << llvm::format("---: slice_bottom_gaddr 0x%lx slice_num:%d\n",
                                        slice_bottom_gaddr, slice_num));

  // Load with matrix format
  ctx.tdma_load(&bottom_preload->lmem, slice_bottom_gaddr, CTRL_NEURON);

  slice_bottom_gaddr += count_per_slice;

  for (int slice_idx = 0; slice_idx <= slice_num; slice_idx++) {
    Bank *bottom = bottom_preload;
    bottom_preload = nullptr;

    // if has top data of previous round, to store it into global memory
    if (top_prev) {
      ctx.tdma_store(&top_prev->lmem, slice_top_gaddr, CTRL_NEURON);
      slice_top_gaddr += prev_count_per_slice;

      pbanks->free(top_prev);
      top_prev = nullptr;

      if (slice_idx == slice_num) {
        break;
      }
    }

    LLVM_DEBUG(
      llvm::errs() << llvm::format(
          "count_per_slice:%d, slice_idx:%d,  slice_num:%d\n",
          count_per_slice, slice_idx, slice_num);
    );

    ctx.parallel_enable();

    Bank *top = pbanks->alloc(l_shape, FMT_I8);

    // thread 0. calculate gradients
    {
      // Covert to tensor format
      bmk1880v2_tensor_lmem_t tl_bottom;
      tl_bottom.start_address = bottom->lmem.start_address;
      tl_bottom.fmt = bottom->lmem.fmt;
      tl_bottom.shape = {bottom->lmem.shape.n, bottom->lmem.shape.c, 1, bottom->lmem.shape.w};
      tl_bottom.stride = {bottom->lmem.stride.n, bottom->lmem.stride.c, bottom->lmem.stride.h, 1};

      bmk1880v2_tensor_lmem_t tl_top;
      tl_top.start_address = top->lmem.start_address;
      tl_top.fmt = top->lmem.fmt;
      tl_top.shape = {top->lmem.shape.n, top->lmem.shape.c, 1, top->lmem.shape.w};
      tl_top.stride = {top->lmem.stride.n, top->lmem.stride.c, top->lmem.stride.h, 1};

      bmk1880v2_tiu_element_wise_max_param_t p13;
      p13.max = &tl_top;
      p13.a = &tl_bottom;
      p13.b_is_const = 1;
      p13.b_is_signed = 1;
      p13.b_val = 0;
      p13.layer_id = layer_id;
      ctx.tiu_element_wise_max(&p13);
    }

    // thread 1. preload relu and top_diff data for next round.
    if (slice_idx != slice_num - 1) {
      int current = (slice_idx + 1) * count_per_slice;

      count_per_slice = count - current <= count_per_slice ? count - current : count_per_slice;
      l_shape = ctx.matrix_lmem_shape_t1(count_per_slice);

      bottom_preload = pbanks->alloc(l_shape, FMT_I8);

      ctx.tdma_load(&bottom_preload->lmem, slice_bottom_gaddr, CTRL_NEURON);

      slice_bottom_gaddr += count_per_slice;
    }

    ctx.parallel_disable();

    pbanks->free(bottom);
    bottom = nullptr;

    top_prev = top;
    top = nullptr;
  }

  // ctx.tl_free(slope);
  // ctx.tl_free(float_zero);

  delete pbanks;
}

void bmnet_relu_fixed_forward_bmkernel(const BM1880v2BackendContext &ctx, u32 stream_id,
                                       u32 inst_id, u32 layer_id, const u32 *depends,
                                       u32 depends_len, u64 bottom_gaddr, u64 top_gaddr,
                                       float negative_slope, int input_n, int input_c, int input_h,
                                       int input_w, int threshold_x_quantized_len,
                                       const int *threshold_x_quantized,
                                       const int *right_shift_array) {
  // for (int i = 0; i < threshold_x_quantized_len; i++) {
  //  VLOG(3) << "threshold_x_quantized/right_shift_array[" << i << "]:" << threshold_x_quantized[i]
  //          << "/" << right_shift_array[i];
  //}
  _forward_internel(ctx, layer_id, bottom_gaddr, top_gaddr, negative_slope, input_n, input_c,
                    input_h, input_w, threshold_x_quantized_len, threshold_x_quantized,
                    right_shift_array);
}

#if 1
//
// Output: bottom
//
static void tl_leaky_relu(const BM1880v2BackendContext &ctx, u32 layer_id,
                          bmk1880v2_tensor_lmem_t &bottom, bmk1880v2_tensor_lmem_t &relu,
                          bmk1880v2_tensor_lmem_t &neg, int GT_right_shift_width,
                          int LE_right_shift_width, int GT_scale, int LE_scale) {
  // 0. relu = relu(bottom)
  bmk1880v2_tiu_element_wise_max_param_t p13;
  p13.max = &relu;
  p13.a = &bottom;
  p13.b_is_const = 1;
  p13.b_is_signed = 1;
  p13.b_val = 0;
  p13.layer_id = layer_id;
  ctx.tiu_element_wise_max(&p13);

  // 1. relu = (relu * GT_scale) >> GT_right_shift_width
  bmk1880v2_tiu_element_wise_mul_param_t p;
  p.res_high = nullptr;
  p.res_low = &relu;
  p.a = &relu;
  p.b_val = GT_scale;
  p.b_is_signed = true;
  p.b_is_const = 1;
  p.rshift_bits = GT_right_shift_width;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_element_wise_mul(&p);

  // 2. neg = neg(0, botom)
  bmk1880v2_tiu_element_wise_min_param_t p7;
  p7.min = &neg;
  p7.a = &bottom;
  p7.b_is_const = 1;
  p7.b_val = 0;
  p7.b_is_signed = 1;
  p7.layer_id = layer_id;
  ctx.tiu_element_wise_min(&p7);

  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope) >> LE_right_shift_width
  p.res_high = nullptr;
  p.res_low = &neg;
  p.a = &neg;
  p.b_val = LE_scale;
  p.b_is_signed = true;
  p.b_is_const = 1;
  p.rshift_bits = LE_right_shift_width;
  p.layer_id = layer_id;
  ctx.tiu_element_wise_mul(&p);

  // 4. bottom = or relu, neg
  bmk1880v2_tiu_element_wise_or_int8_param_t p9;
  p9.res = &bottom;
  p9.a = &relu;
  p9.b = &neg;
  p9.layer_id = layer_id;
  ctx.tiu_element_wise_or_int8(&p9);
}

void bmnet_leakyrelu_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx, u32 stream_id, u32 inst_id, u32 layer_id, const u32 *depends,
    u32 depends_len, u64 input_gaddr, u64 output_gaddr, int input_n, int input_c, int input_h,
    int input_w, int GT_right_shift_width, int LE_right_shift_width, int GT_scale, int LE_scale,
    int threshold_x_quantized_len, const int *threshold_x_quantized, const int *right_shift_array) {
  DEBUG_BMNET(llvm::errs() << llvm::format("bmnet_leakyrelu_fixed_forward_bmkernel:\n"
                                        "  layer_id %d\n"
                                        "  input_gddr: %lx, output_gaddr: %lx\n"
                                        "  input (%d, %d, %d, %d)\n"
                                        "  GT_scale:%d, LE_scale:%d\n"
                                        "  GT_right_shift_width:%d, LE_right_shift_width:%d\n",
                                        layer_id, input_gaddr, output_gaddr, input_n, input_c,
                                        input_h, input_w, GT_scale, LE_scale, GT_right_shift_width,
                                        LE_right_shift_width));

  for (int i = 0; i < threshold_x_quantized_len; i++) {
    llvm::errs() << "threshold_x_quantized/right_shift_array[" << i << "]:" << threshold_x_quantized[i]
            << "/" << right_shift_array[i];
  }

  // Split input based on local memory
  u32 total_eu = ctx.hw.npu_num * ctx.hw.eu_num;
  u32 lane_size = ctx.hw.local_mem_size;
  u32 total_mem_size = ctx.hw.npu_num * ctx.hw.local_mem_size;
  u32 max_N = (1 << 12) - 1;  // 1880v2: 12 bit
  u32 count = input_n * input_c * input_h * input_w;
  u32 tiled_N = count / total_eu / 3;  // 3 blobs
  tiled_N = (tiled_N > max_N) ? max_N : tiled_N;

  // local tensor shape(tiled_N, npu_num, 1, eu_num)
  bmk1880v2_tensor_lmem_shape_t tl_shape = {tiled_N, static_cast<u32>(ctx.hw.npu_num), 1,
                                            static_cast<u32>(ctx.hw.eu_num)};
  bmk1880v2_tensor_lmem_stride_t tl_stride =
      ctx.tensor_lmem_default_stride(tl_shape, /*eu_align=*/1);

  // Find max tiled_N
  u32 required_size = 0;
  do {
    tl_shape.n = tiled_N;
    tl_stride = ctx.tensor_lmem_default_stride(tl_shape, /*eu_align=*/1);
    required_size = 3 * tl_shape.n * tl_stride.n;  // 3 blobs

    if (required_size <= lane_size) {
      break;
    }

  } while (--tiled_N);

  DEBUG_BMNET(
      llvm::errs() << llvm::format("  tiled_bottom shape (%d, %d, %d, %d), stride (%d, %d, %d, %d)\n"
                                "  required_size %d kB/lane\n",
                                tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w, tl_stride.n,
                                tl_stride.c, tl_stride.h, tl_stride.w, required_size / 1024));

  ASSERT(tiled_N);
  if (!tiled_N) {
    return;
  }

  // Tiled local memory layout:
  //   tiled bottom/result
  //   tiled relu
  //   tiled neg

  // Tiled bottom
  required_size /= 3;  // for 3 blobs
  bmk1880v2_tensor_lmem_t tl_tiled_bottom;
  tl_tiled_bottom.start_address = 0;
  tl_tiled_bottom.fmt = FMT_I8;
  tl_tiled_bottom.shape = tl_shape;
  tl_tiled_bottom.stride = tl_stride;

  // Tiled relu
  bmk1880v2_tensor_lmem_t tl_tiled_relu = tl_tiled_bottom;
  tl_tiled_relu.start_address = tl_tiled_bottom.start_address + required_size;

  // Tiled neg
  bmk1880v2_tensor_lmem_t tl_tiled_neg = tl_tiled_bottom;
  tl_tiled_neg.start_address = tl_tiled_relu.start_address + required_size;

  // In unit of tiled_N * npu_num * eu_num
  u32 global_input_offset = 0;
  for (u32 i = 0; i < (count / total_eu / tiled_N); i++) {
    // Load as a chunk of contiguous memory in global memory, not use global shape/stride
    // Local memory use tensor shape to maximize eu utilization.

    DEBUG_BMNET(llvm::errs() << llvm::format("  [%d] tdma load: gaddr 0x%x+0x%x\n", i, input_gaddr,
                                          global_input_offset));

    ctx.tdma_load(&tl_tiled_bottom, input_gaddr + global_input_offset, CTRL_NEURON);

    tl_leaky_relu(ctx, layer_id, tl_tiled_bottom, tl_tiled_relu, tl_tiled_neg, GT_right_shift_width,
                  LE_right_shift_width, GT_scale, LE_scale);

    // Store bottom as a chunk of contiguous memory, not use global shape/stride
    ctx.tdma_store(&tl_tiled_bottom, output_gaddr + global_input_offset, CTRL_NEURON);

    // Next input offset
    global_input_offset += tiled_N * total_eu;

  }  // for (u32 i = 0; i < (count/total_eu/tiled_N); i++)

  // Remaining count, in unit of npu_num * eu_num
  if (global_input_offset < count) {
    tiled_N = (count - global_input_offset) / total_eu;

    // Update shape, stride
    tl_shape.n = tiled_N;
    tl_stride = ctx.tensor_lmem_default_stride(tl_shape, /*eu_align=*/1);
    required_size = tl_shape.n * tl_stride.n;

    DEBUG_BMNET(
        llvm::errs() << llvm::format("  tiled_bottom shape (%d, %d, %d, %d), stride (%d, %d, %d, %d)\n"
                                  "  required_size %d kB/lane\n",
                                  tl_shape.n, tl_shape.c, tl_shape.h, tl_shape.w, tl_stride.n,
                                  tl_stride.c, tl_stride.h, tl_stride.w, required_size / 1024));

    // Tiled bottom
    tl_tiled_bottom.shape = tl_shape;
    tl_tiled_bottom.stride = tl_stride;

    // Tiled relu
    tl_tiled_relu = tl_tiled_bottom;
    tl_tiled_relu.start_address = tl_tiled_bottom.start_address + required_size;

    // Tiled neg
    tl_tiled_neg = tl_tiled_bottom;
    tl_tiled_neg.start_address = tl_tiled_relu.start_address + required_size;

    // Load as a chunk of contiguous memory in global memory, not use global shape/stride
    // Local memory use tensor shape to maximize eu utilization.
    DEBUG_BMNET(llvm::errs() << llvm::format("  tdma load: gaddr 0x%x+0x%x\n", input_gaddr,
                                          global_input_offset));

    ctx.tdma_load(&tl_tiled_bottom, input_gaddr + global_input_offset, CTRL_NEURON);

    tl_leaky_relu(ctx, layer_id, tl_tiled_bottom, tl_tiled_relu, tl_tiled_neg, GT_right_shift_width,
                  LE_right_shift_width, GT_scale, LE_scale);

    // Store bottom as a chunk of contiguous memory, not use global shape/stride
    ctx.tdma_store(&tl_tiled_bottom, output_gaddr + global_input_offset, CTRL_NEURON);

    global_input_offset += tiled_N * total_eu;
  }

  // Remaining count, in unit of eu_num
  if (global_input_offset < count) {
    ASSERT(0);
  }
}
#else
void bmnet_leakyrelu_fixed_forward_bmkernel(const BM1880v2BackendContext &ctx, u32 stream_id,
                                            u32 inst_id, const u32 *depends, u32 depends_len,
                                            u64 input_gaddr, u64 output_gaddr, int input_n,
                                            int input_c, int input_h, int input_w,
                                            int GT_right_shift_width, int LE_right_shift_width,
                                            int GT_scale, int LE_scale) {
  DEBUG_BMNET(llvm::errs() << "bmnet_leakyrelu_fixed_forward_bmkernel"
                        << llvm::format(": "
                                        "input_gddr: %lx, output_gaddr:%lx\n"
                                        "GT_scale:%d, LE_scale:%d\n"
                                        "GT_right_shift_width:%d, LE_right_shift_width:%d\n"
                                        "input_n:%d input_c:%d input_h:%d input_w:%d\n",
                                        input_gaddr, output_gaddr, GT_scale, LE_scale,
                                        GT_right_shift_width, LE_right_shift_width, input_n,
                                        input_c, input_h, input_w));
  input_n = ceiling_func_shift(input_n, NODECHIP_SHIFT);
  int count = input_n * input_c * input_h * input_w;
  int slice_num = get_slice_num_element_wise(ctx, 3, count + 1);
  gaddr_t slice_bottom_gaddr = input_gaddr;
  gaddr_t slice_top_gaddr = output_gaddr;

  for (int slice_idx = 0; slice_idx < slice_num; slice_idx++) {
    int count_sec = count / slice_num + (slice_idx < count % slice_num);

    // load with matrix format
    // set shape
    bmk1880v2_matrix_lmem_shape_t input_shape = ctx.matrix_lmem_shape_t1(count_sec);
    bmk1880v2_matrix_lmem_t *tl_bottom =
        ctx.lmem_alloc_matrix(input_shape, FMT_I8, 1);                                 // EU-aligned
    bmk1880v2_matrix_lmem_t *tl_relu = ctx.lmem_alloc_matrix(input_shape, FMT_I8, 1);  // EU-aligned
    bmk1880v2_matrix_lmem_t *tl_neg = ctx.lmem_alloc_matrix(input_shape, FMT_I8, 1);   // EU-aligned

    // Global memory from reshaped local memory
    bmk1880v2_matrix_tgmem_t ts_bottom;
    ts_bottom.base_reg_index = BM1880v2BackendContext::NEURON_MEMORY;
    ts_bottom.start_address = slice_bottom_gaddr;
    ts_bottom.shape = {static_cast<u32>(input_shape.n), static_cast<u32>(input_shape.col)};
    ts_bottom.stride = {static_cast<u32>(input_shape.col)};

    DEBUG_BMNET(llvm::errs() << llvm::format("    [%d] tdma_g2l_tensor_copy: bottom\n"
                                          "      shape(n=%d, c=%d, w=%d, col=%d)\n",
                                          slice_idx, tl_bottom->shape.n, tl_bottom->shape.c,
                                          tl_bottom->shape.w, tl_bottom->shape.col));

    bmk1880v2_tdma_tg2l_matrix_copy_param_t p1;
    p1.src = &ts_bottom;
    p1.dst = tl_bottom;
    ctx.tdma_g2l_matrix_copy(&p1);

    // Convert to tensor format
    bmk1880v2_tensor_lmem_t bottom;
    bottom.start_address = tl_bottom->start_address;
    bottom.fmt = tl_bottom->fmt;
    bottom.shape = {tl_bottom->shape.n, tl_bottom->shape.c, 1, tl_bottom->shape.w};
    bottom.stride = {tl_bottom->stride.n, tl_bottom->stride.c, tl_bottom->stride.h, 1};

    bmk1880v2_tensor_lmem_t relu;
    relu.start_address = tl_relu->start_address;
    relu.fmt = tl_relu->fmt;
    relu.shape = {tl_relu->shape.n, tl_relu->shape.c, 1, tl_relu->shape.w};
    relu.stride = {tl_relu->stride.n, tl_relu->stride.c, tl_relu->stride.h, 1};

    bmk1880v2_tensor_lmem_t neg;
    neg.start_address = tl_neg->start_address;
    neg.fmt = tl_neg->fmt;
    neg.shape = {tl_neg->shape.n, tl_neg->shape.c, 1, tl_neg->shape.w};
    neg.stride = {tl_neg->stride.n, tl_neg->stride.c, tl_neg->stride.h, 1};

    // 0. relu = relu(bottom)
    bmk1880v2_tiu_element_wise_max_param_t p13;
    p13.max = &relu;
    p13.a = &bottom;
    p13.b_is_const = 1;
    p13.b_is_signed = 1;
    p13.b_val = 0;
    ctx.tiu_element_wise_max(&p13);

    // 1. relu = (relu * GT_scale) >> GT_right_shift_width
    bmk1880v2_tiu_element_wise_mul_param_t p;
    p.res_high = nullptr;
    p.res_low = &relu;
    p.a = &relu;
    p.b_val = GT_scale;
    p.b_is_signed = true;
    p.b_is_const = 1;
    p.rshift_bits = GT_right_shift_width;
    p.relu_enable = 0;
    ctx.tiu_element_wise_mul(&p);

    // 2. neg = neg(0, botom)
    bmk1880v2_tiu_element_wise_min_param_t p7;
    p7.min = &neg;
    p7.a = &bottom;
    p7.b_is_const = 1;
    p7.b_val = 0;
    p7.b_is_signed = 1;
    ctx.tiu_element_wise_min(&p7);

    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope) >> LE_right_shift_width
    p.res_high = nullptr;
    p.res_low = &neg;
    p.a = &neg;
    p.b_val = LE_scale;
    p.b_is_signed = true;
    p.b_is_const = 1;
    p.rshift_bits = LE_right_shift_width;
    ctx.tiu_element_wise_mul(&p);

    // 4. bottom = or relu, neg
    bmk1880v2_tiu_element_wise_or_int8_param_t p9;
    p9.res = &bottom;
    p9.a = &relu;
    p9.b = &neg;
    ctx.tiu_element_wise_or_int8(&p9);

    // Store with matrix format
    // move result to global
    // Global memory shape == local memory shape
    // Gobal memory stride from local memory shape
    bmk1880v2_matrix_tgmem_t ts_top;
    ts_top.base_reg_index = BM1880v2BackendContext::NEURON_MEMORY;
    ts_top.start_address = slice_top_gaddr;
    ts_top.shape = {tl_bottom->shape.n, tl_bottom->shape.col};
    ts_top.stride = {tl_bottom->shape.col};

    DEBUG_BMNET(std::cout << llvm::format("    [%d] gdma_store: bottom\n"
                                          "      shape(n=%d, c=%d, w=%d, col=%d)\n",
                                          slice_idx, tl_bottom->shape.n, tl_bottom->shape.c,
                                          tl_bottom->shape.w, tl_bottom->shape.col));

    bmk1880v2_tdma_l2tg_matrix_copy_param_t p10;
    p10.src = tl_bottom;
    p10.dst = &ts_top;
    ctx.tdma_l2g_matrix_copy(&p10);

    // free
    ctx.lmem_free_matrix(tl_neg);
    ctx.lmem_free_matrix(tl_relu);
    ctx.lmem_free_matrix(tl_bottom);

    slice_bottom_gaddr += count_sec * INT8_SIZE;
    slice_top_gaddr += count_sec * INT8_SIZE;
  }
}
#endif

//}  // namespace bmnet