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
                              int input_h, int input_w) {
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
                                       int input_w) {
  _forward_internel(ctx, layer_id, bottom_gaddr, top_gaddr, negative_slope, input_n, input_c,
                    input_h, input_w);
}

//}  // namespace bmnet
