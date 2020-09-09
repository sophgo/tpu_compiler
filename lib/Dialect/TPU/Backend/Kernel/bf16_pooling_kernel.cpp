/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_pooling_kernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <iostream>
#include "CviBackendContext.h"

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_pooling"

#define ASSERT(x) assert(x)
// #define ENABLE_DEBUG_MSG

static int get_csize_local_bitwidth(const CviBackendContext &ctx, int h, int w, int bitwidth) {
  // TODO: pass fmt
  return ALIGN(h * w * bitsize_of_fmt(CVK_FMT_BF16), EU_NUM) * bitwidth / 8;
}

static int get_csize_global_bitwidth(const CviBackendContext &ctx, int h, int w, int bitwidth){
  //ASSERT(ctx.hw.chip_version != BM_CHIP_BM1880);

  return ALIGN(h * w * bitwidth, 32) / 8;
}

// namespace bmnet {

static int gcd(int a, int b) {
  int r = a | b;

  if (!a || !b) {
    return r;
  }

  /* Isolate lsbit of r */
  r &= -r;

  while (!(b & r)) {
    b >>= 1;
  }
  if (b == r) {
    return r;
  }

  for (;;) {
    while (!(a & r)) {
      a >>= 1;
    }
    if (a == r) {
      return r;
    }
    if (a == b) {
      return a;
    }

    if (a < b) {
      int tmp = a;
      a = b;
      b = tmp;
    }
    a -= b;
    a >>= 1;
    if (a & r) {
      a += b;
    }
    a >>= 1;
  }
}

static int lcm(int a, int b) {
  if (a && b) {
    return (a / gcd(a, b)) * b;
  } else {
    return 0;
  }
}

static int index_bit_width(int kernel_size) {
  if (kernel_size > 256) {
    return 16;
  } else if (kernel_size > 16) {
    return 8;
  } else {
    return 4;
  }
}

static int index_size_lmem(const CviBackendContext &ctx, int kernel_size, int n, int c, int h,
                           int w) {
  int bit_width = index_bit_width(kernel_size);
  return n * ALIGN(c, NPU_NUM) * get_csize_local_bitwidth(ctx, h, w, bit_width);
}

static int index_size_gmem(const CviBackendContext &ctx, int kernel_size, int n, int c, int h,
                           int w) {
  int bit_width = index_bit_width(kernel_size);
  return n * c * get_csize_global_bitwidth(ctx, h, w, bit_width);
}

static int tensor_size_gmem(const CviBackendContext &ctx, int n, int c, int h, int w) {
  return n * c * h * w * sizeof(uint16_t);
}

static cvk_tl_t *alloc_tensor_lmem(const CviBackendContext &ctx, int n, int c,
                                                  int h, int w) {
  cvk_tl_shape_t shape = {static_cast<uint32_t>(n), static_cast<uint32_t>(c),
                                         static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
  return ctx.lmem_alloc_tensor(shape, CVK_FMT_BF16, /*eu_align=*/1);
}

typedef struct {
  gaddr_t ifmap_gaddr, ofmap_gaddr;
  gaddr_t index_gaddr, o_findex_gaddr;
  int n, c, h, w;
  int extra_pad_b, extra_pad_r;
  bool last_column;
  int kh, kw;
  int pad_top, pad_bot;
  int pad_left, pad_right;
  int stride_h, stride_w;
  int is_avg_pooling;
  float avg_const;  // default(passing 0.0f) is 1/kh*kw
  int do_relu;
} pooling_t;

static int pooling_out_h(const pooling_t *p) {
  return floor(float((p->h + p->pad_top + p->pad_bot - p->kh)) / p->stride_h + 1);
}

static int pooling_out_w(const pooling_t *p) {
  return floor(float(p->w + p->pad_left + p->pad_right - p->kw) / p->stride_w + 1);
}

static int pooling_aligned_out_h(const pooling_t *p) {
  int out_w = pooling_out_w(p);
  int aligned_indices = 32 / index_bit_width(p->kh * p->kw);
  return lcm(out_w, aligned_indices) / out_w;
}

static int pooling_size_lmem(const CviBackendContext &ctx, const pooling_t *p) {
  int is_max_pooling = !p->is_avg_pooling;
  int out_h = pooling_out_h(p);
  int out_w = pooling_out_w(p);

  // TODO: pass fmt
  int in_size = _tensor_size_lmem(ctx, p->n, p->c, p->h, p->w, CVK_FMT_BF16);
  int out_size = _tensor_size_lmem(ctx, p->n, p->c, out_h, out_w, CVK_FMT_BF16);

  int index_size = 0;
  if (is_max_pooling) {
    index_size = index_size_lmem(ctx, p->kh * p->kw, p->n, p->c, out_h, out_w);
  }

  return in_size + out_size + index_size;
}

static int split_pooling_forward(const CviBackendContext &ctx, const pooling_t *_p,
                                 int *step_c, int *step_h) {
  int target_size = NPU_NUM * LOCAL_MEM_SIZE;
  int npu_num = NPU_NUM;
  *step_c = _p->c;
  *step_h = _p->h;
  if (pooling_size_lmem(ctx, _p) <= target_size) {
    return 0;
  }

  pooling_t p = *_p;
  if (p.c > npu_num) {
    *step_c = p.c = npu_num;
    int size = pooling_size_lmem(ctx, &p);
    if (size <= target_size) {
      *step_c = (target_size / size) * npu_num;
      return 0;
    }
  }

  /* Every output index slice must conform to 32-bit
   * alignment constrant in global memory.
   */
  p.pad_top = p.pad_bot = 0;
  int min_in_h = p.stride_h * pooling_aligned_out_h(&p);
  p.h = min_in_h + p.kh;
  int min_size = pooling_size_lmem(ctx, &p);
  if (min_size > target_size) {
    return -1;
  }

  for (int i = target_size / min_size; i < _p->h; i++) {
    p.h = min_in_h * i + p.kh;
    if (pooling_size_lmem(ctx, &p) > target_size) {
      *step_h = min_in_h * (i - 1);
      break;
    }
  }
  return 0;
}

static void pooling_forward_slice(const CviBackendContext &ctx, uint32_t layer_id,
                                  const pooling_t *all, const pooling_t *slice) {
  const pooling_t *s = slice;
  int out_h = pooling_out_h(slice);
  int out_w = pooling_out_w(slice);

  cvk_tl_t *ifmap = alloc_tensor_lmem(ctx, s->n, s->c, s->h, s->w);
  cvk_tl_t *ofmap = alloc_tensor_lmem(ctx, s->n, s->c, out_h, out_w);
  int is_max_pooling = !s->is_avg_pooling;
  cvk_tg_t ts_pooling;

  cvk_tg_shape_t ts_all_in_shape = {
      static_cast<uint32_t>(all->n), static_cast<uint32_t>(all->c), static_cast<uint32_t>(all->h),
      static_cast<uint32_t>(all->w)};

  ts_pooling.start_address = s->ifmap_gaddr;
  ts_pooling.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(s->ifmap_gaddr);
  ts_pooling.fmt = CVK_FMT_BF16;
  ts_pooling.shape.n = s->n;
  ts_pooling.shape.c = s->c;
  ts_pooling.shape.h = s->h;
  ts_pooling.shape.w = s->w;
  ts_pooling.stride = ctx.tg_default_stride(ts_all_in_shape, CVK_FMT_BF16);
  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &ts_pooling;
  p.dst = ifmap;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  tdma_g2l_bf16_tensor_copy\n"
      "    src addr 0x%lx, shape (%d, %d, %d, %d)\n"
      "    dst addr 0x%x, shape (%d, %d, %d, %d)\n",
      p.src->start_address, p.src->shape.n, p.src->shape.c, p.src->shape.h, p.src->shape.w,
      p.dst->start_address, p.dst->shape.n, p.dst->shape.c, p.dst->shape.h, p.dst->shape.w););

  ctx.tdma_g2l_bf16_tensor_copy(&p);

  int pad_bot = s->pad_bot;
  if (s->last_column) {
    pad_bot = s->pad_bot + s->extra_pad_b;
  }

  if (is_max_pooling) {
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = ofmap;
    param.ifmap = ifmap;
    param.kh = s->kh;
    param.kw = s->kw;
    param.pad_top = s->pad_top;
    param.pad_bottom = pad_bot;
    param.pad_left = s->pad_left;
    param.pad_right = s->pad_right + s->extra_pad_r;
    param.stride_h = s->stride_h;
    param.stride_w = s->stride_w;
    param.ins_fp = 0;
    param.layer_id = layer_id;

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "  tiu_bf16_max_pooling\n"
        "    ifmap shape (%d, %d, %d, %d)\n"
        "    ofmap shape (%d, %d, %d, %d)\n"
        "    kh %d, kw %d, stride_h %d, stride_w %d\n"
        "    pad_top: %d pad_bot:%d pad_left %d pad_right %d\n",
        ifmap->shape.n, ifmap->shape.c, ifmap->shape.h, ifmap->shape.w, ofmap->shape.n,
        ofmap->shape.c, ofmap->shape.h, ofmap->shape.w, s->kh, s->kw, s->stride_h, s->stride_w,
        param.pad_top, param.pad_bottom, param.pad_left, param.pad_right););

    ctx.tiu_max_pooling(&param);
  } else {
    float avg_const = s->avg_const;

    ASSERT(avg_const == 0.0f);
    if (avg_const == 0.0f) {
      // kernel will do kernel average, uses 1.0
      // avg_const = 1 / ((float)s->kh * s->kw);
      avg_const = 1.0;
    }

    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = ofmap;
    param.ifmap = ifmap;
    param.kh = s->kh;
    param.kw = s->kw;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = s->pad_top;
    param.pad_bottom = pad_bot;
    param.pad_left = s->pad_left;
    param.pad_right = s->pad_right + s->extra_pad_r;
    param.stride_h = s->stride_h;
    param.stride_w = s->stride_w;
    param.avg_pooling_const = ctx.convert_fp32_to_bf16(avg_const);
    param.layer_id = layer_id;

    LLVM_DEBUG(llvm::errs() << llvm::format(
        "  tiu_bf16_avg_pooling\n"
        "    ifmap shape (%d, %d, %d, %d)\n"
        "    ofmap shape (%d, %d, %d, %d)\n"
        "    kh %d, kw %d, stride_h %d, stride_w %d\n"
        "    avg_const %f, 0x%x\n",
        ifmap->shape.n, ifmap->shape.c, ifmap->shape.h, ifmap->shape.w, ofmap->shape.n,
        ofmap->shape.c, ofmap->shape.h, ofmap->shape.w, s->kh, s->kw, s->stride_h, s->stride_w,
        avg_const, param.avg_pooling_const););

    ctx.tiu_average_pooling(&param);
  }

  cvk_tg_t tg_output;
  cvk_tg_shape_t ts_all_out_shape = {
      static_cast<uint32_t>(all->n), static_cast<uint32_t>(all->c), static_cast<uint32_t>(pooling_out_h(all)),
      static_cast<uint32_t>(pooling_out_w(all))};

  tg_output.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(s->ofmap_gaddr);
  tg_output.fmt = CVK_FMT_BF16;
  tg_output.start_address = s->ofmap_gaddr;
  tg_output.shape.n = s->n;
  tg_output.shape.c = s->c;
  tg_output.shape.h = out_h;
  tg_output.shape.w = out_w;
  tg_output.stride = ctx.tg_default_stride(ts_all_out_shape, CVK_FMT_BF16);
  cvk_tdma_l2g_tensor_copy_param_t out_param = {0};
  out_param.src = ofmap;
  out_param.dst = &tg_output;

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  tdma_l2g_bf16_tensor_copy\n"
      "    src addr 0x%x, shape (%d, %d, %d, %d)\n"
      "    dst addr 0x%lx, shape (%d, %d, %d, %d)\n",
      out_param.src->start_address, out_param.src->shape.n, out_param.src->shape.c,
      out_param.src->shape.h, out_param.src->shape.w, out_param.dst->start_address,
      out_param.dst->shape.n, out_param.dst->shape.c, out_param.dst->shape.h,
      out_param.dst->shape.w););

  ctx.tdma_l2g_bf16_tensor_copy(&out_param);

  ctx.lmem_free_tensor(ofmap);
  ctx.lmem_free_tensor(ifmap);
}

static void adjust_nc(const CviBackendContext &ctx, int *n, int *c) {
  *c *= *n;
  *n = 1;
  while (*c >= 0x1000) {
    *c /= 2;
    *n *= 2;
  }
}

static void adjust_pad(pooling_t *s) {
  int out_h = (s->h + s->pad_top + s->pad_bot - s->kh) / s->stride_h + 1;
  if ((out_h * s->stride_h < s->h + s->pad_top) &&
      ((s->h + s->pad_top + s->pad_bot - s->kh) % s->stride_h != 0)) {
    s->extra_pad_b = (s->stride_h - (s->h + s->pad_top + s->pad_bot - s->kh) % s->stride_h);
  }

  int out_w = (s->w + s->pad_left + s->pad_right - s->kw) / s->stride_w + 1;
  if ((out_w * s->stride_w < s->w + s->pad_left) &&
      ((s->w + s->pad_left + s->pad_right - s->kw) % s->stride_w != 0)) {
    s->extra_pad_r = (s->stride_w - (s->w + s->pad_left + s->pad_right - s->kw) % s->stride_w);
  }
}

void bf16_pooling_forward_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                 gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr, gaddr_t index_gaddr,
                                 gaddr_t o_findex_gaddr, int n, int c, int h, int w, int kh, int kw,
                                 int pad_top, int pad_bot, int pad_left, int pad_right,
                                 int stride_h, int stride_w, int is_avg_pooling,
                                 float avg_const,  // default(passing 0.0f) is 1/kh*kw
                                 int do_relu, const bool ceil_mode) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "bf16_pooling_forward_kernel:\n"
      "    ifmap_gaddr 0x%lx, ofmap_gaddr 0x%lx, index_gaddr 0x%lx, o_findex_gaddr 0x%lx\n"
      "    shape (%d, %d, %d, %d), kh %d, kw %d\n"
      "    pad_top %d, pad_bot %d, pad_left %d, pad_right %d, stride_h %d, stride_w %d\n"
      "    is_avg_pooling %d, avg_const %f, do_relu %d, ceil_mode %d\n",
      ifmap_gaddr, ofmap_gaddr, index_gaddr, o_findex_gaddr, n, c, h, w, kh, kw, pad_top, pad_bot,
      pad_left, pad_right, stride_h, stride_w, is_avg_pooling, avg_const, do_relu, ceil_mode););

  ASSERT(!do_relu); /* Don't support relu for now. */

  adjust_nc(ctx, &n, &c);

  pooling_t pooling = {.ifmap_gaddr = ifmap_gaddr,
                       .ofmap_gaddr = ofmap_gaddr,
                       .index_gaddr = index_gaddr,
                       .o_findex_gaddr = o_findex_gaddr,
                       .n = n,
                       .c = c,
                       .h = h,
                       .w = w,
                       .extra_pad_b = 0,
                       .extra_pad_r = 0,
                       .last_column = true,
                       .kh = kh,
                       .kw = kw,
                       .pad_top = pad_top,
                       .pad_bot = pad_bot,
                       .pad_left = pad_left,
                       .pad_right = pad_right,
                       .stride_h = stride_h,
                       .stride_w = stride_w,
                       .is_avg_pooling = is_avg_pooling,
                       .avg_const = avg_const,
                       .do_relu = do_relu};

  if (ceil_mode) {
    adjust_pad(&pooling);
  }
  int step_c = c, step_h = h;
  int err = split_pooling_forward(ctx, &pooling, &step_c, &step_h);
  ASSERT(err == 0 && step_h >= stride_h);

  for (int ci = 0; ci < c; ci += step_c) {
    int slice_c = std::min(step_c, c - ci);
    int out_h = pooling_out_h(&pooling);
    int out_w = pooling_out_w(&pooling);
    gaddr_t ifmap_nc = ifmap_gaddr + tensor_size_gmem(ctx, 1, ci, h, w);
    gaddr_t ofmap_nc = ofmap_gaddr + tensor_size_gmem(ctx, 1, ci, out_h, out_w);
    gaddr_t index_nc = index_gaddr + index_size_gmem(ctx, kh * kw * sizeof(uint16_t), 1,
                                                     ci, out_h, out_w);

    pooling_t slice = pooling;
    slice.ifmap_gaddr = ifmap_nc;
    slice.ofmap_gaddr = ofmap_nc;
    slice.index_gaddr = index_nc;

    slice.c = slice_c;
    if (step_h == h) {
      slice.last_column = true;
      pooling_forward_slice(ctx, layer_id, &pooling, &slice);
      continue;
    }

    for (int hi = -pad_top; hi < h + pad_bot; hi += step_h) {
      int slice_h = std::min(step_h + kh, h + pad_bot - hi);
      if (hi + slice_h >= h + pad_bot) {
        slice.last_column = true;
      } else {
        slice.last_column = false;
      }

      if (slice_h < kh) {
        break;
      }

      int out_hi = (hi + pad_top) / stride_h;
      slice.pad_top = -std::min(0, hi);
      slice.pad_bot = std::max(0, hi + slice_h - h);
      slice.ifmap_gaddr = ifmap_nc + tensor_size_gmem(ctx, 1, 1, hi + slice.pad_top, w);
      slice.ofmap_gaddr = ofmap_nc + tensor_size_gmem(ctx, 1, 1, out_hi, out_w);
      slice.index_gaddr =
          index_nc + index_size_gmem(ctx, kh * kw * sizeof(uint16_t), 1, 1, out_hi, out_w);
      slice.h = slice_h - slice.pad_top - slice.pad_bot;
      pooling_forward_slice(ctx, layer_id, &pooling, &slice);
    }
  }
}
