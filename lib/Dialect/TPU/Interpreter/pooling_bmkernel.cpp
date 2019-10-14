/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
//#include <support/Debug.h>
//#include <support/Format.h>
//#include <targets/plat-bm188x/bmkernel/bmkernel_api.h>
//#include <targets/Target.hpp>
//#include <targets/plat-bm188x/BM1880v2BackendContext.hpp>
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_pooling"

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

static int get_csize_local(const BackendContext &ctx, int h, int w)
{
  int unit_size = ctx.hw.unit_size;
  ASSERT(unit_size > 0);
  return ALIGN(h * w, ctx.hw.eu_num) * unit_size;
}

static int get_csize_local_bitwidth(const BackendContext &ctx, int h, int w, int bitwidth)
{
  return ALIGN(h * w, ctx.hw.eu_num) * bitwidth / 8;
}

static int get_csize_global_bitwidth(const BackendContext &ctx, int h, int w, int bitwidth)
{
  //ASSERT(ctx.hw.chip_version != BM_CHIP_BM1880);
  return ALIGN(h * w * bitwidth, 32) / 8;
}

static int get_cstride_local(const BackendContext *ctx, int h, int w)
{
  return ALIGN(h * w, ctx->hw.eu_num);
}

static int get_cstride_global_bitwidth(const BackendContext &ctx, int h, int w, int bitwidth)
{
  //ASSERT(ctx.hw.chip_version != BM_CHIP_BM1880);
  return ALIGN(h * w * bitwidth, 32) / bitwidth;
}

//namespace bmnet {

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

static fmt_t data_format_of_index(int bit_width) {
  switch (bit_width) {
    case 4:
      return FMT_I4;
    case 8:
      return FMT_I8;
    case 16:
      return FMT_I16;
    default:
      ASSERT(0);
  }
  return FMT_INVALID;
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

static int index_size_lmem(const BM1880v2BackendContext &ctx, int kernel_size, int n, int c, int h,
                           int w) {
  int bit_width = index_bit_width(kernel_size);
  return n * ALIGN(c, NPU_NUM) * get_csize_local_bitwidth(ctx, h, w, bit_width);
}

static int index_size_gmem(const BM1880v2BackendContext &ctx, int kernel_size, int n, int c, int h,
                           int w) {
  int bit_width = index_bit_width(kernel_size);
  return n * c * get_csize_global_bitwidth(ctx, h, w, bit_width);
}

static int tensor_size_lmem(const BM1880v2BackendContext &ctx, int n, int c, int h, int w) {
  return n * ALIGN(c, NPU_NUM) * get_csize_local(ctx, h, w);
}

static int tensor_size_gmem(const BM1880v2BackendContext &ctx, int n, int c, int h, int w) {
  return n * c * h * w * INT8_SIZE;
}

static bmk1880v2_tensor_lmem_t *alloc_tensor_lmem(const BM1880v2BackendContext &ctx, int n, int c,
                                                  int h, int w) {
  bmk1880v2_tensor_lmem_shape_t shape = {static_cast<u32>(n), static_cast<u32>(c),
                                         static_cast<u32>(h), static_cast<u32>(w)};
  return ctx.lmem_alloc_tensor(shape, FMT_I8, 1);
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
  int right_shift_width;
  const int *threshold_x_quantized;
} pooling_t;

static int pooling_out_h(const pooling_t *p) {
  int out_h;
  if (p->last_column && p->extra_pad_b != 0) {
    out_h = (p->h + p->pad_top + p->pad_bot - p->kh) / p->stride_h + 2;
  } else {
    out_h = (p->h + p->pad_top + p->pad_bot - p->kh) / p->stride_h + 1;
  }
  return out_h;
}

static int pooling_out_w(const pooling_t *p) {
  int out_w;
  if (p->extra_pad_r != 0) {
    out_w = (p->w + p->pad_left + p->pad_right - p->kw) / p->stride_w + 2;
  } else {
    out_w = (p->w + p->pad_left + p->pad_right - p->kw) / p->stride_w + 1;
  }
  return out_w;
}

static int pooling_aligned_out_h(const pooling_t *p) {
  int out_w = pooling_out_w(p);
  int aligned_indices = 32 / index_bit_width(p->kh * p->kw);
  return lcm(out_w, aligned_indices) / out_w;
}

static int pooling_size_lmem(const BM1880v2BackendContext &ctx, const pooling_t *p) {
  int is_max_pooling = !p->is_avg_pooling;
  int out_h = pooling_out_h(p);
  int out_w = pooling_out_w(p);

  int in_size = tensor_size_lmem(ctx, p->n, p->c, p->h, p->w);
  int out_size = tensor_size_lmem(ctx, p->n, p->c, out_h, out_w);
  int index_size = 0;
  if (is_max_pooling) {
    index_size = index_size_lmem(ctx, p->kh * p->kw, p->n, p->c, out_h, out_w);
  }

  return in_size + out_size + index_size;
}

static stride_t pooling_ifmap_stride(const pooling_t *p) {
  return stride_st4(p->w * p->h * p->c, p->w * p->h, p->w, 1);
}

static stride_t pooling_ofmap_stride(const pooling_t *p) {
  int out_h = pooling_out_h(p);
  int out_w = pooling_out_w(p);
  return stride_st4(out_w * out_h * p->c, out_w * out_h, out_w, 1);
}

static stride_t pooling_index_stride(const BM1880v2BackendContext &ctx, const pooling_t *p) {
  int out_h = pooling_out_h(p);
  int out_w = pooling_out_w(p);
  int width = index_bit_width(p->kh * p->kw);
  int idx_c = get_cstride_global_bitwidth(ctx, out_h, out_w, width);
  return stride_st4(idx_c * p->c, idx_c, out_w, 1);
}

static int split_pooling_forward(const BM1880v2BackendContext &ctx, const pooling_t *_p,
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

static int avg_quantize_rshift(float denominator, uint8_t *avg_const) {
  float scale_max = 256.0;
  float denominator_max = (denominator * scale_max);
  if (denominator_max < 1) {
    LOG(FATAL) << "denominator_max is too small =" << denominator_max
               << " denominator=" << denominator << " scale_max=" << scale_max;
  }

  int m = 0;
  float denominator_quantized = 1;

  while (denominator_quantized < denominator_max) {
    m++;
    denominator_quantized = (1 << m);
  }
  m = m - 1;
  // LOG(INFO) << "m=" << m;
  denominator_quantized = static_cast<float>(1 << m);

  float scale = denominator_quantized / denominator;
  LOG(INFO) << "denominator_quantized =" << denominator_quantized << ",scale =" << scale;

  *avg_const = static_cast<uint8_t>(std::floor(scale));
  return m;
}

static void pooling_forward_slice(const BM1880v2BackendContext &ctx, u32 layer_id,
                                  const pooling_t *all, const pooling_t *slice) {
  const pooling_t *s = slice;
  int out_h = pooling_out_h(slice);
  int out_w = pooling_out_w(slice);

  bmk1880v2_tensor_lmem_t *ifmap = alloc_tensor_lmem(ctx, s->n, s->c, s->h, s->w);
  bmk1880v2_tensor_lmem_t *ofmap = alloc_tensor_lmem(ctx, s->n, s->c, out_h, out_w);
  int is_max_pooling = !s->is_avg_pooling;
  bmk1880v2_tensor_tgmem_t ts_pooling;

  bmk1880v2_tensor_tgmem_shape_t ts_all_in_shape = {
      static_cast<u32>(all->n), static_cast<u32>(all->c), static_cast<u32>(all->h),
      static_cast<u32>(all->w)};

  ts_pooling.start_address = s->ifmap_gaddr;
  ts_pooling.base_reg_index = BM1880v2BackendContext::NEURON_MEMORY;
  ts_pooling.fmt = FMT_I8;
  ts_pooling.shape.n = s->n;
  ts_pooling.shape.c = s->c;
  ts_pooling.shape.h = s->h;
  ts_pooling.shape.w = s->w;
  ts_pooling.stride = ctx.tensor_tgmem_default_stride(ts_all_in_shape);
  bmk1880v2_tdma_tg2l_tensor_copy_param_t p;
  p.src = &ts_pooling;
  p.dst = ifmap;
  ctx.tdma_g2l_tensor_copy(&p);

  float avg_const = s->avg_const;
  ASSERT(avg_const == 0.0f);
  //uint8_t avg_const_fixed = 0;
  //int rshift = 0;
  //if (s->is_avg_pooling && avg_const == 0.0f) {
  //  rshift = avg_quantize_rshift(static_cast<float>(s->kh) * s->kw, &avg_const_fixed);
  //  DEBUG_BMNET(
  //      llvm::errs() << llvm::format("avg_const_fixed=%u,avg_rshift=%d\n", avg_const_fixed, rshift));
  //}

  int pad_bot = s->pad_bot;
  if (s->last_column) {
    pad_bot = s->pad_bot + s->extra_pad_b;
  }

  int threshold_x_quantized = 1;
  int right_shift_width = 0;
  if (all->threshold_x_quantized) {
    threshold_x_quantized = all->threshold_x_quantized[0];
    right_shift_width = all->right_shift_width;
  }
  LOG(INFO) << "threshold_x_quantized =" << threshold_x_quantized
            << ",right_shift_width =" << right_shift_width;

  if (is_max_pooling) {
    bmk1880v2_tiu_max_pooling_param_t param;
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
    param.layer_id = layer_id;
    ctx.tiu_max_pooling(&param);

    bmk1880v2_tiu_element_wise_mul_param_t p2;
    p2.res_high = nullptr;
    p2.res_low = ofmap;
    p2.a = ofmap;
    p2.b_val = threshold_x_quantized;
    p2.b_is_signed = 0;
    p2.b_is_const = 1;
    p2.rshift_bits = right_shift_width;
    p2.layer_id = layer_id;
    ctx.tiu_element_wise_mul(&p2);
  } else {
    // ASSERT(0); // TODO(wwcai)
    //int rshift_quantized = rshift + right_shift_width;
    //int avg_const_quantized = static_cast<int>(avg_const_fixed) * threshold_x_quantized;
    //while (avg_const_quantized > 255) {
    //  avg_const_quantized >>= 1;
    //  rshift_quantized--;
    //}
    int rshift_quantized = right_shift_width;
    int avg_const_quantized = threshold_x_quantized;
    bmk1880v2_tiu_average_pooling_param_t param;
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
    param.avg_pooling_const = avg_const_quantized;
    param.rshift_bits = rshift_quantized;
    param.layer_id = layer_id;
    ctx.tiu_average_pooling(&param);
    LOG(INFO) << "avg_const_quantized =" << avg_const_quantized
              << ",rshift_quantized =" << rshift_quantized;
  }

  bmk1880v2_tensor_tgmem_t tg_output;
  bmk1880v2_tensor_tgmem_shape_t ts_all_out_shape = {
      static_cast<u32>(all->n), static_cast<u32>(all->c), static_cast<u32>(pooling_out_h(all)),
      static_cast<u32>(pooling_out_w(all))};

  tg_output.base_reg_index = BM1880v2BackendContext::NEURON_MEMORY;
  tg_output.fmt = FMT_I8;
  tg_output.start_address = s->ofmap_gaddr;
  tg_output.shape.n = s->n;
  tg_output.shape.c = s->c;
  tg_output.shape.h = out_h;
  tg_output.shape.w = out_w;
  tg_output.stride = ctx.tensor_tgmem_default_stride(ts_all_out_shape);
  bmk1880v2_tdma_l2tg_tensor_copy_param_t out_param;
  out_param.src = ofmap;
  out_param.dst = &tg_output;
  ctx.tdma_l2g_tensor_copy(&out_param);
  ctx.lmem_free_tensor(ofmap);
  ctx.lmem_free_tensor(ifmap);
}

static void adjust_nc(const BM1880v2BackendContext &ctx, int *n, int *c) {
  *n = ceiling_func_shift(*n, NODECHIP_SHIFT);
  *c *= *n;
  *n = 1;
  while (*c >= 0x1000) {  // TODO(wwcai): 0x10000 in bm1682
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

void bmnet_pooling_fixed_forward_bmkernel(
    const BM1880v2BackendContext &ctx, u32 stream_id, u32 inst_id, u32 layer_id, const u32 *depends,
    u32 depends_len, gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr, gaddr_t index_gaddr,
    gaddr_t o_findex_gaddr, int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w, int is_avg_pooling,
    float avg_const,  // default(passing 0.0f) is 1/kh*kw
    int do_relu, int right_shift_width, const int *threshold_x_quantized, const bool ceil_mode) {
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
                       .do_relu = do_relu,
                       .right_shift_width = right_shift_width,
                       .threshold_x_quantized = threshold_x_quantized};

  if (ceil_mode) {
    adjust_pad(&pooling);
  }
  int step_c = c, step_h = h;
  int err = split_pooling_forward(ctx, &pooling, &step_c, &step_h);
  ASSERT(err == 0 && step_h >= stride_h);

  for (int ci = 0; ci < c; ci += step_c) {
    int slice_c = math_min(step_c, c - ci);
    int out_h = pooling_out_h(&pooling);
    int out_w = pooling_out_w(&pooling);
    gaddr_t ifmap_nc = ifmap_gaddr + tensor_size_gmem(ctx, 1, ci, h, w);
    gaddr_t ofmap_nc = ofmap_gaddr + tensor_size_gmem(ctx, 1, ci, out_h, out_w);
    gaddr_t index_nc = index_gaddr + index_size_gmem(ctx, kh * kw, 1, ci, out_h, out_w);

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
      int slice_h = math_min(step_h + kh, h + pad_bot - hi);
      if (hi + slice_h >= h + pad_bot) {
        slice.last_column = true;
      } else {
        slice.last_column = false;
      }

      if (slice_h < kh) {
        break;
      }

      int out_hi = (hi + pad_top) / stride_h;
      slice.pad_top = -math_min(0, hi);
      slice.pad_bot = math_max(0, hi + slice_h - h);
      slice.ifmap_gaddr = ifmap_nc + tensor_size_gmem(ctx, 1, 1, hi + slice.pad_top, w);
      slice.ofmap_gaddr = ofmap_nc + tensor_size_gmem(ctx, 1, 1, out_hi, out_w);
      slice.index_gaddr = index_nc + index_size_gmem(ctx, kh * kw, 1, 1, out_hi, out_w);
      slice.h = slice_h - slice.pad_top - slice.pad_bot;
      pooling_forward_slice(ctx, layer_id, &pooling, &slice);
    }
  }
}

//}  // namespace bmnet
