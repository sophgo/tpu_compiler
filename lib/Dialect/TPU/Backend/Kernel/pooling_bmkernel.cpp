/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: pooling_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_pooling"

#define ASSERT(x) assert(x)

static int get_csize_local_bitwidth(const CviBackendContext &ctx, int h, int w, int bitwidth)
{
  return ALIGN(h * w, EU_NUM) * bitwidth / 8;
}

static int get_csize_global_bitwidth(const CviBackendContext &ctx, int h, int w, int bitwidth)
{
  //ASSERT(ctx.hw.chip_version != BM_CHIP_BM1880);
  return ALIGN(h * w * bitwidth, 32) / 8;
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
  return n * c * h * w * sizeof(uint8_t);
}

static cvk_tl_t *alloc_tensor_lmem(const CviBackendContext &ctx, int n, int c,
                                                  int h, int w) {
  cvk_tl_shape_t shape = {static_cast<uint32_t>(n), static_cast<uint32_t>(c),
                                         static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
  return ctx.lmem_alloc_tensor(shape, CVK_FMT_I8, 1);
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

  int in_size = tensor_size_lmem(ctx, p->n, p->c, p->h, p->w);
  int out_size = tensor_size_lmem(ctx, p->n, p->c, out_h, out_w);
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

  // NOTICE: cant shift lmem address cuz \assert_stride_type_0(ifmap) limitation
  // that ifmap->stride.c should be equal with h/w
  // manual simulate 'stride' once sw > 31
  // TODO: padding != 0
  int is_split_w = s->stride_w > 15; // 1880v2 support sw [1,31]
  int slice_w_nr = is_split_w ? s->w / s->stride_w : 1;
  int slice_h_nr = is_split_w ? s->h / s->stride_h : 1;

  for (int ih = 0; ih < slice_h_nr; ih++) {
    for (int iw = 0; iw < slice_w_nr; iw++) {
      int ofmap_shift = 0;
      int ifmap_shift = 0;
      int ifmap_w = s->w;
      int ifmap_h = s->h;
      int extend_out_w = out_w;
      int extend_out_h = out_h;
      if (is_split_w) {
        int step_w = iw * s->stride_w;
        int step_h = ih * s->stride_h;
        ifmap_shift = (step_h * all->w) + step_w;
        ofmap_shift = ih * slice_w_nr + iw; // shift one

        ifmap_w = s->w - step_w;
        ifmap_h = s->h - step_h;
        pooling_t extend_s = *slice;
        extend_s.w = ifmap_w;
        extend_s.h = ifmap_h;
        extend_out_h = pooling_out_h(&extend_s);
        extend_out_w = pooling_out_w(&extend_s);
      }

      cvk_tl_t *ifmap = alloc_tensor_lmem(ctx, s->n, s->c, ifmap_h, ifmap_w);
      cvk_tl_t *ofmap = alloc_tensor_lmem(ctx, s->n, s->c, extend_out_h, extend_out_w);
      int is_max_pooling = !s->is_avg_pooling;
      cvk_tg_t ts_pooling;

      cvk_tg_shape_t ts_all_in_shape = {
        static_cast<uint32_t>(all->n), static_cast<uint32_t>(all->c), static_cast<uint32_t>(all->h),
        static_cast<uint32_t>(all->w)};

      ts_pooling.start_address = s->ifmap_gaddr + ifmap_shift;
      ts_pooling.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(s->ifmap_gaddr);
      ts_pooling.fmt = CVK_FMT_I8;
      ts_pooling.shape.n = s->n;
      ts_pooling.shape.c = s->c;
      ts_pooling.shape.h = ifmap_h;
      ts_pooling.shape.w = ifmap_w;
      ts_pooling.stride = ctx.tg_default_stride(ts_all_in_shape, ts_pooling.fmt);
      cvk_tdma_g2l_tensor_copy_param_t p = {0};
      p.src = &ts_pooling;
      p.dst = ifmap;
      ctx.tdma_g2l_tensor_copy(&p);

      float avg_const = s->avg_const;
      ASSERT(avg_const == 0.0f);
      //uint8_t avg_const_fixed = 0;
      //int rshift = 0;
      //if (s->is_avg_pooling && avg_const == 0.0f) {
      //  rshift = avg_quantize_rshift(static_cast<float>(s->kh) * s->kw, &avg_const_fixed);
      //  LLVM_DEBUG(
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
        param.layer_id = layer_id;
        ctx.tiu_max_pooling(&param);

        LLVM_DEBUG(llvm::errs() << llvm::format(
              "  tiu_max_pooling\n"
              "    ifmap shape (%d, %d, %d, %d)\n"
              "    ofmap shape (%d, %d, %d, %d)\n"
              "    kh %d, kw %d, stride_h %d, stride_w %d\n",
              ifmap->shape.n, ifmap->shape.c, ifmap->shape.h, ifmap->shape.w, ofmap->shape.n,
              ofmap->shape.c, ofmap->shape.h, ofmap->shape.w, s->kh, s->kw, s->stride_h, s->stride_w));
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
        param.avg_pooling_const = avg_const_quantized;
        param.rshift_bits = rshift_quantized;
        param.layer_id = layer_id;
        ctx.tiu_average_pooling(&param);
      }

      cvk_tg_t tg_output;
      cvk_tg_shape_t ts_all_out_shape = {
        static_cast<uint32_t>(all->n), static_cast<uint32_t>(all->c), static_cast<uint32_t>(pooling_out_h(all)),
        static_cast<uint32_t>(pooling_out_w(all))};

      tg_output.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(s->ofmap_gaddr);
      tg_output.fmt = CVK_FMT_I8;
      tg_output.start_address = s->ofmap_gaddr + ofmap_shift;
      tg_output.shape.n = s->n;
      tg_output.shape.c = s->c;
      tg_output.shape.h = extend_out_h;
      tg_output.shape.w = extend_out_w;
      tg_output.stride = ctx.tg_default_stride(ts_all_out_shape, tg_output.fmt);
      cvk_tdma_l2g_tensor_copy_param_t out_param = {0};
      out_param.src = ofmap;
      out_param.dst = &tg_output;
      ctx.tdma_l2g_tensor_copy(&out_param);
      ctx.lmem_free_tensor(ofmap);
      ctx.lmem_free_tensor(ifmap);
    }
  }
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
  int out_h = floor(float(s->h + s->pad_top + s->pad_bot - s->kh) / s->stride_h + 1);
  if ((out_h * s->stride_h < s->h + s->pad_top) &&
      ((s->h + s->pad_top + s->pad_bot - s->kh) % s->stride_h != 0)) {
    s->extra_pad_b = (s->stride_h - (s->h + s->pad_top + s->pad_bot - s->kh) % s->stride_h);
  }

  int out_w = floor(float(s->w + s->pad_left + s->pad_right - s->kw) / s->stride_w + 1);
  if ((out_w * s->stride_w < s->w + s->pad_left) &&
      ((s->w + s->pad_left + s->pad_right - s->kw) % s->stride_w != 0)) {
    s->extra_pad_r = (s->stride_w - (s->w + s->pad_left + s->pad_right - s->kw) % s->stride_w);
  }
}

void cvi_backend_tg_fixed_pooling_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr, gaddr_t index_gaddr,
    gaddr_t o_findex_gaddr, int n, int c, int h, int w, int kh, int kw, int pad_top, int pad_bot,
    int pad_left, int pad_right, int stride_h, int stride_w, int is_avg_pooling,
    float avg_const,  // default(passing 0.0f) is 1/kh*kw
    int do_relu, int right_shift_width, const int *threshold_x_quantized, const bool ceil_mode) {
  ASSERT(!do_relu); /* Don't support relu for now. */

  ctx.set_layer_id(layer_id); // pmu used

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
    int slice_c = std::min(step_c, c - ci);
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
      slice.index_gaddr = index_nc + index_size_gmem(ctx, kh * kw, 1, 1, out_hi, out_w);
      slice.h = slice_h - slice.pad_top - slice.pad_bot;
      pooling_forward_slice(ctx, layer_id, &pooling, &slice);
    }
  }
}

//}  // namespace bmnet
