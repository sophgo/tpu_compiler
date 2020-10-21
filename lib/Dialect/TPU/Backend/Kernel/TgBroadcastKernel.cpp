/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 *
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

enum class BroadcastType {
  BroadcastAdd,
  BroadcastSub
};

#define DEBUG_TYPE "TgBroadcastKernel"
void broadcast_one_to_all_lane(const CviBackendContext &ctx, cvk_tl_t inputBuffer, cvk_fmt_t fmt) {
  assert(inputBuffer.shape.h * inputBuffer.shape.w < ((1 << 16 )- 1));//Reg bit field = 16 bit
  // reshape
  cvk_tl_t tl_src;
  tl_src.start_address = inputBuffer.start_address;
  tl_src.fmt = fmt;
  tl_src.shape = ctx.tl_shape_t4(inputBuffer.shape.n, 1, NPU_NUM, inputBuffer.shape.h * inputBuffer.shape.w);
  tl_src.stride = ctx.tl_default_stride(tl_src.shape, fmt, /*eu_align=*/1);
  tl_src.stride.h = 0;

  cvk_tl_t tl_dst;
  tl_dst.start_address = inputBuffer.start_address;
  tl_dst.fmt = fmt;
  tl_dst.shape = ctx.tl_shape_t4(inputBuffer.shape.n, NPU_NUM, 1, inputBuffer.shape.h * inputBuffer.shape.w);
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/1);

  cvk_tdma_l2l_tensor_copy_param_t p = {0};
  p.src = &tl_src;
  p.dst = &tl_dst;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "         L2L Reshape:\n"
                  "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                  "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                  p.src->start_address, p.src->shape.n,
                  p.src->shape.c, p.src->shape.h, p.src->shape.w, p.src->stride.n,
                  p.src->stride.c, p.src->stride.h, p.src->stride.w, p.dst->start_address,
                  p.dst->shape.n, p.dst->shape.c, p.dst->shape.h, p.dst->shape.w,
                  p.dst->stride.n, p.dst->stride.c, p.dst->stride.h, p.dst->stride.w));
  ctx.tdma_l2l_tensor_copy(&p);
}

void tg_broadcast_kernel(const CviBackendContext &ctx,
                          uint32_t layer_id, gaddr_t ga_inputs[],
                          gaddr_t ga_output, int n, int c, int h,
                          int w, int bn, int bc, int bh, int bw,
                          bool do_relu, int32_t rshift, const int32_t *multipliers,
                          BroadcastType type, cvk_fmt_t fmt) {
  llvm::errs() << llvm::format("a shape:[%d,%d,%d,%d] b shape:[%d,%d,%d,%d], relu:%d, fmt:%d\n",
                               n, c, h, w, bn, bc, bh, bw, do_relu, fmt);
  assert(!do_relu);
  // Only support to broadcast in n & c dimension.
  assert(bn == 1);
  assert(bc == 1);
  assert(bh == h);
  assert(bw == w);

  uint32_t lmem_required = (uint32_t)LOCAL_MEM_SIZE + 1;
  int32_t step_n = n;
  int32_t step_c = NPU_NUM;
  int32_t step_h = h;
  int32_t step_w = w;

  cvk_tl_shape_t shape_a, shape_b;

  uint32_t elt_size = (fmt == CVK_FMT_BF16) ? 2 : 1;
  // bf16  input0 + input1
  // int8  input0 + input1 + output_high
  int block_num = (fmt == CVK_FMT_BF16) ? 1 : 2;

  // determin the shape of tile.
  for (step_w = w; step_w > 0; step_w--) {
    for (step_h = h; step_h > 0; step_h--) {
      for (step_n = n; step_n > 0; step_n--) {
        shape_a = ctx.tl_shape_t4(step_n, step_c, step_h, step_w);
        shape_b = ctx.tl_shape_t4(1, 1, step_h, step_w);
        lmem_required = ctx.lmem_tensor_to_size(shape_a, fmt, 1) * block_num +
                  ctx.lmem_tensor_to_size(shape_b, fmt, 1);
        if (lmem_required <= (uint32_t)LOCAL_MEM_SIZE) {
          goto after_loop;
        }
      }
    }
  }

after_loop:
  if (lmem_required > (uint32_t)LOCAL_MEM_SIZE) {
    assert(0);
  }

  llvm::errs() << llvm::format("step:[%d,%d,%d,%d],lmem:%d\n", step_n, step_c, step_h,
                               step_w, (int)lmem_required);

  cvk_tl_t *tl_a = ctx.lmem_alloc_tensor(shape_a, fmt, 1);
  cvk_tl_t *tl_b = ctx.lmem_alloc_tensor(shape_b, fmt, 1);
  cvk_tl_t *tl_a_h = nullptr;
  if (fmt == CVK_FMT_I8) {
    tl_a_h = ctx.lmem_alloc_tensor(shape_a, fmt, 1);
    assert(tl_a_h);
  }
  assert(tl_a && tl_b);

  for (int pos_h = 0; pos_h < h; pos_h += step_h) {
    int cur_h = std::min(h - pos_h, step_h);
    for (int pos_w = 0; pos_w < w; pos_w += step_w) {
      int cur_w = std::min(w - pos_w, step_w);
      shape_b = ctx.tl_shape_t4(1, 1, cur_h, cur_w);

      // load b to lmem
      cvk_tl_t operand;
      uint64_t b_offset = (pos_w + pos_h * bw) * elt_size;
      operand.start_address = tl_b->start_address;
      operand.shape = shape_b;
      operand.stride = ctx.tl_default_stride(shape_b, fmt, 1);
      operand.fmt = fmt;
      cvk_tg_stride_t stride = ctx.tg_default_stride(bc, bh, bw, fmt);
      ctx.tdma_load_stride(&operand, ga_inputs[1] + b_offset, stride);

      llvm::errs() << llvm::format(
          "load b, addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
          operand.start_address, shape_b.n, shape_b.c, shape_b.h, shape_b.w, stride.n,
          stride.c, stride.h, stride.w, b_offset);

      // broadcast b to all lanes
      broadcast_one_to_all_lane(ctx, operand, fmt);
      shape_b = ctx.tl_shape_t4(1, NPU_NUM, cur_h, cur_w);
      cvk_tl_t operand_b;
      operand_b.start_address = tl_b->start_address;
      operand_b.shape = shape_b;
      operand_b.stride = ctx.tl_default_stride(shape_b, fmt, /*eu_align=*/1);
      operand_b.fmt = fmt;

      for (int pos_c = 0; pos_c < c; pos_c += step_c) {
        int cur_c = std::min(c - pos_c, step_c);
        for (int pos_n = 0; pos_n < n; pos_n += step_n) {
          int cur_n = std::min(n - pos_n, step_n);

          llvm::errs() << llvm::format("cur_n:%d, cur_c:%d, cur_h:%d, cur_w:%d\n", cur_n,
                                       cur_c, cur_h, cur_w);
          cvk_tl_t operand_a, operand_res, operand_res_h;
          shape_a = ctx.tl_shape_t4(cur_n, cur_c, cur_h, cur_w);
          operand_a.start_address = tl_a->start_address; // start of lmem
          operand_a.shape = shape_a;
          operand_a.stride = ctx.tl_default_stride(shape_a, fmt, /*eu_align=*/1);
          operand_a.fmt = fmt;
          cvk_tg_stride_t g_stride = ctx.tg_default_stride(c, h, w, fmt);
          uint64_t a_offset =
              (pos_n * c * h * w + pos_c * h * w + pos_h * w + pos_w) * elt_size;
          ctx.tdma_load_stride(&operand_a, ga_inputs[0] + a_offset, g_stride);
          llvm::errs() << llvm::format(
              "load a, addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
              operand_a.start_address, shape_a.n, shape_a.c, shape_a.h, shape_a.w,
              g_stride.n, g_stride.c, g_stride.h, g_stride.w, a_offset);

          operand_res.start_address = tl_a->start_address; // start of lmem
          operand_res.shape = shape_a;
          operand_res.stride =
              ctx.tl_default_stride(shape_a, fmt, /*eu_align=*/1);
          operand_res.fmt = fmt;
          if (fmt == CVK_FMT_BF16) {
            if (type == BroadcastType::BroadcastAdd) {
              cvk_tiu_add_param_t p = {0};
              p.res_high = 0;
              p.res_low = &operand_res;
              p.a_high = 0;
              p.a_low = &operand_a;
              p.b_is_const = 0;
              p.b.high = 0;
              p.b.low = &operand_b;
              p.rshift_bits = 0;
              p.layer_id = layer_id;
              p.relu_enable = 0;
              ctx.tiu_add(&p);
            } else if (type == BroadcastType::BroadcastSub) {
              cvk_tiu_sub_param_t p = {0};
              p.res_high = 0;
              p.res_low = &operand_res;
              p.a_high = 0;
              p.a_low = &operand_a;
              p.b_high = 0;
              p.b_low = &operand_b;
              p.rshift_bits = 0;
              p.layer_id = layer_id;
              ctx.tiu_sub(&p);
            }
          } else if (fmt == CVK_FMT_I8) {
            operand_res_h.start_address = tl_a_h->start_address; // start of lmem
            operand_res_h.shape = shape_a;
            operand_res_h.stride =
                ctx.tl_default_stride(shape_a, fmt, /*eu_align=*/1);
            operand_res_h.fmt = fmt;

            cvk_tiu_mul_param_t p1 = {0};
            p1.res_high = &operand_res_h;
            p1.res_low = &operand_res;
            p1.a = &operand_a;
            p1.b_const.val = multipliers[0];
            p1.b_const.is_signed = true;
            p1.b_is_const = true;
            p1.rshift_bits = 0;
            p1.layer_id = layer_id;
            p1.relu_enable = 0;
            ctx.tiu_mul(&p1);

            cvk_tiu_mac_param_t p2 = {0};
            p2.res_high = &operand_res_h;
            p2.res_low = &operand_res;
            p2.a = &operand_b;
            p2.res_is_int8 = true;

            if (type == BroadcastType::BroadcastAdd) {
              p2.b_const.val = multipliers[1];
            } else if (type == BroadcastType::BroadcastSub) {
              p2.b_const.val = -multipliers[1];
            }

            p2.b_is_const = true;
            p2.b_const.is_signed = true;
            p2.lshift_bits = 0;
            p2.rshift_bits = rshift;
            p2.layer_id = layer_id;
            p2.relu_enable = do_relu;
            ctx.tiu_mac(&p2);
          }
          // store result
          ctx.tdma_store_stride(&operand_res, ga_output + a_offset, g_stride);
          llvm::errs() << llvm::format(
              "store, addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
              operand_res.start_address, shape_a.n, shape_a.c, shape_a.h, shape_a.w,
              g_stride.n, g_stride.c, g_stride.h, g_stride.w, a_offset);
        }
      }
    }
  }

  if (tl_a_h) {
    ctx.lmem_free_tensor(tl_a_h);
  }

  ctx.lmem_free_tensor(tl_b);
  ctx.lmem_free_tensor(tl_a);
}

void cvi_backend_tg_int8_broadcast_add_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                         gaddr_t ga_inputs[], gaddr_t ga_output,
                         int32_t n, int32_t c, int32_t h, int32_t w,
                         int32_t bn, int32_t bc, int32_t bh, int32_t bw,
                         bool do_relu, const int32_t rshift,
                         const int32_t *multipliers) {
  tg_broadcast_kernel(ctx, layer_id, ga_inputs, ga_output,
                       n, c, h, w, bn, bc, bh, bw, do_relu, rshift, multipliers, BroadcastType::BroadcastAdd, CVK_FMT_I8);
}

void cvi_backend_tg_int8_broadcast_sub_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                         gaddr_t ga_inputs[], gaddr_t ga_output,
                         int32_t n, int32_t c, int32_t h, int32_t w,
                         int32_t bn, int32_t bc, int32_t bh, int32_t bw,
                         bool do_relu, const int32_t rshift,
                         const int32_t *multipliers) {
  tg_broadcast_kernel(ctx, layer_id, ga_inputs, ga_output,
                       n, c, h, w, bn, bc, bh, bw, do_relu, rshift, multipliers, BroadcastType::BroadcastSub, CVK_FMT_I8);
}

void cvi_backend_tg_bf16_broadcast_add_kernel(const CviBackendContext &ctx,
                                              uint32_t layer_id, gaddr_t ga_inputs[],
                                              gaddr_t ga_output, int n, int c, int h,
                                              int w, int bn, int bc, int bh, int bw,
                                              bool do_relu) {
  tg_broadcast_kernel(ctx, layer_id, ga_inputs, ga_output,
                        n, c, h, w, bn, bc, bh, bw, do_relu, 0, nullptr, BroadcastType::BroadcastAdd, CVK_FMT_BF16);
}

void cvi_backend_tg_bf16_broadcast_sub_kernel(const CviBackendContext &ctx,
                                              uint32_t layer_id, gaddr_t ga_inputs[],
                                              gaddr_t ga_output, int n, int c, int h,
                                              int w, int bn, int bc, int bh, int bw,
                                              bool do_relu) {
  tg_broadcast_kernel(ctx, layer_id, ga_inputs, ga_output,
                        n, c, h, w, bn, bc, bh, bw, do_relu, 0, nullptr, BroadcastType::BroadcastSub, CVK_FMT_BF16);
}