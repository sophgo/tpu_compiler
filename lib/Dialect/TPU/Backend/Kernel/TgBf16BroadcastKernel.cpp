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

#define DEBUG_TYPE "TgBf16BroadcastKernel"
void broadcast_one_to_all_lane(const CviBackendContext &ctx, cvk_tl_t inputBuffer) {
  assert(inputBuffer.shape.h * inputBuffer.shape.w < ((1 << 16 )- 1));//Reg bit field = 16 bit
  // reshape
  cvk_tl_t tl_src;
  tl_src.start_address = inputBuffer.start_address;
  tl_src.fmt = CVK_FMT_BF16;
  tl_src.shape = ctx.tl_shape_t4(inputBuffer.shape.n, 1, NPU_NUM, inputBuffer.shape.h * inputBuffer.shape.w);
  tl_src.stride = ctx.tl_default_stride(tl_src.shape, CVK_FMT_BF16, /*eu_align=*/1);
  tl_src.stride.h = 0;

  cvk_tl_t tl_dst;
  tl_dst.start_address = inputBuffer.start_address;
  tl_dst.fmt = CVK_FMT_BF16;
  tl_dst.shape = ctx.tl_shape_t4(inputBuffer.shape.n, NPU_NUM, 1, inputBuffer.shape.h * inputBuffer.shape.w);
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, CVK_FMT_BF16, /*eu_align=*/1);

  cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
  p2.src = &tl_src;
  p2.dst = &tl_dst;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "         L2L Reshape:\n"
                  "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                  "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                  p2.src->start_address, p2.src->shape.n,
                  p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                  p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                  p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                  p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
  ctx.tdma_l2l_tensor_copy(&p2);
}

void cvi_backend_tg_bf16_broadcast_sub_kernel(const CviBackendContext &ctx,
                                              uint32_t layer_id, gaddr_t ga_inputs[],
                                              gaddr_t ga_output, int n, int c, int h,
                                              int w, int bn, int bc, int bh, int bw,
                                              bool do_relu) {

  llvm::errs() << llvm::format("a shape:[%d,%d,%d,%d] b shape:[%d,%d,%d,%d], relu:%d\n",
                               n, c, h, w, bn, bc, bh, bw, do_relu);
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

  // determin the shape of tile.
  for (step_w = w; step_w > 0; step_w--) {
    for (step_h = h; step_h > 0; step_h--) {
      for (step_n = n; step_n > 0; step_n--) {
        shape_a = ctx.tl_shape_t4(step_n, step_c, step_h, step_w);
        shape_b = ctx.tl_shape_t4(1, 1, step_h, step_w);
        lmem_required = ctx.lmem_tensor_to_size(shape_a, CVK_FMT_BF16, 1) * 2 +
                        ctx.lmem_tensor_to_size(shape_b, CVK_FMT_BF16, 1);
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

  cvk_tl_t *tl_a = ctx.lmem_alloc_tensor(shape_a, CVK_FMT_BF16, 1);
  cvk_tl_t *tl_b = ctx.lmem_alloc_tensor(shape_b, CVK_FMT_BF16, 1);
  cvk_tl_t *tl_o = ctx.lmem_alloc_tensor(shape_a, CVK_FMT_BF16, 1);
  assert(tl_a && tl_b && tl_o);

  for (int pos_h = 0; pos_h < h; pos_h += step_h) {
    int cur_h = std::min(h - pos_h, step_h);
    for (int pos_w = 0; pos_w < w; pos_w += step_w) {
      int cur_w = std::min(w - pos_w, step_w);
      shape_b = ctx.tl_shape_t4(1, 1, cur_h, cur_w);

      // load b to lmem
      cvk_tl_t operand;
      uint64_t b_offset = (pos_w + pos_h * bw) * sizeof(uint16_t);
      operand.start_address = tl_b->start_address;
      operand.shape = shape_b;
      operand.stride = ctx.tl_default_stride(shape_b, CVK_FMT_BF16, 1);
      operand.fmt = CVK_FMT_BF16;
      cvk_tg_stride_t stride = ctx.tg_default_stride(bc, bh, bw, CVK_FMT_BF16);
      ctx.tdma_load_stride(&operand, ga_inputs[1] + b_offset, stride);

      llvm::errs() << llvm::format(
          "load b, addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
          operand.start_address, shape_b.n, shape_b.c, shape_b.h, shape_b.w, stride.n,
          stride.c, stride.h, stride.w, b_offset);

      // broadcast b to all lanes
      broadcast_one_to_all_lane(ctx, operand);
      shape_b = ctx.tl_shape_t4(1, NPU_NUM, cur_h, cur_w);
      cvk_tl_t operand_b;
      operand_b.start_address = tl_b->start_address;
      operand_b.shape = shape_b;
      operand_b.stride = ctx.tl_default_stride(shape_b, CVK_FMT_BF16, /*eu_align=*/1);
      operand_b.fmt = CVK_FMT_BF16;

      for (int pos_c = 0; pos_c < c; pos_c += step_c) {
        int cur_c = std::min(c - pos_c, step_c);
        for (int pos_n = 0; pos_n < n; pos_n += step_n) {
          int cur_n = std::min(n - pos_n, step_n);

          llvm::errs() << llvm::format("cur_n:%d, cur_c:%d, cur_h:%d, cur_w:%d\n", cur_n,
                                       cur_c, cur_h, cur_w);
          cvk_tl_t operand_a, operand_res;
          shape_a = ctx.tl_shape_t4(cur_n, cur_c, cur_h, cur_w);
          operand_a.start_address = tl_a->start_address; // start of lmem
          operand_a.shape = shape_a;
          operand_a.stride = ctx.tl_default_stride(shape_a, CVK_FMT_BF16, /*eu_align=*/1);
          operand_a.fmt = CVK_FMT_BF16;
          cvk_tg_stride_t g_stride = ctx.tg_default_stride(c, h, w, CVK_FMT_BF16);
          uint64_t a_offset =
              (pos_n * c * h * w + pos_c * h * w + pos_h * w + pos_w) * sizeof(uint16_t);
          ctx.tdma_load_stride(&operand_a, ga_inputs[0] + a_offset, g_stride);
          llvm::errs() << llvm::format(
              "load a, addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
              operand_a.start_address, shape_a.n, shape_a.c, shape_a.h, shape_a.w,
              g_stride.n, g_stride.c, g_stride.h, g_stride.w, a_offset);

          operand_res.start_address = tl_a->start_address; // start of lmem
          operand_res.shape = shape_a;
          operand_res.stride =
              ctx.tl_default_stride(shape_a, CVK_FMT_BF16, /*eu_align=*/1);
          operand_res.fmt = CVK_FMT_BF16;

          cvk_tiu_sub_param_t p5 = {0};
          p5.res_high = 0;
          p5.res_low = &operand_res;
          p5.a_high = 0;
          p5.a_low = &operand_a;
          p5.b_high = 0;
          p5.b_low = &operand_b;
          p5.rshift_bits = 0;
          p5.layer_id = layer_id;
          ctx.tiu_sub(&p5);

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
  ctx.lmem_free_tensor(tl_o);
  ctx.lmem_free_tensor(tl_b);
  ctx.lmem_free_tensor(tl_a);
}