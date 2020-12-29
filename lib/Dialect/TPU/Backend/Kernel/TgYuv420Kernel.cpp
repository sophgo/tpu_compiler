/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgSwapChannelKernel.cpp
 * Description:
 */

#include "TgYuv420Kernel.hpp"

// input = [n, 6, h/2, w/2]
// &tl_y = [n, 1, h, w]
// &tl_u = [n, 1, h/2, w/2]
// &tl_v = [n, 1, h/2, w/2]
// output = [n, 3, h, w]

// r = y + 1.402 * (v - 128);
// g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128);
// b = y + 1.772 * (u - 128);
//

void TgYuv420Kernel::init(uint32_t layer_id, gaddr_t ga_input,
                          gaddr_t ga_output, int n, int c, int h, int w,
                          const std::vector<int> &order, int channel_align,
                          int w_align, cvk_fmt_t fmt) {
  assert(c == 3 && "rgb channel must be 3");
  // convert to output shape
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  if (order.empty()) {
    this->order.push_back(0);
    this->order.push_back(1);
    this->order.push_back(2);
  } else {
    this->order = order;
    assert(order.size() == 3);
    for (auto &channel : order) {
      assert(channel < 3 && channel >= 0);
    }
  }
  this->fmt = fmt;
  if (fmt == CVK_FMT_I8) {
    // only support u8, regard i8 as u8
    fmt = CVK_FMT_U8;
  }
  assert(fmt == CVK_FMT_U8);
  y_w_aligned = align_up(w, w_align);
  uv_w_aligned = align_up(w / 2, w_align);
  int y_offset = 0;
  int u_offset = align_up(y_offset + y_w_aligned * h, channel_align);
  int v_offset = align_up(u_offset + uv_w_aligned * h / 2, channel_align);
  n_stride = align_up(v_offset + uv_w_aligned * h / 2, channel_align);
  ga_y = ga_input + y_offset;
  ga_u = ga_input + u_offset;
  ga_v = ga_input + v_offset;

  y_gstride = {static_cast<uint32_t>(n_stride),
               static_cast<uint32_t>(2 * y_w_aligned),
               static_cast<uint32_t>(2),
               static_cast<uint32_t>(1)};
  uv_gstride = {static_cast<uint32_t>(n_stride),
                static_cast<uint32_t>(uv_w_aligned),
                static_cast<uint32_t>(1),
                static_cast<uint32_t>(1)};
  rgb_gstride = {static_cast<uint32_t>(h * w * 3),
                 static_cast<uint32_t>(2 * w),
                 static_cast<uint32_t>(2),
                 static_cast<uint32_t>(1)};
}

void TgYuv420Kernel::allocLmem() {
  cvk_tl_shape_t shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  for (int i = 0; i < BLOB_NUM; i++) {
    tl_mem[i] = ctx.lmem_alloc_tensor(shape, CVK_FMT_BF16, 1);
  }
}

void TgYuv420Kernel::deallocLmem() {
  for (int i = BLOB_NUM - 1; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
}

void TgYuv420Kernel::selectTilePolicy() {
  ctx.tiling_packing(tiles, n, h / 2, w / 2, 1, CVK_FMT_BF16, BLOB_NUM, 0,
                     CviBackendContext::TilingNCHW, true);
}

void TgYuv420Kernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx / 4];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  cvk_tl_stride_t stride = ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);
  for (int i = 0; i < BLOB_NUM; i++) {
    tl_mem[i]->shape = shape;
    tl_mem[i]->stride = stride;
  }
  tl_y = *tl_mem[0 + (step_idx % 2)];
  tl_u = *tl_mem[2 + (step_idx / 4 % 2)];
  tl_v = *tl_mem[4 + (step_idx / 4 % 2)];
  tl_r = *tl_mem[6 + (step_idx % 2)];
  tl_g = *tl_mem[8 + (step_idx % 2)];
  tl_b = *tl_mem[10 + (step_idx % 2)];
}

void TgYuv420Kernel::load_u8_to_bf16(cvk_tl_t *dst, uint64_t src_gaddr,
                                     cvk_tg_stride_t stride) {
  cvk_tg_t src;
  src.start_address = src_gaddr;
  src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
  src.int8_rnd_mode = 0;
  src.fmt = fmt; // CVK_FMT_U8
  src.shape = {dst->shape.n, dst->shape.c, dst->shape.h, dst->shape.w};
  src.stride = stride;
  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = dst;
  ctx.tdma_g2l_tensor_copy(&p);
}

void TgYuv420Kernel::store_bf16_to_u8(cvk_tl_t *src, uint64_t dst_gaddr,
                                      cvk_tg_stride_t stride) {

  cvk_tg_t dst;
  dst.start_address = dst_gaddr;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
  dst.fmt = fmt; // CVK_FMT_U8
  dst.shape = {src->shape.n, src->shape.c, src->shape.h, src->shape.w};
  dst.stride = stride;

  cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
  p1.src = src;
  p1.dst = &dst;
  ctx.tdma_l2g_tensor_copy(&p1);
}

void TgYuv420Kernel::load(int32_t step_idx) {
  refresh(step_idx);
  auto &tile = tiles[step_idx / 4];
  int y_index = step_idx % 4;
  if (y_index == 0) {
    uint64_t u_gaddr =
        ga_u + tile.pos_n * n_stride + tile.pos_c * uv_w_aligned + tile.pos_h;
    uint64_t v_gaddr =
        ga_v + tile.pos_n * n_stride + tile.pos_c * uv_w_aligned + tile.pos_h;
    load_u8_to_bf16(&tl_u, u_gaddr, uv_gstride);
    load_u8_to_bf16(&tl_v, v_gaddr, uv_gstride);
  }
  // load &tl_y
  uint64_t y_gaddr = ga_y + tile.pos_n * n_stride +
                     (tile.pos_c * 2 + y_index / 2) * y_w_aligned +
                     tile.pos_h * 2 + y_index % 2;
  load_u8_to_bf16(&tl_y, y_gaddr, y_gstride);
}

void TgYuv420Kernel::store(int32_t step_idx) {
  refresh(step_idx);
  auto &tile = tiles[step_idx / 4];
  cvk_tl_t *bgr[3] = {&tl_b, &tl_g, &tl_r};
  int y_index = step_idx % 4;
  uint64_t rgb_offset = tile.pos_n * 3 * h * w +
                        (tile.pos_c * 2 + y_index / 2) * w + tile.pos_h * 2 +
                        y_index % 2;
  for (int n_idx = 0; n_idx < tile.n; n_idx++) {
    for (int order_idx = 0; order_idx < 3; order_idx++) {
      int c = order[order_idx];
      bgr[c]->shape.n = 1;
      uint64_t rgb_gaddr =
          ga_output + rgb_offset + (n_idx * 3 + order_idx) * h * w;
      store_bf16_to_u8(bgr[c], rgb_gaddr, rgb_gstride);
      bgr[c]->start_address += bgr[c]->stride.n;
    }
  }
}

void TgYuv420Kernel::compute(int32_t step_idx) {
  refresh(step_idx);
  if (step_idx % 4 == 0) {
    cvk_tiu_add_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_u;
    p1.a_high = nullptr;
    p1.a_low = &tl_u;
    p1.b_is_const = true;
    p1.b_const.val = ctx.convert_fp32_to_bf16(-128.0f);
    p1.b_const.is_signed = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = false;
    ctx.tiu_add(&p1);
    cvk_tiu_add_param_t p2 = {0};
    p2.res_high = nullptr;
    p2.res_low = &tl_v;
    p2.a_high = nullptr;
    p2.a_low = &tl_v;
    p2.b_is_const = true;
    p2.b_const.val = ctx.convert_fp32_to_bf16(-128.0f);
    p2.b_const.is_signed = 1;
    p2.rshift_bits = 0;
    p2.layer_id = layer_id;
    p2.relu_enable = false;
    ctx.tiu_add(&p2);
  }
  // &tl_r
  cvk_tiu_copy_param_t p3 = {0};
  p3.src = &tl_y;
  p3.dst = &tl_r;
  p3.layer_id = layer_id;
  ctx.tiu_copy(&p3);
  cvk_tiu_mac_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_r;
  p4.res_is_int8 = 0;
  p4.a = &tl_v;
  p4.b_const.val = ctx.convert_fp32_to_bf16(1.402f);
  p4.b_is_const = 1;
  p4.b_const.is_signed = 1;
  p4.lshift_bits = 0;
  p4.rshift_bits = 0;
  p4.relu_enable = 1;
  ctx.tiu_mac(&p4);
  cvk_tiu_min_param_t pmin = {0};
  pmin.min = &tl_r;
  pmin.a = &tl_r;
  pmin.b_is_const = 1;
  pmin.b_const.val = ctx.convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  ctx.tiu_min(&pmin);

  // &tl_b
  cvk_tiu_copy_param_t p5 = {0};
  p5.src = &tl_y;
  p5.dst = &tl_b;
  p5.layer_id = layer_id;
  ctx.tiu_copy(&p5);
  cvk_tiu_mac_param_t p6 = {0};
  p6.res_high = nullptr;
  p6.res_low = &tl_b;
  p6.res_is_int8 = 0;
  p6.a = &tl_u;
  p6.b_const.val = ctx.convert_fp32_to_bf16(1.772f);
  p6.b_is_const = 1;
  p6.b_const.is_signed = 1;
  p6.lshift_bits = 0;
  p6.rshift_bits = 0;
  p6.relu_enable = 1;
  ctx.tiu_mac(&p6);
  pmin.min = &tl_b;
  pmin.a = &tl_b;
  pmin.b_is_const = 1;
  pmin.b_const.val = ctx.convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  ctx.tiu_min(&pmin);

  // &tl_g
  cvk_tiu_copy_param_t p7 = {0};
  p7.src = &tl_y;
  p7.dst = &tl_g;
  p7.layer_id = layer_id;
  ctx.tiu_copy(&p7);
  cvk_tiu_mac_param_t p8 = {0};
  p8.res_high = nullptr;
  p8.res_low = &tl_g;
  p8.res_is_int8 = 0;
  p8.a = &tl_u;
  p8.b_const.val = ctx.convert_fp32_to_bf16(-0.34414f);
  p8.b_is_const = 1;
  p8.b_const.is_signed = 1;
  p8.lshift_bits = 0;
  p8.rshift_bits = 0;
  p8.relu_enable = 0;
  ctx.tiu_mac(&p8);
  cvk_tiu_mac_param_t p9 = {0};
  p9.res_high = nullptr;
  p9.res_low = &tl_g;
  p9.res_is_int8 = 0;
  p9.a = &tl_v;
  p9.b_const.val = ctx.convert_fp32_to_bf16(-0.71414f);
  p9.b_is_const = 1;
  p9.b_const.is_signed = 1;
  p9.lshift_bits = 0;
  p9.rshift_bits = 0;
  p9.relu_enable = 1;
  ctx.tiu_mac(&p9);
  pmin.min = &tl_g;
  pmin.a = &tl_g;
  pmin.b_is_const = 1;
  pmin.b_const.val = ctx.convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  ctx.tiu_min(&pmin);
}

void TgYuv420Kernel::schedule() {
  allocLmem();
  int32_t total_steps = 4 * tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    ctx.parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_yuv420_csc_kernel(const CviBackendContext &ctx,
                                      uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, int n, int c, int h,
                                      int w, const std::vector<int> &order,
                                      cvk_fmt_t fmt) {
  TgYuv420Kernel kernel(ctx);
  // yuv channel align is 4KB, w_align is 32B
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, order, 0x1000,
              0x20, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}