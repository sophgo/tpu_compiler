#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"

// only fill base_reg_index/int8_rnd_mode
static void init_tgmem(const CviBackendContext &ctx, cvk_tg_t* t, gaddr_t addr) {
  t->base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(addr);
  t->int8_rnd_mode = 0;
}

static void tile_w(const CviBackendContext &ctx,
    gaddr_t input_gaddr, cvk_tg_shape_t input_shape, cvk_fmt_t input_fmt,
    gaddr_t output_gaddr, cvk_tg_shape_t output_shape, cvk_fmt_t output_fmt,
    int w_factor, uint32_t layer_id) {

  cvk_tdma_g2g_tensor_copy_param_t p;
  cvk_tg_t src, dst;

  init_tgmem(ctx, &src, input_gaddr);
  init_tgmem(ctx, &dst, output_gaddr);
  // tile w
  // we reshape nchw to nc, h, 1, w for align copy w
  src.fmt = input_fmt;
  src.start_address = input_gaddr;
  src.shape = input_shape;
  src.shape.n = src.shape.n * src.shape.c;
  src.shape.c = src.shape.h;
  src.shape.h = 1;
  src.stride = ctx.tg_default_stride(src.shape, src.fmt);
  src.shape.h = w_factor; // fake, just prevent shape size eq check
  src.stride.h = 0; // stall for copy w

  dst.fmt = output_fmt;
  dst.start_address = output_gaddr;
  dst.shape = output_shape;
  dst.shape.n = src.shape.n;
  dst.shape.c = src.shape.c;
  dst.shape.h = w_factor;
  dst.shape.w = src.shape.w;
  dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

  p.src = &src;
  p.dst = &dst;
  p.layer_id = layer_id;

  ctx.tdma_g2g_bf16_tensor_copy(&p);
}

int cvi_backend_tg_tile_kernel(const CviBackendContext &ctx,
    gaddr_t input_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    cvi_backend_fmt_t bk_input_fmt,
    gaddr_t output_gaddr,
    int output_n, int output_c, int output_h, int output_w,
    cvi_backend_fmt_t bk_output_fmt,
    int* tile_factors, int tile_factors_len,
    uint32_t layer_id) {

  cvk_tg_shape_t input_shape = {(uint32_t)input_n, (uint32_t)input_c, (uint32_t)input_h, (uint32_t)input_w};
  cvk_tg_shape_t output_shape = {(uint32_t)output_n, (uint32_t)output_c, (uint32_t)output_h, (uint32_t)output_w};
  cvk_fmt_t input_fmt = cvi_to_cvk_fmt(bk_input_fmt);
  cvk_fmt_t output_fmt = cvi_to_cvk_fmt(bk_output_fmt);

  // g2g
  assert(input_fmt == output_fmt && "only support input fmt = outpu fmt");
  assert(tile_factors_len <= 2 && "current only support w tile[1] and h tile[0]");
  if (!tile_factors_len) {
    return 0; // do nothing
  }

  enum dims {
    DIMS_N = 0,
    DIMS_C,
    DIMS_H,
    DIMS_W,
    DIMS_MAX,
  };

  int tile_dims[DIMS_MAX] = {1, 1, 1, 1};
  int dim_start = DIMS_MAX - tile_factors_len;
  int i;
  for (i = 0; i < tile_factors_len; i++) {
    assert(tile_factors[i] > 0 && "tile_factors should > 0");
    tile_dims[i + dim_start] = tile_factors[i];
  }


  if (tile_dims[DIMS_W] != 1 && tile_dims[DIMS_H] != 1) {
    // first tile w
    // second tile h
    // 1,2,3,4->1,2,6,8
    // => 1,2,3,4(src)->1,2,3,8(dst)
    // => 1,2,3,8(dst)->1,2,6,8(dst)
    // tile w
    cvk_tg_shape_t _input_shape = input_shape;
    cvk_tg_shape_t _output_shape = output_shape;
    _output_shape.h = _input_shape.h;

    {
      // 1 2 3 4 -> 1 2 3 4 1 2 3 4
      // 5 6 7 8    x x x x x x x x
      //            1 2 3 4 1 2 3 4
      //            x x x x x x x x
      cvk_tdma_g2g_tensor_copy_param_t p;
      cvk_tg_t src, dst;

      init_tgmem(ctx, &src, input_gaddr);
      init_tgmem(ctx, &dst, output_gaddr);
      // tile w
      // we reshape nchw to nc, h, 1, w for align copy w
      src.fmt = input_fmt;
      src.start_address = input_gaddr;
      src.shape = _input_shape;
      src.shape.n = src.shape.n * src.shape.c;
      src.shape.c = src.shape.h;
      src.shape.h = 1;
      src.stride = ctx.tg_default_stride(src.shape, src.fmt);
      src.shape.h = tile_dims[DIMS_W]; // fake, just prevent shape size eq check
      src.stride.h = 0; // stall for copy w

      dst.fmt = output_fmt;
      dst.start_address = output_gaddr;
      dst.shape = _output_shape;
      dst.shape.n = src.shape.n;
      dst.shape.c = src.shape.c;
      dst.shape.h = tile_dims[DIMS_W];
      dst.shape.w = src.shape.w;

      // shift next nc
      dst.stride = ctx.tg_default_stride(output_shape, dst.fmt);
      uint32_t dst_stride_n = dst.stride.c;

      dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);
      dst.stride.n = dst_stride_n;

      p.src = &src;
      p.dst = &dst;
      p.layer_id = layer_id;

      ctx.tdma_g2g_bf16_tensor_copy(&p);
    }

    // reshape to n, c, 1, hw
    _input_shape = input_shape;
    _output_shape = output_shape;

    _input_shape.w = input_shape.h * _output_shape.w;
    _input_shape.h = 1;
    _output_shape.w = _input_shape.w;
    _output_shape.h = tile_dims[DIMS_H];

    // copy itself
    {
      cvk_tdma_g2g_tensor_copy_param_t p;
      cvk_tg_t src, dst;

      init_tgmem(ctx, &src, input_gaddr);
      init_tgmem(ctx, &dst, output_gaddr);

      // tile h, cuz previous tile w placed by channel, we just copy in each channel
      src.fmt = input_fmt;
      src.start_address = output_gaddr;
      src.shape = _input_shape;
      src.stride = ctx.tg_default_stride(src.shape, src.fmt);

      src.shape.h = tile_dims[DIMS_H]; // fake, just prevent shape size eq check
      src.stride.h = 0; // stall for copy w

      dst.fmt = output_fmt;
      dst.start_address = output_gaddr;
      dst.shape = _output_shape;
      dst.shape.h = tile_dims[DIMS_H];
      dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

      // previous w-tile placed by c
      src.stride.c = dst.stride.c;
      src.stride.n = dst.stride.n;

      p.src = &src;
      p.dst = &dst;
      p.layer_id = layer_id;

      ctx.tdma_g2g_bf16_tensor_copy(&p);
    }
  }
  else {
    // tile align dims
    if (tile_dims[DIMS_W] != 1) {
      tile_w(ctx,
          input_gaddr, input_shape, input_fmt,
          output_gaddr, output_shape, output_fmt,
          tile_dims[DIMS_W], layer_id);
    }
    else if (tile_dims[DIMS_H] != 1) {
      // reshape to n, c, 1, hw
      cvk_tg_shape_t _input_shape = input_shape;
      cvk_tg_shape_t _output_shape = output_shape;

      _input_shape.w = _input_shape.h * _input_shape.w;
      _input_shape.h = 1;
      _output_shape.w = _output_shape.h * _output_shape.w;
      _output_shape.h = 1;

      tile_w(ctx,
          input_gaddr, _input_shape, input_fmt,
          output_gaddr, _output_shape, output_fmt,
          tile_dims[DIMS_H], layer_id);
    }
    else {
      assert(0 && "not reachable");
    }
  }

  return 0;

}
