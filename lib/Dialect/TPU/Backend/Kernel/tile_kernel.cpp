#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"


static void tile_w(const CviBackendContext &ctx,
    gaddr_t input_gaddr, cvk_tg_shape_t input_shape, cvk_fmt_t input_fmt,
    gaddr_t output_gaddr, cvk_tg_shape_t output_shape, cvk_fmt_t output_fmt,
    int w_factor, uint32_t layer_id) {
      cvk_tg_shape_t src_shape = input_shape;
      src_shape.n = src_shape.n * src_shape.c;
      src_shape.c = src_shape.h;
      src_shape.h = 1;
      cvk_tg_stride_t src_stride = ctx.tg_default_stride(src_shape, input_fmt);
      src_shape.h = w_factor; // fake, just prevent shape size eq check
      src_stride.h = 0;                // stall for copy w

      cvk_tg_shape_t dst_shape = output_shape;
      dst_shape.n = src_shape.n;
      dst_shape.c = src_shape.c;
      dst_shape.h = w_factor;
      dst_shape.w = src_shape.w;
      cvk_tg_stride_t dst_stride = ctx.tg_default_stride(output_shape, output_fmt);
      ctx.set_layer_id(layer_id);
      ctx.tdma_g2g_tensor_copy(input_gaddr, src_shape, src_stride,
                               output_gaddr, dst_shape, dst_stride,
                               CVK_FMT_BF16);
}

int cvi_backend_tg_tile_kernel(const CviBackendContext &ctx,
    gaddr_t input_gaddr,
    int input_n, int input_c, int input_h, int input_w,
    cvk_fmt_t input_fmt,
    gaddr_t output_gaddr,
    int output_n, int output_c, int output_h, int output_w,
    cvk_fmt_t output_fmt,
    int* tile_factors, int tile_factors_len,
    uint32_t layer_id) {

  cvk_tg_shape_t input_shape = {(uint32_t)input_n, (uint32_t)input_c, (uint32_t)input_h, (uint32_t)input_w};
  cvk_tg_shape_t output_shape = {(uint32_t)output_n, (uint32_t)output_c, (uint32_t)output_h, (uint32_t)output_w};

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
      cvk_tg_shape_t src_shape = _input_shape;
      src_shape.n = src_shape.n * src_shape.c;
      src_shape.c = src_shape.h;
      src_shape.h = 1;
      cvk_tg_stride_t src_stride = ctx.tg_default_stride(src_shape, input_fmt);
      src_shape.h = tile_dims[DIMS_W]; // fake, just prevent shape size eq check
      src_stride.h = 0;                // stall for copy w

      cvk_tg_shape_t dst_shape = _output_shape;
      dst_shape.n = src_shape.n;
      dst_shape.c = src_shape.c;
      dst_shape.h = tile_dims[DIMS_W];
      dst_shape.w = src_shape.w;
      cvk_tg_stride_t dst_stride = ctx.tg_default_stride(output_shape, output_fmt);
      uint32_t dst_stride_n = dst_stride.c;
      dst_stride = ctx.tg_default_stride(dst_shape, output_fmt);
      dst_stride.n = dst_stride_n;
      ctx.set_layer_id(layer_id);
      ctx.tdma_g2g_tensor_copy(input_gaddr, src_shape, src_stride,
                               output_gaddr, dst_shape, dst_stride,
                               CVK_FMT_BF16);
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
      // tile h, cuz previous tile w placed by channel, we just copy in each
      // channel
      cvk_tg_shape_t src_shape = _input_shape;
      cvk_tg_shape_t dst_shape = _output_shape;
      cvk_tg_stride_t src_stride = ctx.tg_default_stride(src_shape, input_fmt);
      src_shape.h = tile_dims[DIMS_H]; // fake, just prevent shape size eq check
      src_stride.h = 0;                // stall for copy w
      dst_shape.h = tile_dims[DIMS_H];
      cvk_tg_stride_t dst_stride = ctx.tg_default_stride(dst_shape, output_fmt);
      src_stride.c = dst_stride.c;
      src_stride.n = dst_stride.n;
      ctx.set_layer_id(layer_id);
      ctx.tdma_g2g_tensor_copy(output_gaddr, src_shape, src_stride,
                               output_gaddr, dst_shape, dst_stride,
                               CVK_FMT_BF16);
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
