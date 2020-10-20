#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "TgFixedReduceKernel"

// ReduceMax along C axis:
//   1. Tensor load (N, C, H, W) -> (N, 1, C, H*W)
//   2. Max pooling, kernel (C, 1), (N, 1, C, H*W) -> (N, 1, 1, H*W)
//   3. Tensor store, (N, 1, 1, H*W) -> (N, 1, 1, H*W)
static void cvi_backend_tg_fixed_reduce_max_chl_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w) {
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(n, 1, c, h * w);
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(n, 1, 1, h * w);

  int eu_align = 0; // contiguous memory
  cvk_fmt_t fmt = CVK_FMT_I8;
  uint32_t neededSize =
      ctx.lmem_tensor_to_size(tl_input_shape, fmt, eu_align) +
      ctx.lmem_tensor_to_size(tl_output_shape, fmt, eu_align);

  assert(neededSize <= (uint32_t)LOCAL_MEM_SIZE && "Not support tiling yet");

  // 1. Tensor load, (N, 1, C, H * W)
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_input_shape, fmt, eu_align);
  cvi_backend_tl_load_stride(ctx,
                             layer_id,
                             ga_input,
                             tl_input->start_address,
                             n, 1, c, h * w,
                             1, c, h * w,
                             false,       // DoTranspose
                             eu_align,
                             true,        // isNeuron
                             fmt,
                             fmt
                             );

  // 2. Max or Mean, kernel (C, 1)
  cvk_tl_t *tl_output =
      ctx.lmem_alloc_tensor(tl_output_shape, fmt, eu_align);
  {
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = tl_output;
    param.ifmap = tl_input;
    param.kh = c;
    param.kw = 1;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.ins_fp = 0;
    param.layer_id = layer_id;

    ctx.tiu_max_pooling(&param);
  }

  // 3. Tensor store, (N, 1, 1, H * W)
  cvi_backend_tl_store_stride(ctx,
                              layer_id,
                              ga_output,
                              tl_output->start_address,
                              n, 1, 1, h * w,
                              1, 1, h * w,
                              false,    // DoTranspose
                              eu_align, // DoAligned
                              true,     // isNeuron
                              fmt,
                              fmt
                              );

  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}

static void cvi_backend_tg_fixed_reduce_max_hw_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int kh, int kw) {
  int oh = h / kh;
  int ow = w / kw;
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(n, c, h, w);
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(n, c, oh, ow);

  int eu_align = 1; // TIU always needs eu-aligned
  cvk_fmt_t fmt = CVK_FMT_I8;
  uint32_t neededSize =
      ctx.lmem_tensor_to_size(tl_input_shape, fmt, eu_align) +
      ctx.lmem_tensor_to_size(tl_output_shape, fmt, eu_align);

  assert(neededSize <= (uint32_t)LOCAL_MEM_SIZE && "Not support tiling yet");

  // 1. Tensor load, (N, C, H, W)
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_input_shape, fmt, eu_align);
  cvi_backend_tl_load_stride(ctx,
                             layer_id,
                             ga_input,
                             tl_input->start_address,
                             n, c, h, w,
                             c, h, w,
                             false,       // DoTranspose
                             eu_align,
                             true,        // isNeuron
                             fmt,
                             fmt
                             );

  // 2. Max
  cvk_tl_t *tl_output =
      ctx.lmem_alloc_tensor(tl_output_shape, fmt, eu_align);
  {
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = tl_output;
    param.ifmap = tl_input;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.ins_fp = 0;
    param.layer_id = layer_id;

    ctx.tiu_max_pooling(&param);
  }

  // 3. Tensor store, (N, C, OH, OW)
  cvi_backend_tl_store_stride(ctx,
                              layer_id,
                              ga_output,
                              tl_output->start_address,
                              n, c, oh, ow,
                              c, oh, ow,
                              false,    // DoTranspose
                              eu_align, // DoAligned
                              true,     // isNeuron
                              fmt,
                              fmt
                              );

  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}

static void cvi_backend_tg_fixed_reduce_mean_hw_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int kh, int kw,
    int rshift, int multiplier) {
  int oh = h / kh;
  int ow = w / kw;
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(n, c, h, w);
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(n, c, oh, ow);

  int eu_align = 1; // TIU always needs eu-aligned
  cvk_fmt_t fmt = CVK_FMT_I8;
  uint32_t neededSize =
      ctx.lmem_tensor_to_size(tl_input_shape, fmt, eu_align) +
      ctx.lmem_tensor_to_size(tl_output_shape, fmt, eu_align);

  assert(neededSize <= (uint32_t)LOCAL_MEM_SIZE && "Not support tiling yet");

  // 1. Tensor load, (N, C, H, W)
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_input_shape, fmt, eu_align);
  cvi_backend_tl_load_stride(ctx,
                             layer_id,
                             ga_input,
                             tl_input->start_address,
                             n, c, h, w,
                             c, h, w,
                             false,       // DoTranspose
                             eu_align,
                             true,        // isNeuron
                             fmt,
                             fmt
                             );

  // 2. Mean
  cvk_tl_t *tl_output =
      ctx.lmem_alloc_tensor(tl_output_shape, fmt, eu_align);
  {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = tl_output;
    param.ifmap = tl_input;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.avg_pooling_const = multiplier;
    param.rshift_bits = rshift;
    param.layer_id = layer_id;

    ctx.tiu_average_pooling(&param);
  }

  // 3. Tensor store, (N, C, OH, OW)
  cvi_backend_tl_store_stride(ctx,
                              layer_id,
                              ga_output,
                              tl_output->start_address,
                              n, c, oh, ow,
                              c, oh, ow,
                              false,    // DoTranspose
                              eu_align, // DoAligned
                              true,     // isNeuron
                              fmt,
                              fmt
                              );

  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}

void cvi_backend_tg_fixed_reduce_max_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int axes[], int num_axes) {
  if (axes[0] == 1) {
    // along C axis
    cvi_backend_tg_fixed_reduce_max_chl_kernel(ctx,
                                               layer_id,
                                               ga_input,
                                               ga_output,
                                               n,
                                               c,
                                               h,
                                               w);
  } else if (axes[0] == 2 || axes[0] == 3) {
    // along H or W axis
    int kh = (axes[0] == 2) ? h : 1;
    int kw = (axes[0] == 3) ? w : 1;
    cvi_backend_tg_fixed_reduce_max_hw_kernel(ctx,
                                              layer_id,
                                              ga_input,
                                              ga_output,
                                              n,
                                              c,
                                              h,
                                              w,
                                              kh,
                                              kw);
  }
}

void cvi_backend_tg_fixed_reduce_mean_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int rshift, int multiplier,
    int axes[], int num_axes) {
  if (axes[0] == 2 || axes[0] == 3) {
    // along H or W axis
    int kh = (axes[0] == 2) ? h : 1;
    int kw = (axes[0] == 3) ? w : 1;
    cvi_backend_tg_fixed_reduce_mean_hw_kernel(ctx,
                                               layer_id,
                                               ga_input,
                                               ga_output,
                                               n,
                                               c,
                                               h,
                                               w,
                                               kh,
                                               kw,
                                               rshift,
                                               multiplier);
  }
}
