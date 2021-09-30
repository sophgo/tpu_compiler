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
    param.ins_val = -128;
    param.ins_fp = 0xff7f;
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

// ReduceMin along C axis:
//   1. Tensor load (N, C, H, W) -> (N, 1, C, H*W)
//   2. Min pooling, kernel (C, 1), (N, 1, C, H*W) -> (N, 1, 1, H*W)
//   3. Tensor store, (N, 1, 1, H*W) -> (N, 1, 1, H*W)
static void cvi_backend_tg_fixed_reduce_min_chl_kernel(
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
    param.ins_val = -128;
    param.ins_fp = 0xff7f;
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
  if (kh == 1) {
    c = n * c * h;
    n = 1;
    h = 1;
  }
  int oh = h / kh;
  int ow = w / kw;
  cvk_fmt_t fmt = CVK_FMT_I8;
  int step_c = std::min(c, MAX_CHANNEL);
  while (step_c > 0) {
    auto shape_in = ctx.tl_shape_t4(n, step_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, step_c, oh, ow);
    uint32_t lmem_need = ctx.lmem_tensor_to_size(shape_in, fmt, 1) +
                         ctx.lmem_tensor_to_size(shape_out, fmt, 1);
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (step_c % NPU_NUM == 0) {
      step_c -= NPU_NUM;
    } else {
      step_c -= (step_c % NPU_NUM);
    }
  }
  auto in_gstride = ctx.tg_default_stride(c, h, w, fmt);
  auto out_gstride = ctx.tg_default_stride(c, oh, ow, fmt);
  assert(step_c > 0 && "tiling failed");
  for (int pos_c = 0; pos_c < c; pos_c += step_c) {
    int cur_c = std::min(step_c, c - pos_c);
    auto shape_in = ctx.tl_shape_t4(n, cur_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, cur_c, oh, ow);
    auto tl_input = ctx.lmem_alloc_tensor(shape_in, fmt, 1);
    auto tl_output = ctx.lmem_alloc_tensor(shape_out, fmt, 1);
    ctx.tdma_load_stride(tl_input, ga_input + pos_c * in_gstride.c, in_gstride);
    // 2. Max
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
      param.ins_val = -128;
      param.ins_fp = 0xff7f;
      param.layer_id = layer_id;

      ctx.tiu_max_pooling(&param);
    }
    ctx.tdma_store_stride(tl_output, ga_output + pos_c * out_gstride.c, out_gstride);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }



}

static void cvi_backend_tg_fixed_reduce_min_hw_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int kh, int kw) {
  if (kh == 1) {
    c = n * c * h;
    n = 1;
    h = 1;
  }

  int oh = h / kh;
  int ow = w / kw;
  cvk_fmt_t fmt = CVK_FMT_I8;
  int step_c = std::min(c, MAX_CHANNEL);
  while (step_c > 0) {
    auto shape_in = ctx.tl_shape_t4(n, step_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, step_c, oh, ow);
    uint32_t lmem_need = ctx.lmem_tensor_to_size(shape_in, fmt, 1) +
                         ctx.lmem_tensor_to_size(shape_out, fmt, 1);
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (step_c % NPU_NUM == 0) {
      step_c -= NPU_NUM;
    } else {
      step_c -= (step_c % NPU_NUM);
    }
  }

  auto in_gstride = ctx.tg_default_stride(c, h, w, fmt);
  auto out_gstride = ctx.tg_default_stride(c, oh, ow, fmt);
  assert(step_c > 0 && "tiling failed");
  for (int pos_c = 0; pos_c < c; pos_c += step_c) {
    int cur_c = std::min(step_c, c - pos_c);
    auto shape_in = ctx.tl_shape_t4(n, cur_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, cur_c, oh, ow);
    auto tl_input = ctx.lmem_alloc_tensor(shape_in, fmt, 1);
    auto tl_output = ctx.lmem_alloc_tensor(shape_out, fmt, 1);
    ctx.tdma_load_stride(tl_input, ga_input + pos_c * in_gstride.c, in_gstride);

    // Min
    {
      int mul_const = -1;
      int relu_enable = 0;
      int shift_bits = 0;

      // tl_input = tl_input * -1
      cvk_tiu_mul_param_t p_in = {0};
      p_in.res_high = NULL;
      p_in.res_low = tl_input;
      p_in.a = tl_input;
      p_in.b_is_const = 1;
      p_in.b_const.val = mul_const;
      p_in.b_const.is_signed = 1;
      p_in.rshift_bits = shift_bits;
      p_in.relu_enable = relu_enable;
      ctx.tiu_mul(&p_in);

      // tl_output = maxpooling(tl_input)
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
      param.ins_val = -128;
      param.ins_fp = 0xff7f;
      param.layer_id = layer_id;
      ctx.tiu_max_pooling(&param);

      // tl_output = tl_output * -1
      cvk_tiu_mul_param_t p_out = {0};
      p_out.res_high = NULL;
      p_out.res_low = tl_output;
      p_out.a = tl_output;
      p_out.b_is_const = 1;
      p_out.b_const.val = mul_const;
      p_out.b_const.is_signed = 1;
      p_out.rshift_bits = shift_bits;
      p_out.relu_enable = relu_enable;
      ctx.tiu_mul(&p_out);
    }

    ctx.tdma_store_stride(tl_output, ga_output + pos_c * out_gstride.c, out_gstride);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
}

static void cvi_backend_tg_fixed_reduce_mean_hw_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_output, int n, int c, int h, int w, int kh, int kw, int rshift,
    int multiplier) {
  if (kh == 1) {
    c = n * c * h;
    n = 1;
    h = 1;
  }
  int oh = h / kh;
  int ow = w / kw;
  cvk_fmt_t fmt = CVK_FMT_I8;
  int step_c = std::min(c, MAX_CHANNEL);
  while (step_c > 0) {
    auto shape_in = ctx.tl_shape_t4(n, step_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, step_c, oh, ow);
    uint32_t lmem_need = ctx.lmem_tensor_to_size(shape_in, fmt, 1) +
                         ctx.lmem_tensor_to_size(shape_out, fmt, 1);
    if (lmem_need <= (uint32_t)LOCAL_MEM_SIZE) {
      break;
    }
    if (step_c % NPU_NUM == 0) {
      step_c -= NPU_NUM;
    } else {
      step_c -= (step_c % NPU_NUM);
    }
  }
  auto in_gstride = ctx.tg_default_stride(c, h, w, fmt);
  auto out_gstride = ctx.tg_default_stride(c, oh, ow, fmt);
  assert(step_c > 0 && "tiling failed");
  for (int pos_c = 0; pos_c < c; pos_c += step_c) {
    int cur_c = std::min(step_c, c - pos_c);
    auto shape_in = ctx.tl_shape_t4(n, cur_c, h, w);
    auto shape_out = ctx.tl_shape_t4(n, cur_c, oh, ow);
    auto tl_input = ctx.lmem_alloc_tensor(shape_in, fmt, 1);
    auto tl_output = ctx.lmem_alloc_tensor(shape_out, fmt, 1);
    ctx.tdma_load_stride(tl_input, ga_input + pos_c * in_gstride.c, in_gstride);
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
    param.ins_val = param.avg_pooling_const;
    param.ins_fp = 0;
    ctx.tiu_average_pooling(&param);
    ctx.tdma_store_stride(tl_output, ga_output + pos_c * out_gstride.c, out_gstride);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
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

void cvi_backend_tg_fixed_reduce_min_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int n, int c, int h, int w,
    int axes[], int num_axes) {
  if (axes[0] == 1) {
    // along C axis
    cvi_backend_tg_fixed_reduce_min_chl_kernel(ctx,
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
    cvi_backend_tg_fixed_reduce_min_hw_kernel(ctx,
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
  int kh = h;
  int kw = w;
  if (num_axes == 2) {
    if (axes[0] == 2 && axes[1] == 3) {
      // along H and W axis
      kh = h;
      kw = w;
    } else {
      assert(0 && "Not support axis for reducemean yet");
    }
  } else if (num_axes == 1) {
    if (axes[0] == 2 || axes[0] == 3) {
      // along H or W axis
      kh = (axes[0] == 2) ? h : 1;
      kw = (axes[0] == 3) ? w : 1;
    } else {
      assert(0 && "Not support axis for reducemean yet");
    }
  } else {
    assert(0 && "Not support axis for reducemean yet");
  }

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
