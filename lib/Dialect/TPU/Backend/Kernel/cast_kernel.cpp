#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"

cvk_tl_stride_t tl_fp32_stride(const CviBackendContext &ctx, cvk_tl_t* tl, int eu_align) {
  int fmt_sz = 4; // 4 means fp32 takes 4 bytes
  cvk_tl_stride_t s;

  s.w = fmt_sz;
  s.h = tl->shape.w * fmt_sz;
  s.c = tl->shape.h * tl->shape.w * fmt_sz;

  if (eu_align) {
    s.c = align_up(s.c, EU_NUM);
  }

  s.n = s.c * ceiling_func(tl->shape.c, NPU_NUM);
  return s;
}

void lmem_shrink_fp32_bf16(const CviBackendContext &ctx,
    cvk_tl_t* lmem_bf16, cvk_tl_t* lmem_fp32,
    int bf16_n, int bf16_c, int bf16_h, int bf16_w, uint32_t layer_id) {

    assert((uint32_t)bf16_w * 2 == lmem_fp32->shape.w &&
        lmem_fp32->shape.h == (uint32_t)bf16_h &&
        lmem_fp32->shape.c == (uint32_t)bf16_c &&
        lmem_fp32->shape.n == (uint32_t)bf16_n &&
        "the fp32's width should be twice than bf16's"
        );

    // move high 16bit as bf16 format
    *lmem_bf16 = *lmem_fp32;
    lmem_bf16->shape = ctx.shape_t4(bf16_n, bf16_c, bf16_h, bf16_w);
    lmem_bf16->stride = ctx.tl_default_stride(lmem_bf16->shape, lmem_bf16->fmt, /*eu_align=*/0);

    // fake shape for cmodel constrain that shape SHOULD be equal
    lmem_fp32->shape = ctx.shape_t4(bf16_n, bf16_c, bf16_h, bf16_w);
    lmem_fp32->stride = tl_fp32_stride(ctx, lmem_fp32);

    laddr_t lmem_fp32_addr = lmem_fp32->start_address;
    lmem_fp32->start_address = lmem_fp32_addr + 2; // start with high 16 bits

    cvk_tiu_copy_param_t param = {0};
    param.src = lmem_fp32;
    param.dst = lmem_bf16;
    param.layer_id = layer_id;
    ctx.tiu_copy(&param);
}

void convert_fp32_bf16_kernel(const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
                          const uint32_t *depends, uint32_t depends_len, gaddr_t input_gaddr, gaddr_t output_gaddr,
                          int batch, int channel, int height, int width) {

  ctx.set_layer_id(layer_id);

  int blob_num = 2; // 2 means fp32 takes twice size of bf16
  int input_n = batch;
  int input_c = channel;
  int input_h = height;
  int input_w = width;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 2; // 2 means reserve 2 byte fp32->bf16

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;

  // lmem fmt store as bf16
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;

    // * 2 means fp32 takes twice size of bf16
    cvk_tl_shape_t input_shape = ctx.shape_t4(n, c, h, w * 2);

    // no need to eu align
    cvk_tl_t *bottom_fp32 = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);

    // load fp32 directly, * 2 means fp32 takes twice size of bf16
    ctx.tdma_load_bf16(bottom_fp32, input_gaddr + gaddr_offset * 2);

    laddr_t bottom_fp32_addr = bottom_fp32->start_address;

    // move high 16bit as bf16 format
    cvk_tl_t bottom_bf16;
    lmem_shrink_fp32_bf16(ctx, &bottom_bf16, bottom_fp32, n, c, h, w, layer_id);

    // contiguous store back
    ctx.tdma_store_bf16(&bottom_bf16, output_gaddr + gaddr_offset);

    // release
    bottom_fp32->start_address = bottom_fp32_addr;
    ctx.lmem_free_tensor(bottom_fp32);
  }
}

void fill_fp32_lmem_0(const CviBackendContext &ctx, uint32_t layer_id,
                          int batch, int channel, int height, int width) {
  int blob_num = 3; // 2 for output fp32, 1 for load bf16
  int input_n = batch;
  int input_c = channel;
  int input_h = height;
  int input_w = width;

  cvk_fmt_t fmt = CVK_FMT_BF16;

  // +2 means we prevent wrap to top, reserver it
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 2;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;

  // lmem fmt store as bf16
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  int i = 0;
  int n = tiling_info[i].first.n;
  int c = tiling_info[i].first.c;
  int h = tiling_info[i].first.h;
  int w = tiling_info[i].first.w;

  cvk_tl_t *bottom;

  // force clean all
  cvk_tl_shape_t input_shape = ctx.shape_t4(n, c, h * 2, w * 2); // fp32 takes 4 times than int8
  bottom = ctx.lmem_alloc_tensor(input_shape, CVK_FMT_I8, /*eu_align=*/0);
  cvk_tiu_xor_int8_param_t param = {0};
  param.res = bottom;
  param.a = bottom;
  param.b = bottom;
  param.layer_id = layer_id;
  ctx.tiu_xor_int8(&param);

  ctx.lmem_free_tensor(bottom);
}

void convert_bf16_fp32_kernel(
    const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int batch,
    int channel, int height, int width) {

  ctx.set_layer_id(layer_id);

  // as same reason as `convert_fp32_bf16_kernel`,
  // we keep contiguous layout once store back

  int blob_num = 3; // 3 means fp32 take twice than bf16 + bf16
  int input_n = batch;
  int input_c = channel;
  int input_h = height;
  int input_w = width;
  cvk_fmt_t fmt = CVK_FMT_BF16;

  // +2 means we prevent wrap to top, reserver it
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 2;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;

  // lmem fmt store as bf16
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  // just clear one cuz we make sure evry iterate place the same index
  fill_fp32_lmem_0(ctx, layer_id, batch, channel, height, width);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t input_shape = ctx.shape_t4(n, c, h, w);
    // 2 means fp32 takes twice size than bf16
    cvk_tl_shape_t output_shape = ctx.shape_t4(n, c, h, w * 2);
    cvk_tl_shape_t gap_shape = ctx.shape_t4(1, c, 1, coeff_lane_shape);

    // load bf16 diectly
    cvk_tl_t *top = ctx.lmem_alloc_tensor(output_shape, fmt, /*eu_align=*/0);
    cvk_tl_t *gap = ctx.lmem_alloc_tensor(gap_shape, fmt, /*eu_align=*/0);
    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);

    laddr_t top_lddr = top->start_address;
    top->start_address = top_lddr + 2; // + 2 means bf16 takes high 16bit

    ctx.tdma_load_bf16(bottom, input_gaddr + gaddr_offset);

    // fake for shape MUST eq limitation in cmodel
    top->shape = bottom->shape;
    top->stride = tl_fp32_stride(ctx, top);

    // copy bf16 to fp32 high 16bit part
    cvk_tiu_copy_param_t param = {0};
    param.src = bottom;
    param.dst = top;
    param.layer_id = layer_id;
    ctx.tiu_copy(&param);

    // store fp32 contiguous
    top->start_address = top_lddr;
    top->shape = output_shape;
    top->stride = ctx.tl_default_stride(top->shape, top->fmt, /*eu_align=*/0);
    // 2 means fp32 take twice than bf16
    ctx.tdma_store_bf16(top, output_gaddr + gaddr_offset * 2);

    // release
    ctx.lmem_free_tensor(bottom);
    ctx.lmem_free_tensor(gap);
    ctx.lmem_free_tensor(top);
  }
}
