#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include "CviBackendContext.h"

void convert_fp32_bf16_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                              gaddr_t input_gaddr, gaddr_t output_gaddr,
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
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;

    // * 2 means fp32 takes twice size of bf16
    cvk_tl_shape_t input_shape = ctx.tl_shape_t4(n, c, h, w * 2);

    // no need to eu align
    cvk_tl_t *bottom_fp32 = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);

    // load fp32 directly, * 2 means fp32 takes twice size of bf16
    ctx.tdma_load(bottom_fp32, input_gaddr + gaddr_offset * 2);

    laddr_t bottom_fp32_addr = bottom_fp32->start_address;

    // move high 16bit as bf16 format
    cvk_tl_t bottom_bf16;
    ctx.lmem_shrink_fp32_bf16(&bottom_bf16, bottom_fp32, n, c, h, w, layer_id);

    // contiguous store back
    ctx.tdma_store(&bottom_bf16, output_gaddr + gaddr_offset);

    // release
    bottom_fp32->start_address = bottom_fp32_addr;
    ctx.lmem_free_tensor(bottom_fp32);
  }
}

void convert_bf16_fp32_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                              gaddr_t input_gaddr, gaddr_t output_gaddr,
                              int batch, int channel, int height, int width) {

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
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  // just clear one cuz we make sure evry iterate place the same index
  ctx.fill_fp32_lmem_0(layer_id, batch, channel, height, width);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t input_shape = ctx.tl_shape_t4(n, c, h, w);
    // 2 means fp32 takes twice size than bf16
    cvk_tl_shape_t output_shape = ctx.tl_shape_t4(n, c, h, w * 2);
    cvk_tl_shape_t gap_shape = ctx.tl_shape_t4(1, c, 1, coeff_lane_shape);

    // load bf16 diectly
    cvk_tl_t *top = ctx.lmem_alloc_tensor(output_shape, fmt, /*eu_align=*/0);
    cvk_tl_t *gap = ctx.lmem_alloc_tensor(gap_shape, fmt, /*eu_align=*/0);
    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(input_shape, fmt, /*eu_align=*/0);

    laddr_t top_lddr = top->start_address;
    top->start_address = top_lddr + 2; // + 2 means bf16 takes high 16bit

    ctx.tdma_load(bottom, input_gaddr + gaddr_offset);

    // fake for shape MUST eq limitation in cmodel
    top->shape = bottom->shape;
    top->stride = ctx.tl_fp32_stride(top);

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
    ctx.tdma_store(top, output_gaddr + gaddr_offset * 2);

    // release
    ctx.lmem_free_tensor(bottom);
    ctx.lmem_free_tensor(gap);
    ctx.lmem_free_tensor(top);
  }
}
