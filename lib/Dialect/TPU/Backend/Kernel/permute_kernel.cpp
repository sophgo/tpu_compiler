/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: permute_kernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "cvi_backend_permute_kernel"

typedef enum {
  NCHW_N = 0,
  NCHW_C = 1,
  NCHW_H = 2,
  NCHW_W = 3,
  NCHW_MAX_DIMS
} NCHW_DIMS;


// Tensor load with global offset and global shape
static void permute_default_tensor_load(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    cvk_tg_stride_t &ifmap_gstride, int n_pos, cvk_tl_t *tl_ifmap) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = tl_ifmap->fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {
      tl_ifmap->shape.n, tl_ifmap->shape.c, tl_ifmap->shape.h,
      tl_ifmap->shape.w};
  tg_ifmap.stride = ifmap_gstride;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = tl_ifmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  permute_default_tensor_load\n"
      << "    tg offset " << ga_ifmap_offset
      << ", shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h << ")\n"
      << "    tl shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h
      << ", " << param.dst->stride.w << ")\n");

  ctx._tdma_g2l_tensor_copy(&param);
}



// Tensor load with global offset and global shape
static void permute_nhwc_to_nchw_tensor_load(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    cvk_tg_stride_t &ifmap_gstride, int n_pos, cvk_tl_t *tl_ifmap) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = tl_ifmap->fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {
      tl_ifmap->shape.n, tl_ifmap->shape.c, tl_ifmap->shape.h,
      tl_ifmap->shape.w};
  tg_ifmap.stride = ifmap_gstride;
  tl_ifmap->stride = ctx.tl_default_stride(tl_ifmap->shape, tl_ifmap->fmt, 1);

  cvk_tdma_g2l_tensor_copy_chw_rotated_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = tl_ifmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  permute_default_tensor_load\n"
      << "    tg offset " << ga_ifmap_offset
      << ", shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h << ")\n"
      << "    tl shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h
      << ", " << param.dst->stride.w << ")\n");

  ctx.tdma_g2l_tensor_copy_chw_rotated(&param);
}


//
//  Permute 0231, (N, C, H, W) -> (N, H, W, C)
//    tensor load
//    tensor move, hw transpose
//    tensor store, cw transpose
//
//  0  1  2  3
// (N, C, H, W) -> (N, H, W, C)
// (1, 2, 4, 4) -> (1, 4, 4, 2)
//
// Source (1, 2, 4, 4)
//
//             Tile 1           ||          Tile 0
//         H3           H2      ||      H1           H0
// || 16 15 14 13 | 12 11 10  9 ||  8  7  6  5 |  4  3  2  1 ||   C0
// || 32 31 30 29 | 28 27 26 25 || 24 23 22 21 | 20 19 18 17 ||   C1
//
//
// Destination (1, 4, 4, 2)
//
//  20  4 | 19  3 | 18  2 | 17  1    C0    Tile 0
//  24  8 | 23  7 | 22  6 | 21  5    C1
//  ==============================================
//  28 12 | 27 11 | 26 10 | 25  9    C2    Tile 1
//  32 16 | 31 15 | 30 14 | 29 13    C3
//
// 1. Tile 0
// 1.1. Tensor load
//    src shape (1, 2, 2, 4), stride (32, 16, 4), offset 0
//    dst shape (1, 2, 2, 4), stride (8, 8, 4)
//
//         H1            H0
//     8  7  6  5 |  4  3  2  1    C0
//    24 23 22 21 | 20 19 18 17    C1
//
// 1.2. Tensor move, HW transpose
//    src shape (1, 2, 2, 4), stride (8, 8, 4, 1)
//    dst shape (1, 2, 2, 4), stride (8, 8, 1, 2)
//
//      H3     H2      H1      H0
//     8  4 | 7  3 |  6  2 |  5  1    C0
//    24 20 |23 19 | 22 18 | 21 17    C1
//
// 1.3. Tensor store, CW transpose
//    src shape (1, 2, 4, 2), stride (8, 8, 2)
//    dst shape (1, 2, 4, 2), stride (16, 8, 2), offset 0
//
//    H3      H2      H1      H0
//  20  4 | 19  3 | 18  2 | 17  1    C0
//  24  8 | 23  7 | 22  6 | 21  5    C1
//
//
// 2. Tile 1
// 2.1. Tensor load
//    src shape (1, 2, 2, 4), stride (32, 16, 4), offset 8
//    dst shape (1, 2, 2, 4), stride (8, 8, 4)
//
//         H1            H0
//    16 15 14 13 | 12 11 10  9    C0
//    32 31 30 29 | 28 27 26 25    C1
//
// 2.2. Tensor move, HW transpose
//    src shape (1, 2, 2, 4), stride (8, 8, 4, 1)
//    dst shape (1, 2, 2, 4), stride (8, 8, 1, 2)
//
//      H3      H2      H1      H0
//    16 12 | 15 11 | 14 10 | 13  9    C0
//    32 28 | 31 27 | 30 26 | 29 25    C1
//
// 2.3. Tensor store, CW transpose
//    src shape (1, 2, 4, 2), stride (8, 8, 2)
//    dst shape (1, 2, 4, 2), stride (16, 8, 2), offset 16
//
//      H3      H2      H1      H0
//    28 12 | 27 11 | 26 10 | 25  9    C0
//    32 16 | 31 15 | 30 14 | 29 13    C1
//
//    destination in global memory
//    shape (1, 4, 4, 2), stride (32, 8, 2)
//    gm_permuted_strides[order_n 0] = dst_gm_stride.n 32
//    gm_permuted_strides[order_c 2] = dst_gm_stride.c 8
//    gm_permuted_strides[order_h 3] = dst_gm_stride.h 2
//
//    tile1 1
//    source in global memory, offset [0][0][2][0], 9
//      src_gm_offset = 2 * h_stride = 2 * 4 = 8, used in first load
//
//    destination in global memory, offset [0][2][0][0], 9
//    dst_gm_offset = 2 * c_stride = 2 * 8 = 16
//
//    src[i][j][k][l] = dst[i][k][l][j]
//    src[0][0][2][0] = dst[0][2][0][0]
//

// Only support N/H tiling.
// Two local memory needed:
//    First one for tensor load
//    Second one for tensor move
static void permute_0231_cw_tp_split(
    const CviBackendContext &ctx, std::vector<uint32_t> &shapes, cvk_fmt_t fmt,
    int eu_align, std::vector<uint32_t> &tiledSteps) {

  // Split H
  for (uint32_t tiledH = tiledSteps[NCHW_H]; tiledH != 0; --tiledH) {
    // Split N
    for (uint32_t tiledN = tiledSteps[NCHW_N]; tiledN != 0; --tiledN) {
      cvk_tl_shape_t tl_load_shape = {
          tiledN, tiledSteps[NCHW_C], tiledH, tiledSteps[NCHW_W]};
      uint32_t loadSizePerLane =
          ctx.lmem_tensor_to_size(tl_load_shape, fmt, eu_align);

      cvk_tl_shape_t tl_move_shape = {
          tiledN, tiledSteps[NCHW_C], tiledSteps[NCHW_W], tiledH};
      uint32_t moveSizePerLane =
          ctx.lmem_tensor_to_size(tl_move_shape, fmt, eu_align);

      uint32_t totalNeededPerLane = loadSizePerLane + moveSizePerLane;
      if (totalNeededPerLane <= (uint32_t)LOCAL_MEM_SIZE) {
        tiledSteps[NCHW_N] = tiledN;
        tiledSteps[NCHW_H] = tiledH;

        LLVM_DEBUG(llvm::dbgs()
            << "    permute_0231_cw_tp_split:\n"
            << "      shape(" << shapes[0]
            << ", " << shapes[1]
            << ", " << shapes[2]
            << ", " << shapes[3]
            << ")\n"
            << "      tiledSteps(" << tiledSteps[0]
            << ", " << tiledSteps[1]
            << ", " << tiledSteps[2]
            << ", " << tiledSteps[3]
            << ")\n"
            << "     totalNeededPerLane " << totalNeededPerLane
            << ", totalSizePerLane " << LOCAL_MEM_SIZE
            << "\n");
        return;
      }
    }
  }

  assert(0 && "Expect valid split for 0231 permute");

  tiledSteps[NCHW_N] = 0;
  tiledSteps[NCHW_H] = 0;
}

static void permute_0231_cw_tp(
    const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint32_t order_n, uint32_t order_c, uint32_t order_h, uint32_t order_w,
    cvk_fmt_t fmt) {

  // TDMA store w/ CW-TP only supports int8
  assert((fmt == CVK_FMT_I8) && "TDMA store w/ cw-tp only support int8");

  uint32_t eu_align = 0; // contiguous memory

  std::vector<uint32_t> srcShapes = {
      input_n, input_c, input_h, input_w};
  std::vector<uint32_t> tiledSteps =
      {input_n, input_c, input_h, input_w};
  permute_0231_cw_tp_split(ctx, srcShapes, fmt, eu_align, tiledSteps);
  if (!tiledSteps[NCHW_N] || !tiledSteps[NCHW_H])
    return;

  // Global stride from global shape
  std::vector<uint32_t> srcStrides(NCHW_MAX_DIMS);
  srcStrides[NCHW_W] = (fmt == CVK_FMT_I8) ? 1 : 2;
  srcStrides[NCHW_H] = srcShapes[NCHW_W] * srcStrides[NCHW_W];
  srcStrides[NCHW_C] = srcShapes[NCHW_H] * srcStrides[NCHW_H];
  srcStrides[NCHW_N] = srcShapes[NCHW_C] * srcStrides[NCHW_C];

  std::vector<uint32_t> orders= {
      order_n, order_c, order_h, order_w};
  std::vector<uint32_t> dstShapes = {
      srcShapes[orders[NCHW_N]], srcShapes[orders[NCHW_C]],
      srcShapes[orders[NCHW_H]], srcShapes[orders[NCHW_W]]};
  std::vector<uint32_t> dstStrides(NCHW_MAX_DIMS);
  dstStrides[NCHW_W] = (fmt == CVK_FMT_I8) ? 1 : 2;
  dstStrides[NCHW_H] = dstShapes[NCHW_W] * dstStrides[NCHW_W];
  dstStrides[NCHW_C] = dstShapes[NCHW_H] * dstStrides[NCHW_H];
  dstStrides[NCHW_N] = dstShapes[NCHW_C] * dstStrides[NCHW_C];

  // Derive destination offset from source position
  std::vector<uint32_t> dstIndex(NCHW_MAX_DIMS);
  dstIndex[orders[NCHW_N]] = 0;
  dstIndex[orders[NCHW_C]] = 1;
  dstIndex[orders[NCHW_H]] = 2;
  dstIndex[orders[NCHW_W]] = 3;

  //
  // Main tiled transpose routine
  //
  std::vector<uint32_t> srcPoss = {0, 0, 0, 0};
  for (srcPoss[NCHW_H] = 0; srcPoss[NCHW_H] < srcShapes[NCHW_H];
       srcPoss[NCHW_H] += tiledSteps[NCHW_H]) {
    uint32_t curH =
        std::min(srcShapes[NCHW_H] - srcPoss[NCHW_H], tiledSteps[NCHW_H]);
    for (srcPoss[NCHW_N] = 0; srcPoss[NCHW_N] < srcShapes[NCHW_N];
         srcPoss[NCHW_N] += tiledSteps[NCHW_N]) {
      uint32_t curN =
        std::min(srcShapes[NCHW_N] - srcPoss[NCHW_N], tiledSteps[NCHW_N]);

      std::vector<uint32_t> srcTiledShapes = {
          curN, srcShapes[NCHW_C], curH, srcShapes[NCHW_W]};

      uint32_t srcOffset =
          srcPoss[NCHW_N] * srcStrides[NCHW_N] +
          srcPoss[NCHW_C] * srcStrides[NCHW_C] +
          srcPoss[NCHW_H] * srcStrides[NCHW_H] +
          srcPoss[NCHW_W] * srcStrides[NCHW_W];
      uint32_t dstOffset = srcPoss[NCHW_N] * dstStrides[dstIndex[NCHW_N]] +
                           srcPoss[NCHW_C] * dstStrides[dstIndex[NCHW_C]] +
                           srcPoss[NCHW_H] * dstStrides[dstIndex[NCHW_H]] +
                           srcPoss[NCHW_W] * dstStrides[dstIndex[NCHW_W]];

      LLVM_DEBUG(llvm::dbgs()
          << "    srcPoss[" << srcPoss[0]
          << "][" << srcPoss[1]
          << "][" << srcPoss[2]
          << "][" << srcPoss[3]
          << "] srcTiledShapes["
          << srcTiledShapes[0]
          << "][" << srcTiledShapes[1]
          << "][" << srcTiledShapes[2]
          << "][" << srcTiledShapes[3]
          << "] srcOffset " << srcOffset
          << ", dstOffset " << dstOffset
          << "\n");

      // 1. Tensor load, tiled shape, global stride
      cvk_tl_t *tl_load_dst_tiled = NULL;
      {
        cvk_tg_t tg_src_tiled;
        memset(&tg_src_tiled, 0, sizeof(tg_src_tiled));
        tg_src_tiled.base_reg_index =
            ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
        tg_src_tiled.start_address = ga_ifmap + srcOffset;
        tg_src_tiled.fmt = fmt;
        tg_src_tiled.shape.n = srcTiledShapes[NCHW_N];
        tg_src_tiled.shape.c = srcTiledShapes[NCHW_C];
        tg_src_tiled.shape.h = srcTiledShapes[NCHW_H];
        tg_src_tiled.shape.w = srcTiledShapes[NCHW_W];
        tg_src_tiled.stride.n = srcStrides[NCHW_N];
        tg_src_tiled.stride.c = srcStrides[NCHW_C];
        tg_src_tiled.stride.h = srcStrides[NCHW_H];

        cvk_tl_shape_t tl_dst_tiled_shape = {
            srcTiledShapes[NCHW_N], srcTiledShapes[NCHW_C],
            srcTiledShapes[NCHW_H], srcTiledShapes[NCHW_W]};
        tl_load_dst_tiled =
            ctx.lmem_alloc_tensor(tl_dst_tiled_shape, fmt, eu_align);

        cvk_tdma_g2l_tensor_copy_param_t param;
        param.src = &tg_src_tiled;
        param.dst = tl_load_dst_tiled;
        ctx.tdma_g2l_tensor_copy(&param);
      }

      // 2. Tensor move, HW transpose
      cvk_tl_t *tl_move_dst = NULL;
      {
        cvk_tl_shape_t tl_move_dst_shape = {
            srcTiledShapes[NCHW_N], srcTiledShapes[NCHW_C],
            srcTiledShapes[NCHW_W], srcTiledShapes[NCHW_H]};
        tl_move_dst =
            ctx.lmem_alloc_tensor(tl_move_dst_shape, fmt, eu_align);

        // HW transpose, still use source shape for data transfer
        cvk_tl_t tl_dst_hw_tp;
        ctx.lmem_init_tensor(&tl_dst_hw_tp, tl_load_dst_tiled->shape, fmt,
                             eu_align);
        tl_dst_hw_tp.start_address = tl_move_dst->start_address;
        tl_dst_hw_tp.stride.h = tl_move_dst->stride.w;
        tl_dst_hw_tp.stride.w = tl_move_dst->stride.h;

        cvk_tiu_copy_param_t param;
        param.src = tl_load_dst_tiled;
        param.dst = &tl_dst_hw_tp;
        param.layer_id = layer_id;
        ctx.tiu_copy(&param);
      }

      // 3. Tensor store, CW transpose
      {
        cvk_tg_t tg_dst_tiled;
        memset(&tg_dst_tiled, 0, sizeof(tg_dst_tiled));
        tg_dst_tiled.base_reg_index =
            ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
        tg_dst_tiled.start_address = ga_ofmap + dstOffset;
        tg_dst_tiled.fmt = fmt;
        tg_dst_tiled.shape.n = tl_move_dst->shape.n;
        tg_dst_tiled.shape.c = tl_move_dst->shape.w; // CW transpose
        tg_dst_tiled.shape.h = tl_move_dst->shape.h;
        tg_dst_tiled.shape.w = tl_move_dst->shape.c; // CW transpose
        tg_dst_tiled.stride =
            ctx.tg_default_stride(tg_dst_tiled.shape, fmt);

        cvk_tdma_l2g_tensor_copy_cw_transposed_param_t param;
        param.src = tl_move_dst;
        param.dst = &tg_dst_tiled;
        ctx.tdma_l2g_tensor_copy_cw_transposed(&param);
      }

      // Free local memory
      ctx.lmem_free_tensor(tl_move_dst);
      ctx.lmem_free_tensor(tl_load_dst_tiled);
    }
  }

}

static void permute_xxx3_tp_split(
    const CviBackendContext &ctx, uint32_t input_n, uint32_t input_c,
    uint32_t input_h, uint32_t input_w, cvk_fmt_t fmt, int eu_align,
    uint32_t &n_step) {

  // Shape (N, C, H, W)
  n_step = input_n;
  for (; n_step > 0; --n_step) {
    cvk_tl_shape_t tiled_ifmap_shape = {n_step, input_c, input_h, input_w};
    uint32_t tiled_ifmap_size =
        ctx.lmem_tensor_to_size(tiled_ifmap_shape, fmt, eu_align);

    if (tiled_ifmap_size <= static_cast<uint32_t>(LOCAL_MEM_SIZE))
      break;
  }

  LLVM_DEBUG(llvm::dbgs()
      << "  permute_xxx3_tp_split: n_step " << n_step << "\n");

  assert(n_step && "Expect valid tile");
}

static void permute_xxx3_tp_assign_lmem_layout(
    const CviBackendContext &ctx, uint32_t cur_n, uint32_t input_c,
    uint32_t input_h, uint32_t input_w, cvk_fmt_t fmt, int eu_align,
    cvk_tl_t &tl_ifmap, cvk_tl_t &tl_ofmap) {

  // Shape (N, C, H, W) for both ifmap and ofmap
  ctx.lmem_init_tensor(&tl_ifmap, {cur_n, input_c, input_h, input_w}, fmt,
                       eu_align);
  ctx.lmem_init_tensor(&tl_ofmap, {cur_n, input_c, input_h, input_w}, fmt,
                       eu_align);
}

static void permute_xxx3_tp_tensor_load(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    cvk_tg_stride_t &ifmap_gstride, int n_pos, cvk_tl_t *tl_ifmap) {

  permute_default_tensor_load(ctx, layer_id, ga_ifmap, ifmap_gstride, n_pos,
                              tl_ifmap);
}

static void permute_0312_tp_tensor_load(const CviBackendContext &ctx,
                                        uint32_t layer_id, gaddr_t ga_ifmap,
                                        cvk_tg_stride_t &ifmap_gstride,
                                        int n_pos, cvk_tl_t *tl_ifmap) {

  permute_nhwc_to_nchw_tensor_load(ctx, layer_id, ga_ifmap, ifmap_gstride, n_pos,
                              tl_ifmap);
}

static void permute_tp_tensor_store(const CviBackendContext &ctx,
                                        uint32_t layer_id, gaddr_t ga_ofmap,
                                        cvk_tl_t *tl_ofmap, uint32_t order_n,
                                        uint32_t order_c, uint32_t order_h,
                                        cvk_tg_stride_t &ofmap_gstride,
                                        int n_pos) {

  uint64_t ga_ofmap_offset = ofmap_gstride.n * n_pos;

  cvk_tg_t tg_ofmap = {0};
  tg_ofmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  tg_ofmap.start_address = ga_ofmap + ga_ofmap_offset;
  tg_ofmap.fmt = tl_ofmap->fmt;
  tg_ofmap.shape = {tl_ofmap->shape.n, tl_ofmap->shape.c, tl_ofmap->shape.h,
                    tl_ofmap->shape.w};
  tg_ofmap.stride = {ofmap_gstride.n, ofmap_gstride.c, ofmap_gstride.h};

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = tl_ofmap;
  param.dst = &tg_ofmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs() << "  permute_xxx3_tp_tensor_store\n"
                          << "    tl shape(" << param.src->shape.n << ", "
                          << param.src->shape.c << ", " << param.src->shape.h
                          << ", " << param.src->shape.w << "), stride("
                          << param.src->stride.n << ", " << param.src->stride.c
                          << ", " << param.src->stride.h << ", "
                          << param.src->stride.w << ")\n"
                          << "    tg offset " << ga_ofmap_offset << ", shape("
                          << param.dst->shape.n << ", " << param.dst->shape.c
                          << ", " << param.dst->shape.h << ", "
                          << param.dst->shape.w << "), stride("
                          << param.dst->stride.n << ", " << param.dst->stride.c
                          << ", " << param.dst->stride.h << "\n");

  ctx._tdma_l2g_tensor_copy(&param);
}

// ORDER: [0-2][0-2][0-2]3
// (N, C, H, W) -> (, , , W)
//
// Order: 1023
//        0  1  2  3      1  0  2  3
// Shape (4, 2, 2, 2) -> (2, 4, 2, 2) (ns=16, cs=4, hs=4)
//
//  src shape  (4, 2, 2, 2)           dst shape  (4, 2, 2, 2)
//      stride (ns=8, cs=4, hs=2)         stride (ns=4, cs=16, hs=2)
//
//    H1     H0               ->      H1      H0
//  4  3 |  2  1 | c0, n0            4  3 |  2  1 | c0, n0
//  8  7 |  6  5 | c1, n0           12 11 | 10  9 | c1, n0
// 12 11 | 10  9 | c0, n1           20 19 | 18 17 | c2, n0
// 16 15 | 14 13 | c1, n1           28 27 | 26 25 | c3, n0
// -----------------------         -----------------------
// 20 19 | 18 17 | c0, n2            8  7 |  6  5 | c0, n1
// 24 23 | 22 21 | c1, n2           16 15 | 14 13 | c1, n1
// 28 27 | 26 25 | c0, n3           24 23 | 22 21 | c2, n1
// 32 31 | 30 29 | c1, n2           32 31 | 30 29 | c3, n1
//
//
// Tile 1
//   src ga_offset = 2(cur_n) * 8(src ns) = 16
//   dst ga_offset = 2(cur_n) * 4(dst cs) = 8
//
//    H1     H0               ->      H1      H0
//  X  X |  X  X | c0, n0            X  X |  X  X | c0, n0
//  X  X |  X  X | c1, n0           XX XX | XX  X | c1, n0
// XX XX | XX  X | c0, n1           20 19 | 18 17 | c2, n0, offset = 8
// XX XX | XX XX | c1, n1           28 27 | 26 25 | c3, n0
// -----------------------         -----------------------
// 20 19 | 18 17 | c0, n2            X  X |  X  X | c0, n1
// 24 23 | 22 21 | c1, n2           XX XX | XX XX | c1, n1
// 28 27 | 26 25 | c0, n3           24 23 | 22 21 | c2, n1
// 32 31 | 30 29 | c1, n2           32 31 | 30 29 | c3, n1
//
//
//
// Order: 0213
//        0  1  2  3      0  2  1  3
// Shape (2, 4, 3, 2) -> (2, 3, 4, 2) (ns=24, cs=8, hs=2)
//
// src shape  (2, 4, 3, 2)            dst shape  (2, 4, 3, 2)
//     stride (ns=24, cs=6, hs=2)         stride (ns=24, cs=2, hs=8)
//
//
//   H2      H1      H0               H3      H2      H1      H0
//  6  5 |  4  3 |  2  1 | c0       20 19 | 14 13 |  8  7 |  2  1 | c0
// 12 11 | 10  9 |  8  7 | c1       22 21 | 16 15 | 10  9 |  4  3 | c1
// 18 17 | 16 15 | 14 14 | c2       24 23 | 18 17 | 12 11 |  6  5 | c2
// 24 23 | 22 21 | 20 19 | c3       ----------------------------------
// --------------------------       44 43 | 38 37 | 32 31 | 26 25 | c0
// 30 29 | 28 27 | 26 25 | c0       46 45 | 40 39 | 34 33 | 28 27 | c1
// 36 35 | 34 33 | 32 31 | c1       48 47 | 42 41 | 36 35 | 30 29 | c2
// 42 41 | 40 39 | 38 37 | c2
// 48 47 | 46 45 | 44 43 | c3
//
//
// Tile1
//   src ga_offset = 1(cur_n) * 24(src ns) = 24
//   dst ga_offset = 1(cur_n) * 24(dst ns) = 24
//
//   H2      H1      H0               H3      H2      H1      H0
//  X  X |  X  X |  X  X | c0       XX XX | XX XX |  X  X |  X  X | c0
// XX XX | XX  X |  X  X | c1       XX XX | XX XX | XX  X |  X  X | c1
// XX XX | XX XX | XX XX | c2       XX XX | XX XX | XX XX |  X  X | c2
// XX XX | XX XX | XX XX | c3       ----------------------------------
// --------------------------       44 43 | 38 37 | 32 31 | 26 25 | c0, offset
// 30 29 | 28 27 | 26 25 | c0       46 45 | 40 39 | 34 33 | 28 27 | c1    = 24
// 36 35 | 34 33 | 32 31 | c1       48 47 | 42 41 | 36 35 | 30 29 | c2
// 42 41 | 40 39 | 38 37 | c2
// 48 47 | 46 45 | 44 43 | c3
//
static void permute_xxx3_tp_tensor_store(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ofmap,
    cvk_tl_t *tl_ofmap, uint32_t order_n, uint32_t order_c, uint32_t order_h,
    cvk_tg_stride_t &ofmap_gstride, int n_pos) {

  // It is tricky to use source shape as the destination shape.
  // And the destination strides are permuted.
  uint32_t tg_dst_permuted_strides[3];
  tg_dst_permuted_strides[order_n] = ofmap_gstride.n;
  tg_dst_permuted_strides[order_c] = ofmap_gstride.c;
  tg_dst_permuted_strides[order_h] = ofmap_gstride.h;

  uint64_t ga_ofmap_offset = tg_dst_permuted_strides[0] * n_pos;

  cvk_tg_t tg_ofmap = {0};
  tg_ofmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  tg_ofmap.start_address = ga_ofmap + ga_ofmap_offset;
  tg_ofmap.fmt = tl_ofmap->fmt;
  tg_ofmap.shape = {
      tl_ofmap->shape.n, tl_ofmap->shape.c, tl_ofmap->shape.h,
      tl_ofmap->shape.w};
  tg_ofmap.stride = {
      tg_dst_permuted_strides[0], tg_dst_permuted_strides[1],
      tg_dst_permuted_strides[2]};

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = tl_ofmap;
  param.dst = &tg_ofmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  permute_xxx3_tp_tensor_store\n"
      << "    tl shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << ", " << param.src->shape.w
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h
      << ", " << param.src->stride.w << ")\n"
      << "    tg offset " << ga_ofmap_offset
      << ", shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h << "\n");

  ctx._tdma_l2g_tensor_copy(&param);
}

// (0, 1, 2, 3) -> (, , , 3)
//   TDMA does not has the stride of width(ws).
//   Since the width of destination is unchanged, use tensor store to write one
//   hight to the correct position with ns, cs, hs.
//   It is tricky that destination shape in tensor store is the same as source
//   shape.
static void permute_xxx3_tp(
    const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint32_t order_n, uint32_t order_c, uint32_t order_h, uint32_t order_w,
    cvk_fmt_t fmt) {

  assert(order_n < 3 && "Expect order_n < 3");
  assert(order_c < 3 && "Expect order_c < 3");
  assert(order_h < 3 && "Expect order_h < 3");
  if ((order_n >= 3) || (order_c >= 3) || (order_h >= 3))
    return;

  int eu_align = 0; // no need to align eu
  uint32_t n_step = 0;
  permute_xxx3_tp_split(ctx, input_n, input_c, input_h, input_w, fmt,
                        eu_align, n_step);
  if (!n_step)
    return;

  // Global stride from global shape
  cvk_tg_stride_t ifmap_gstride = {
      input_c * input_h * input_w, input_h * input_w, input_w};
  cvk_tg_stride_t ofmap_gstride = {
      output_c * output_h * output_w, output_h * output_w, output_w};

  for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step) {
    uint32_t cur_n = std::min(input_n - n_pos, n_step);

    // 1. Assign local memory layout
    cvk_tl_t tl_ifmap, tl_ofmap;
    permute_xxx3_tp_assign_lmem_layout(ctx, cur_n, input_c, input_h, input_w,
                                       fmt, eu_align, tl_ifmap, tl_ofmap);

    // 2. tensor load, (N, C, H, W) -> (N, C, H, W)
    permute_xxx3_tp_tensor_load(ctx, layer_id, ga_ifmap, ifmap_gstride,
                                n_pos, &tl_ifmap);

    // 3. tensor store, (N, C, H, W) -> (, , , W)
    permute_xxx3_tp_tensor_store(ctx, layer_id, ga_ofmap, &tl_ofmap, order_n,
                                 order_c, order_h, ofmap_gstride, n_pos);
  } // for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step)
}

// for 0312 case that input channel == 1
static void permute_0312_tp_c_1(const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap,
    cvk_tg_t src_tg, cvk_tg_t dst_tg,
    cvk_tg_stride_t global_input_stride, cvk_tg_stride_t global_output_stride,
    int& h_shift, int& w_shift, int& sec_len_h) {

  cvk_fmt_t fmt = CVK_FMT_I8;
  assert((fmt == CVK_FMT_I8) && "cw_transposed not support bf16");
  int data_type_size = 1; // no bf16 case
  int input_h = src_tg.shape.h;

  src_tg.start_address = ga_ifmap + h_shift;
  src_tg.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src_tg.start_address);
  src_tg.shape.c = sec_len_h;
  src_tg.shape.h = 1;
  src_tg.stride = ctx.tg_default_stride(src_tg.shape, fmt);
  src_tg.stride.n = global_input_stride.n;
  h_shift += src_tg.shape.w * sec_len_h * data_type_size;

  cvk_tl_shape_t tl_src_shape;
  tl_src_shape.n = src_tg.shape.n;
  tl_src_shape.c = sec_len_h;
  tl_src_shape.h = 1;
  tl_src_shape.w = src_tg.shape.w;
  cvk_tl_t *tmp_tl_load =
    ctx.lmem_alloc_tensor(tl_src_shape, fmt, /*eu_align*/ 0);

  // 1. load
  cvk_tdma_g2l_tensor_copy_param_t p1;
  p1.src = &src_tg;
  p1.dst = tmp_tl_load;
  ctx.tdma_g2l_bf16_tensor_copy(&p1);

  cvk_tdma_l2g_tensor_copy_cw_transposed_param_t p2;

  dst_tg.start_address = ga_ofmap + w_shift;
  dst_tg.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst_tg.start_address);
  dst_tg.shape.w = sec_len_h;
  dst_tg.shape.h = 1;
  dst_tg.stride = ctx.tg_default_stride(dst_tg.shape, fmt);
  dst_tg.stride.c = input_h * data_type_size;
  dst_tg.stride.n = global_output_stride.n;
  w_shift += sec_len_h * data_type_size;

  p2.src = tmp_tl_load;
  p2.dst = &dst_tg;

  // store back
  ctx.tdma_l2g_bf16_tensor_copy_cw_transposed(&p2);

  // free
  ctx.lmem_free_tensor(tmp_tl_load);
}

static void permute_0312_tp( const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint32_t order_n, uint32_t order_c, uint32_t order_h, uint32_t order_w,
    cvk_fmt_t fmt){

  int eu_align = 0; // no need to align eu in tg
  int nsecs, hsecs;

  // slice align target cuz `cvk_tdma_g2l_tensor_copy_chw_rotated_param_t`
  // set by target shape
  _split_nh(ctx, 1, output_c, input_n * output_h, output_w, /*blob_num=*/1, /*reserved=*/0, &nsecs, &hsecs);

  // Global stride from global shape
  cvk_tg_stride_t ofmap_gstride = {output_c * output_h * output_w,
    output_h * output_w, output_w};

  int hslice = output_h / hsecs;
  int hresidual = output_h - hslice * hsecs;

  if (input_c == 1) {
    _split_nh(ctx, 1, input_c, input_n * input_h, input_w, /*blob_num=*/1, /*reserved=*/0, &nsecs, &hsecs);

    hslice = input_h / hsecs;
    hresidual = input_h - hslice * hsecs;
  }

  // init structure
  cvk_tg_t src_tg;
  cvk_tg_t dst_tg;
  cvk_tg_shape_t tg_src_shape = {input_n, input_c, input_h, input_w};
  src_tg.shape = tg_src_shape;
  src_tg.stride = ctx.tg_default_stride(src_tg.shape, fmt);
  src_tg.fmt = fmt;

  cvk_tg_shape_t tg_dst_shape = {input_n, output_c, output_h, output_w};
  dst_tg.shape = tg_dst_shape;
  dst_tg.stride = ctx.tg_default_stride(dst_tg.shape, fmt);
  dst_tg.fmt = fmt;

  int h_shift = 0;
  int w_shift = 0;

  // nhwc->nchw, leverage cvk_tdma_g2l_tensor_copy_chw_rotated_param_t
  int src_hw_offset = 0;
  int output_offset = 0;
  uint32_t n_pos = 0;
  for (int hidx = 0; hidx < hsecs; hidx++) {
    int sec_len_h = hslice + (hidx < hresidual);

    if (input_c == 1) {
      // gaitset case
      // 1. reshape n, 1, h, w to n,h,1,w
      // 2. load to lmem
      // 3. store back via cw_transposed, shape is n,w,1,h
      permute_0312_tp_c_1(ctx,
          layer_id, ga_ifmap, ga_ofmap,
          src_tg, dst_tg,
          src_tg.stride, dst_tg.stride,
          h_shift, w_shift, sec_len_h);
    } else {
      // tenserfow case, hwc to chw
      cvk_tg_stride_t ifmap_gstride = {input_c * input_h * input_w,
        sec_len_h * output_w, output_w};

      // 1. Assign local memory layout
      cvk_tl_t tl_ifmap, tl_ofmap;
      permute_xxx3_tp_assign_lmem_layout(ctx, input_n, output_c, sec_len_h, output_w,
          fmt, eu_align, tl_ifmap, tl_ofmap);

      // 2. tensor load, (N, H, W, C) -> (N, C, H, W)
      permute_0312_tp_tensor_load(ctx, layer_id, ga_ifmap + src_hw_offset, ifmap_gstride, n_pos,
          &tl_ifmap);

      // 3. tensor store, (N, C, H, W) -> (N, C, H, W)
      permute_tp_tensor_store(ctx, layer_id, ga_ofmap + output_offset, &tl_ifmap, order_n,
          order_c, order_h, ofmap_gstride, n_pos);

      /**
       * src: <1,2,4,3>             -> dst:<1, 3, 2, 4>
       * array([[[[ 0,  1,  2],     array([[[[ 0,  3,  6,  9],
         [ 3,  4,  5],              [12, 15, 18, 21]],
         [ 6,  7,  8],
         [ 9, 10, 11]],             [[ 1,  4,  7, 10],
                                    [13, 16, 19, 22]],
        [[12, 13, 14],
         [15, 16, 17],              [[ 2,  5,  8, 11],
         [18, 19, 20],              [14, 17, 20, 23]]]]
         [21, 22, 23]]]]
         we should tile from target shape cuz
         `cvk_tdma_g2l_tensor_copy_chw_rotated_param_t` requirment,
         e.g: sec_len_h = 1, hidx = 1, the target start with 12
         (ga_ofmap + 1[sec_len_h] * 4[output_w])
         and we should re-map source input shift to 12
         (ga_ifmap + 1[sec_len_h] * input_h * input_w)
       */
      src_hw_offset += sec_len_h * input_h * input_w;
      output_offset += sec_len_h * output_w; // h * w
    }
  }
}


//
// Permute 0321, (N, C, H, W) -> (N, H, W, C)
//   tensor load
//   tensor store, cw transpose
//
//  0  1  2  3
// (N, C, H, W) -> (N, W, H, C)
// (1, 4, 2, 2) -> (1, 2, 2, 4)
//
//
// Source (1, 4, 2, 2)
//
// Tile 1   Tile 0
//   H1  ||   H0
//  3  2 ||  1  0     C0
//  7  6 ||  5  4     C1
// 11 10 ||  9  8     C2
// 15 14 || 13 12     C3
//
//
// Destination (1, 2, 2, 4)
//
//   Tile 1          Tile 0
//      H1     ||      H0
// 14 10  6  2 || 12  8  4  0   C0
// 15 11  7  3 || 13  9  5  1   C1
//
// 1. Tile 0
// 1.1. Tensor load
//     src shape (1, 4, 1, 2), stride (16, 4, 2), offset 0
//     dst shape (1, 4, 1, 2), stride (2, 2, 2)
//
//   H0
//  1  0     C0
//  5  4     C1
//  9  8     C2
// 13 12     C3
//
// 1.2. Tensor store, CW transpose
//     src shape (1, 4, 1, 2), stride (2, 2, 2)
//     dst shape (1, 2, 1, 4), stride (8, 2, 4), offset 0
//
//      H0
// 12  8  4  0   C0
// 13  9  5  1   C1
//
//
// 2. Tile 1
// 2.1. Tensor load
//    src shape (1, 4, 1, 2), stride (16, 4, 2), offset 2
//    dst shape (1, 4, 1, 2), stride (2, 1, 2)
//
//   H0
//  3  2     C0
//  7  6     C1
// 11 10     C2
// 15 14     C3
//
// 2.2. Tensor store, CW transpose
//     src shape (1, 4, 1, 2), stride (1, 2, 1, 2)
//     dst shape (1, 2, 1, 4), stride (8, 2, 4), offset 4
//
//       H1
//  14 10  6  2    C0
//  15 11  7  3    C1
//

// Only support N/H tiling:
//   one local memory needed
static void permute_0321_cw_tp_split(
    const CviBackendContext &ctx, std::vector<uint32_t> &shapes, cvk_fmt_t fmt,
    int eu_align, std::vector<uint32_t> &tiledSteps) {

  // Split H
  for (uint32_t tiledH = tiledSteps[NCHW_H]; tiledH != 0; --tiledH) {
    // Split N
    for (uint32_t tiledN = tiledSteps[NCHW_N]; tiledN != 0; --tiledN) {
      cvk_tl_shape_t tl_load_shape = {
          tiledN, tiledSteps[NCHW_C], tiledH, tiledSteps[NCHW_W]};
      uint32_t loadSizePerLane =
          ctx.lmem_tensor_to_size(tl_load_shape, fmt, eu_align);

      if (loadSizePerLane <= (uint32_t)LOCAL_MEM_SIZE) {
        tiledSteps[NCHW_N] = tiledN;
        tiledSteps[NCHW_H] = tiledH;

        LLVM_DEBUG(llvm::dbgs()
            << "    permute_0231_cw_tp_split:\n"
            << "      shape(" << shapes[0]
            << ", " << shapes[1]
            << ", " << shapes[2]
            << ", " << shapes[3]
            << ")\n"
            << "      tiledSteps(" << tiledSteps[0]
            << ", " << tiledSteps[1]
            << ", " << tiledSteps[2]
            << ", " << tiledSteps[3]
            << ")\n"
            << "     loadSizePerLane " << loadSizePerLane
            << ", totalSizePerLane " << LOCAL_MEM_SIZE
            << "\n");
        return;
      }
    }
  }

  assert(0 && "Expect valid split for 0321 permute");

  tiledSteps[NCHW_N] = 0;
  tiledSteps[NCHW_H] = 0;
}

static void permute_0321_tp(
    const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint32_t order_n, uint32_t order_c, uint32_t order_h, uint32_t order_w,
    cvk_fmt_t fmt) {

  // TDMA store w/ CW-TP only supports int8
  assert((fmt == CVK_FMT_I8) && "TDMA store w/ cw-tp only support int8");

  uint32_t eu_align = 0; // contiguous memory

  std::vector<uint32_t> srcShapes = {
      input_n, input_c, input_h, input_w};
  std::vector<uint32_t> tiledSteps =
      {input_n, input_c, input_h, input_w};
  permute_0321_cw_tp_split(ctx, srcShapes, fmt, eu_align, tiledSteps);
  if (!tiledSteps[NCHW_N] || !tiledSteps[NCHW_H])
    return;

  // Global stride from global shape
  std::vector<uint32_t> srcStrides(NCHW_MAX_DIMS);
  srcStrides[NCHW_W] = (fmt == CVK_FMT_I8) ? 1 : 2;
  srcStrides[NCHW_H] = srcShapes[NCHW_W] * srcStrides[NCHW_W];
  srcStrides[NCHW_C] = srcShapes[NCHW_H] * srcStrides[NCHW_H];
  srcStrides[NCHW_N] = srcShapes[NCHW_C] * srcStrides[NCHW_C];

  std::vector<uint32_t> orders= {
      order_n, order_c, order_h, order_w};
  std::vector<uint32_t> dstShapes = {
      srcShapes[orders[NCHW_N]], srcShapes[orders[NCHW_C]],
      srcShapes[orders[NCHW_H]], srcShapes[orders[NCHW_W]]};
  std::vector<uint32_t> dstStrides(NCHW_MAX_DIMS);
  dstStrides[NCHW_W] = (fmt == CVK_FMT_I8) ? 1 : 2;
  dstStrides[NCHW_H] = dstShapes[NCHW_W] * dstStrides[NCHW_W];
  dstStrides[NCHW_C] = dstShapes[NCHW_H] * dstStrides[NCHW_H];
  dstStrides[NCHW_N] = dstShapes[NCHW_C] * dstStrides[NCHW_C];

  // Derive destination offset from source position
  std::vector<uint32_t> dstIndex(NCHW_MAX_DIMS);
  dstIndex[orders[NCHW_N]] = 0;
  dstIndex[orders[NCHW_C]] = 1;
  dstIndex[orders[NCHW_H]] = 2;
  dstIndex[orders[NCHW_W]] = 3;

  //
  // Main tiled transpose routine
  //
  std::vector<uint32_t> srcPoss = {0, 0, 0, 0};
  for (srcPoss[NCHW_H] = 0; srcPoss[NCHW_H] < srcShapes[NCHW_H];
       srcPoss[NCHW_H] += tiledSteps[NCHW_H]) {
    uint32_t curH =
        std::min(srcShapes[NCHW_H] - srcPoss[NCHW_H], tiledSteps[NCHW_H]);
    for (srcPoss[NCHW_N] = 0; srcPoss[NCHW_N] < srcShapes[NCHW_N];
         srcPoss[NCHW_N] += tiledSteps[NCHW_N]) {
      uint32_t curN =
        std::min(srcShapes[NCHW_N] - srcPoss[NCHW_N], tiledSteps[NCHW_N]);

      std::vector<uint32_t> srcTiledShapes = {
          curN, srcShapes[NCHW_C], curH, srcShapes[NCHW_W]};

      uint32_t srcOffset =
          srcPoss[NCHW_N] * srcStrides[NCHW_N] +
          srcPoss[NCHW_C] * srcStrides[NCHW_C] +
          srcPoss[NCHW_H] * srcStrides[NCHW_H] +
          srcPoss[NCHW_W] * srcStrides[NCHW_W];
      uint32_t dstOffset = srcPoss[NCHW_N] * dstStrides[dstIndex[NCHW_N]] +
                           srcPoss[NCHW_C] * dstStrides[dstIndex[NCHW_C]] +
                           srcPoss[NCHW_H] * dstStrides[dstIndex[NCHW_H]] +
                           srcPoss[NCHW_W] * dstStrides[dstIndex[NCHW_W]];

      LLVM_DEBUG(llvm::dbgs()
          << "    srcPoss[" << srcPoss[0]
          << "][" << srcPoss[1]
          << "][" << srcPoss[2]
          << "][" << srcPoss[3]
          << "] srcTiledShapes["
          << srcTiledShapes[0]
          << "][" << srcTiledShapes[1]
          << "][" << srcTiledShapes[2]
          << "][" << srcTiledShapes[3]
          << "] srcOffset " << srcOffset
          << ", dstOffset " << dstOffset
          << "\n");

      // 1. Tensor load, tiled shape, global stride
      cvk_tl_t *tl_load_dst_tiled = NULL;
      {
        cvk_tg_t tg_src_tiled;
        memset(&tg_src_tiled, 0, sizeof(tg_src_tiled));
        tg_src_tiled.base_reg_index =
            ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
        tg_src_tiled.start_address = ga_ifmap + srcOffset;
        tg_src_tiled.fmt = fmt;
        tg_src_tiled.shape.n = srcTiledShapes[NCHW_N];
        tg_src_tiled.shape.c = srcTiledShapes[NCHW_C];
        tg_src_tiled.shape.h = srcTiledShapes[NCHW_H];
        tg_src_tiled.shape.w = srcTiledShapes[NCHW_W];
        tg_src_tiled.stride.n = srcStrides[NCHW_N];
        tg_src_tiled.stride.c = srcStrides[NCHW_C];
        tg_src_tiled.stride.h = srcStrides[NCHW_H];

        cvk_tl_shape_t tl_dst_tiled_shape = {
            srcTiledShapes[NCHW_N], srcTiledShapes[NCHW_C],
            srcTiledShapes[NCHW_H], srcTiledShapes[NCHW_W]};
        tl_load_dst_tiled =
            ctx.lmem_alloc_tensor(tl_dst_tiled_shape, fmt, eu_align);

        cvk_tdma_g2l_tensor_copy_param_t param;
        param.src = &tg_src_tiled;
        param.dst = tl_load_dst_tiled;
        ctx.tdma_g2l_tensor_copy(&param);
      }

      // 2. Tensor store, CW transpose
      {
        cvk_tg_t tg_dst_tiled;
        memset(&tg_dst_tiled, 0, sizeof(tg_dst_tiled));
        tg_dst_tiled.base_reg_index =
            ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
        tg_dst_tiled.start_address = ga_ofmap + dstOffset;
        tg_dst_tiled.fmt = fmt;
        tg_dst_tiled.shape.n = tl_load_dst_tiled->shape.n;
        tg_dst_tiled.shape.c = tl_load_dst_tiled->shape.w; // CW transpose
        tg_dst_tiled.shape.h = tl_load_dst_tiled->shape.h;
        tg_dst_tiled.shape.w = tl_load_dst_tiled->shape.c; // CW transpose
        tg_dst_tiled.stride.n = dstStrides[NCHW_N];
        tg_dst_tiled.stride.c = dstStrides[NCHW_C];
        tg_dst_tiled.stride.h = dstStrides[NCHW_H];

        cvk_tdma_l2g_tensor_copy_cw_transposed_param_t param;
        param.src = tl_load_dst_tiled;
        param.dst = &tg_dst_tiled;
        ctx.tdma_l2g_tensor_copy_cw_transposed(&param);
      }

      // Free local memory
      ctx.lmem_free_tensor(tl_load_dst_tiled);
    }
  }

}

static void permute_nc_tp(
    const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, cvk_fmt_t fmt) {

  int n = input_n;
  int c = input_c;
  int h = 1;
  int w = input_h * input_w;
  int element_size = (fmt == CVK_FMT_BF16) ? 2 : 1;

  int max_w = std::min(w, MAX_WIDTH);
  int step_w = max_w;
  int step_c = c;
  int step_n = n;
  uint32_t lmem_required = (uint32_t)LOCAL_MEM_SIZE + 1;
  cvk_tl_shape_t shape;
  for (step_w = max_w; step_w > 0; --step_w) {
    for (step_c = c; step_c > 0; --step_c) {
      for (step_n = n; step_n > 0; step_n -= NPU_NUM) {
        shape = ctx.shape_t4(step_c, step_n, h, step_w);
        lmem_required = ctx.lmem_tensor_to_size(shape, fmt, 1);
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

  llvm::errs() << llvm::format("step, %d,%d,%d,%d\n", step_n, step_c, h, step_w);

  cvk_tl_t *tl_a = ctx.lmem_alloc_tensor(shape, fmt, 1);
  assert(tl_a);

  for (int pos_n = 0; pos_n < n; pos_n += step_n) {
    int cur_n = std::min(n - pos_n, step_n);
    for (int pos_c = 0; pos_c < c; pos_c += step_c) {
      int cur_c = std::min(c - pos_c, step_c);
      for (int pos_w = 0; pos_w < w; pos_w += step_w) {
        int cur_w = std::min(w - pos_w, step_w);
        shape = ctx.shape_t4(cur_c, cur_n, h, cur_w);

        cvk_tl_t tensor;
        tensor.start_address = tl_a->start_address;
        tensor.shape = shape;
        tensor.stride = ctx.tl_default_stride(shape, fmt, 1);
        tensor.fmt = fmt;
        uint64_t offset = (pos_w + pos_c * w + pos_n * c * w) * element_size;
        auto stride = ctx.tg_default_stride({(uint32_t)n, (uint32_t)c, (uint32_t)h, (uint32_t)w}, fmt);
        ctx.tdma_load_stride_bf16(&tensor, ga_ifmap + offset, stride, 1);

        LLVM_DEBUG(llvm::errs() << llvm::format("load, shape:%d,%d,%d,%d, stride: %d,%d,%d,%d, offset:%d\n",
                  cur_n, cur_c, h, cur_w,
                  (int)stride.n, (int)stride.c, (int)stride.h, (int)stride.w,
                  (int)offset));

        shape = ctx.shape_t4(cur_c, cur_n, h, cur_w);
        tensor.shape = shape;
        tensor.stride = ctx.tl_default_stride(shape, fmt, 1);
        offset = (pos_w + pos_n * w + pos_c * n * w) * element_size;
        stride = ctx.tg_default_stride({(uint32_t)c, (uint32_t)n, (uint32_t)h, (uint32_t)w}, fmt);
        ctx.tdma_store_stride_bf16(&tensor, ga_ofmap + offset, stride);
        LLVM_DEBUG(llvm::errs() << llvm::format("store, shape:%d,%d,%d,%d, stride: %d,%d,%d,%d, offset:%d\n",
                  cur_c, cur_n, h, cur_w,
                  (int)stride.n, (int)stride.c, (int)stride.h, (int)stride.w,
                  (int)offset));
      }
    }
  }

  ctx.lmem_free_tensor(tl_a);
}

void cvi_backend_tg_permute(
    const CviBackendContext& ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, uint32_t input_n, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_n, uint32_t output_c, uint32_t output_h,
    uint32_t output_w, uint32_t order_n, uint32_t order_c, uint32_t order_h,
    uint32_t order_w, bool do_permute, cvk_fmt_t fmt) {

  // For tdma
  ctx.set_layer_id(layer_id);

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "cvi_backend_tg_permute:\n"
      "  ga_ifmap 0x%lx, ga_ofmap 0x%lx\n"
      "  in(%d, %d, %d, %d), out(%d, %d, %d, %d)\n"
      "  order(%d, %d, %d, %d), do_permute %d\n",
      ga_ifmap, ga_ofmap, input_n, input_c, input_h, input_w, output_n,
      output_c, output_h, output_w, order_n, order_c, order_h, order_w,
      do_permute););

  if (!do_permute) {
    if (ga_ofmap != ga_ifmap) {
      cvk_tg_shape_t shape_ = {output_n, output_c, output_h, output_w};
      cvk_tg_stride_t stride_ = ctx.tg_default_stride(shape_, fmt);

      cvk_tg_t tg_src;
      tg_src.start_address = ga_ifmap;
      tg_src.fmt = fmt;
      tg_src.shape = shape_;
      tg_src.stride = stride_;

      cvk_tg_t tg_dst;
      tg_dst.start_address = ga_ofmap;
      tg_dst.fmt = fmt;
      tg_dst.shape = shape_;
      tg_dst.stride = stride_;

      ctx.tdma_tg_copy(&tg_dst, &tg_src);
    }
    return;
  }

  if (order_n == 1 && order_c == 0 && order_h == 2 && order_w == 3) {
    permute_nc_tp(ctx, layer_id, ga_ifmap, ga_ofmap, input_n, input_c, input_h,
                  input_w, fmt);
  } else if (order_w == 3) {
    // (0, 1, 2, 3) -> (, , , 3)
    // E.g. yolo_v2, 0213, original 012435
    permute_xxx3_tp(ctx, layer_id, ga_ifmap, ga_ofmap, input_n, input_c,
                    input_h, input_w, output_c, output_h, output_w, order_n,
                    order_c, order_h, order_w, fmt);
  } else if ((order_n == 0) && (order_c == 2) && (order_h == 3) &&
             (order_w == 1)) {
    // 0231 : (N, C, H, W) -> (N, H, W, C)
    // Eg. ssd300
    permute_0231_cw_tp(ctx, layer_id, ga_ifmap, ga_ofmap, input_n, input_c,
                       input_h, input_w, output_c, output_h, output_w,
                       order_n, order_c, order_h, order_w, fmt);
  } else if ((order_n == 0)
    && (order_c == 3) && (order_h == 1) && (order_w == 2)) {
    // nhwc to nchw tranpose order is 0312
    permute_0312_tp(ctx, layer_id, ga_ifmap, ga_ofmap, input_n, input_c,
                       input_h, input_w, output_c, output_h, output_w, order_n,
                       order_c, order_h, order_w, fmt);
  } else if ((order_n == 0) && (order_c == 3) && (order_h == 2) &&
             (order_w == 1)) {
    // 0321, (N, C, H, W) -> (N, W, H, C)
    // E.g. reduceOp with axis [1] needs CW transpose in global memory
    permute_0321_tp(ctx, layer_id, ga_ifmap, ga_ofmap, input_n, input_c,
                    input_h, input_w, output_c, output_h, output_w, order_n,
                    order_c, order_h, order_w, fmt);
  } else {
    llvm::errs() << "Not support order (" << order_n << ", " << order_c << ", " << order_h << ", " << order_w << ") permute case\n";
    assert(0);
  }
}

void cvi_backend_tg_fixed_premute_kernel(
    const CviBackendContext& ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t* depends, uint32_t depends_len,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c, int input_h,
    int input_w, int output_n, int output_c, int output_h, int output_w,
    int order_n, int order_c, int order_h, int order_w, bool do_permute) {
  cvi_backend_tg_permute(ctx, layer_id, ga_ifmap, ga_ofmap, input_n,
                         input_c, input_h, input_w, output_n, output_c,
                         output_h, output_w, order_n, order_c, order_h,
                         order_w, do_permute, CVK_FMT_I8);
}

void cvi_backend_tg_bf16_premute_kernel(
    const CviBackendContext& ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t* depends, uint32_t depends_len,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n, int input_c, int input_h,
    int input_w, int output_n, int output_c, int output_h, int output_w,
    int order_n, int order_c, int order_h, int order_w, bool do_permute) {
  cvi_backend_tg_permute(ctx, layer_id, ga_ifmap, ga_ofmap, input_n,
                         input_c, input_h, input_w, output_n, output_c,
                         output_h, output_w, order_n, order_c, order_h,
                         order_w, do_permute, CVK_FMT_BF16);
}
