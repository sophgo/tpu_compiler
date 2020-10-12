/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: internal_tiling.h
 * Description:
 */

#ifndef CVI_BACKEND_INTERNAL_TILING_API
#define CVI_BACKEND_INTERNAL_TILING_API

class CviBackendContext;

enum TilingDim {
  TilingDimAll = 0, // reshape data and tiling
  TilingDimNH,      // keep shape and ONLY tiling n/h dim
  TilingDimNo,      // no tiling
};

/**
 * \brief tiling pack data with specified dims
 * \shape for \TilingDimNH used, we need to keep origin shape and tile
 *  with specified dims
 * \blob_num blob number in lmem at same time, start with 1
 * \coeff_lane_shape tensor size of coefficient(bias, etc) in lmem,
 *  the tensor size SHOULD reflect with the \fmt, e.g.: the coeff_lane_shape of
 *  <1x1x2x3xi8> should be 6 and <1x1x2x3xbf16> should be 12 that bf16
 *  takes twice size than i8
 *
 * \tiling_info store tiling info in each steps and second shift size
 * reflect with \fmt
 */
void tiling_packing(const CviBackendContext &ctx, int require_shape, int coeff_lane_shape,
                    int blob_num, cvk_fmt_t fmt,
                    std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > *tiling_info,
                    enum TilingDim tiling_along = TilingDimAll,
                    cvk_tg_shape_t* shape = NULL);

void _split_nh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
               uint32_t reserved, int *n_slices, int *h_slices);

void _split_cnh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
                uint32_t reserved, int *c_slices, int *n_slices, int *h_slices);

int __split(const CviBackendContext &ctx, int blob_num, int count);

int shape_size(int n, int c, int h, int w, cvk_fmt_t fmt);
#endif /* CVI_BACKEND_INTERNAL_TILING_API */
