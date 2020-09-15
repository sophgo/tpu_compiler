/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_tiling.h
 * Description:
 */

#ifndef CVI_BACKEND_TILING_API
#define CVI_BACKEND_TILING_API

class CviBackendContext;

int bitsize_of_fmt(uint32_t fmt);
int ceiling_bytesize_of(int bitsize);
int bytesize_of_fmt(cvk_fmt_t fmt);

void tiling_packing(const CviBackendContext &ctx, int require_shape, int coeff_lane_shape,
                    int blob_num, cvk_fmt_t fmt,
                    std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > *tiling_info);

void _split_nh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
               uint32_t reserved, int *n_slices, int *h_slices);

void _split_cnh(const CviBackendContext &ctx, int n, int c, int h, int w, int blob_num,
                uint32_t reserved, int *c_slices, int *n_slices, int *h_slices);

int __split(const CviBackendContext &ctx, int blob_num, int count);

int shape_size(int n, int c, int h, int w, cvk_fmt_t fmt);
#endif /* CVI_BACKEND_TILING_API */
