/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_common.h
 * Description: Intend to provide backend common functionality.
 */

#ifndef CVI_BACKEND_COMMON
#define CVI_BACKEND_COMMON

#include "cvikernel/cvikernel.h"

enum CVI_CHIP_INFO_E {
  CVI_CHIP_VERSION = 0,
  CVI_CHIP_NODE_SHIFT,
  CVI_CHIP_LANE_NUM,
  CVI_CHIP_LANE_SHIFT,
  CVI_CHIP_EU_NUM,
  CVI_CHIP_EU_SHIFT,
  CVI_CHIP_LMEM_SIZE,
  CVI_CHIP_LMEM_BANK,
  CVI_CHIP_INVALID,
};

typedef uint32_t laddr_t;
typedef uint64_t gaddr_t;

#define LA_INVALID     0xffffffffUL
#define GA_INVALID     0xffffffffffffffffULL

#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define ALIGN(x,a)              __ALIGN_MASK(x,(__typeof__(x))(a)-1)

static inline int ceiling_func(int numerator, int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

static inline uint64_t align_up(uint64_t x, uint64_t n)
{
  return ceiling_func(x, n) * n;
}

class CviBackendContext;

CviBackendContext *cvi_backend_create_context(const char *runchip);

void cvi_backend_submit(
    CviBackendContext *ctx);

void cvi_backend_get_cmdbuf(
    CviBackendContext *ctx, std::vector<uint8_t> &cmdbuf);

void cvi_backend_parallel_enable(CviBackendContext *ctx);

void cvi_backend_parallel_disable(CviBackendContext *ctx);

int cvi_backend_chip_context(CviBackendContext *ctx, CVI_CHIP_INFO_E cvi_chip_info_e);

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id);
#endif /* CVI_BACKEND_COMMON */
