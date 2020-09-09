/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_common.h
 * Description: Intend to provide backend common functionality.
 */

#ifndef CVI_BACKEND_COMMON
#define CVI_BACKEND_COMMON

typedef enum CVI_BACKEND_FMT_E {
  CVI_FMT_F32 = 0,
  CVI_FMT_F16,
  CVI_FMT_I32,
  CVI_FMT_I16,
  CVI_FMT_I8,
  CVI_FMT_I4,
  CVI_FMT_I2,
  CVI_FMT_I1,
  CVI_FMT_U32,
  CVI_FMT_U16,
  CVI_FMT_U8,
  CVI_FMT_BF16,
  CVI_FMT_INVALID
} cvi_backend_fmt_t;

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

CviBackendContext *cvi_backend_create_context(
    std::vector<int8_t> &weight_data);

CviBackendContext *cvi_backend_create_context_chip(
    std::vector<int8_t> &weight_data, const char *runchip);

void cvi_backend_submit(
    CviBackendContext *ctx);

void cvi_backend_get_cmdbuf(
    CviBackendContext *ctx, std::vector<uint8_t> &cmdbuf);

void cvi_backend_parallel_enable(CviBackendContext *ctx);

void cvi_backend_parallel_disable(CviBackendContext *ctx);

int cvi_backend_chip_context(CviBackendContext *ctx, CVI_CHIP_INFO_E cvi_chip_info_e);

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id);
#endif /* CVI_BACKEND_COMMON */
