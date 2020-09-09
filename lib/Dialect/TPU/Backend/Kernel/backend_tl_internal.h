/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_tl_internal.h
 * Description:
 */

#ifndef CVI_BACKEND_TL_INTERNAL
#define CVI_BACKEND_TL_INTERNAL

class CviBackendContext;



void bf16_lut_tl_forward_kernel(const CviBackendContext &ctx,
    laddr_t la_ifmap,
    laddr_t la_buf,
    laddr_t la_table_answer,
    laddr_t la_table_answer_mantissa,
    laddr_t la_ofmap,
    uint32_t tensor_n, uint32_t tensor_c, uint32_t tensor_h, uint32_t tensor_w,
    uint32_t table_n, uint32_t table_c, uint32_t table_h, uint32_t table_w,
    uint8_t eu_align, cvk_fmt_t fmt);



///
/// Load Input Neuron from gmem to lmem
///
void cvi_backend_tl_load_i8_al(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap,
    uint32_t input_n, uint32_t input_c, uint32_t input_h, uint32_t input_w);

///
/// Store Input Neuron from lmem to gmem
///
void cvi_backend_tl_store_i8_al(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ofmap, gaddr_t ga_ofmap,
    uint32_t input_n, uint32_t output_c, uint32_t output_h, uint32_t output_w);

// FIXME: sync with \MlirToBackendTranslate.cpp
void _cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw, cvk_fmt_t fmt,
    uint8_t eu_align);

void _cvi_backend_tl_store(
    const CviBackendContext &ctx, uint32_t layer_id, laddr_t la_ofmap,
    gaddr_t ga_ofmap, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow,
    cvk_fmt_t fmt, uint8_t eu_align);

void cvi_backend_tl_load_tensor(
    const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tensor,
    gaddr_t ga_ifmap, uint8_t eu_align);

void cvi_backend_tl_store_tensor(
    const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tensor,
    gaddr_t ga_ofmap, uint8_t eu_align);

void cvi_backend_tl_to_tensor(
    const CviBackendContext &ctx,
    cvk_tl_t *tensor,
    laddr_t la,
    uint32_t tensor_n, uint32_t tensor_c, uint32_t tensor_h, uint32_t tensor_w,
    cvk_fmt_t fmt, uint8_t eu_align);

#endif /* CVI_BACKEND_TL_INTERNAL */
