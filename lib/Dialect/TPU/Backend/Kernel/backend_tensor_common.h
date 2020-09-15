/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: backend_tensor_common.h
 * Description:
 */

#ifndef CVI_BACKEND_TENSOR_COMMON_API
#define CVI_BACKEND_TENSOR_COMMON_API

class CviBackendContext;

int get_csize_local(const CviBackendContext &ctx, int h, int w);

uint32_t __get_lmem_usage(const CviBackendContext &ctx, int n, int c, int h, int w);

int tensor_size_lmem(const CviBackendContext &ctx, int n, int c, int h, int w);

int __get_csize_local(const CviBackendContext &ctx, int h, int w, cvk_fmt_t fmt);

uint32_t ___get_lmem_usage(const CviBackendContext &ctx, int n, int c, int h, int w, cvk_fmt_t fmt);

int _tensor_size_lmem(const CviBackendContext &ctx, int n, int c, int h, int w, cvk_fmt_t fmt);

void init_tensor_tgmem(const CviBackendContext &ctx, cvk_tg_t *t,
                       uint64_t start_address, cvk_tg_shape_t shape,
                       cvk_tg_stride_t stride, cvk_fmt_t fmt);

int is_one_bf16(cvk_fmt_t src, cvk_fmt_t dst) ;

int is_support_fmt(cvk_fmt_t fmt) ;

int is_bf16(cvk_fmt_t fmt);

int _get_csize_local(const CviBackendContext &ctx, int h, int w, cvk_fmt_t fmt);

void _tdma_g2g_tensor_copy(const CviBackendContext &ctx, cvk_tg_t *src,
                           cvk_tg_t *dst);


void tdma_g2g_tensor_copy(
    // src
    const CviBackendContext &ctx, uint64_t src_start_address,
    cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    cvk_fmt_t src_fmt,
    // dst
    uint64_t dst_start_address, cvk_tg_shape_t dst_shape,
    cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt);


int getQuantizeMode(const int *i8_multiplier);

void load_bias_multiplier(const CviBackendContext &ctx,
                          int oc_step,  // output channel
                          bool do_bias, gaddr_t bias_gaddr, int qmode,
                          cvk_tl_t **tl_bias);

void load_32byte_multiplier(const CviBackendContext &ctx, int oc_step, bool do_bias,
                            gaddr_t bias_gaddr, cvk_tl_t **tl_chl_quan_param);

void load_16bytes_bias(const CviBackendContext &ctx, int oc, cvk_tl_t **tl_bias,
                       gaddr_t bias_gaddr);


// copy same shape from system to local
void tdma_g2l_tensor_copy(const CviBackendContext &ctx, cvk_tl_t **tl_bslice,
                          int input_n, int input_c, int input_h, int input_w, gaddr_t input_gaddr,
                          cvk_fmt_t fmt, int eu_align = 1);


// apply quantize int 8 mode
void apply_qi8(const CviBackendContext &ctx, cvk_tl_t *ifmap, uint32_t layer_id, int do_relu,
               int right_shift_width, int threshold_x_quantized);

/*
 * \brief fill fp32 range to 0
 *
 * we tiling all local memory and seperate fp32 / bf16 region
 * fill fp32 region to 0 for export fp32 format
 * for instance:
 *
 *  0       16      32              64       80        96
 *  +------fp0------+------fp1------+-bf16_0--+--bf16_1-+
 *  +
 *  |0x0|0x0|0x0|0x0|0x0|0x0|0x0|0x0|0x13|0x14|0x13|0x23|
 *  +
 *
 *  and we could copy bf16 region with stride to convert fp32 format
 *
 *  0       16        32                64        80        96
 *  +------fp0--------+------fp1--------+--bf16_0--+--bf16_1-+
 *  +
 *  |0x0|0x0|0x13|0x14|0x0|0x0|0x13|0x23|0x13|0x14|0x13|0x23|
 *  +
 *
 */
void fill_fp32_lmem_0(const CviBackendContext &ctx, uint32_t layer_id,
                      int batch, int channel, int height, int width);
/*
 * \brief truncat fp32 low 16bit and concat it
 *
 * it will overwrite itself with different stride,
 * for instance:
 *  fp32 layout in lmem
 *
 *  0         16        32         48       64
 *  +--------fp0--------+--------fp1--------+
 *  +
 *  |0xaa|0x12|0x13|0x14|0xaa|0x12|0x13|0x23|
 *  +
 *
 *  shrink it to bf16, takes high 16bits of fp32,
 *  thie memory layout could be:
 *
 *  0         16        32
 *  +--bf16_0--+--bf16_1+
 *  +
 *  |0x13|0x14|0x13|0x23|
 *  +
 *
 *  \bottom_fp32 fp32 lmem pointer, it should NOT eu_align
 *  \bottom_bf16 bf16 lmem pointer, it should NOT eu_align
 *
 */
void lmem_shrink_fp32_bf16(const CviBackendContext &ctx,
                           cvk_tl_t* lmem_bf16, cvk_tl_t* lmem_fp32,
                           int bf16_n, int bf16_c, int bf16_h, int bf16_w, uint32_t layer_id);

/*
 * \eu_align 0 means do not eu align
 */
cvk_tl_stride_t tl_fp32_stride(const CviBackendContext &ctx, cvk_tl_t* tl, int eu_align = 0);
#endif /* CVI_BACKEND_TENSOR_COMMON_API */
