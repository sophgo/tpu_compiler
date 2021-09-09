#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "TgWhereKernel"

static void tiu_add_const(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* tl_ifmap, cvk_tl_t* tl_ofmap, cvk_tl_t* __tl_working,
        float fill_constant) {

    cvk_tl_t *tl_working = NULL;
    cvk_tl_t _tl_working;

    int _fill_constant = fill_constant;
    if (tl_ofmap->fmt == CVK_FMT_BF16) {
        _fill_constant = ctx.convert_fp32_to_bf16(fill_constant);
    }
    else {
        _tl_working = *__tl_working;
        tl_working = &_tl_working;
        tl_working->shape = tl_ofmap->shape;
        tl_working->stride = tl_ofmap->stride;

        // int8
        ctx.tiu_zeros(layer_id, tl_working);

        // need to hoist to int16 for hw
        cvk_tiu_mul_param_t p = {0};
        p.res_high = tl_working;
        p.res_low = tl_ifmap;
        p.a = tl_ifmap;
        p.b_const.val = 1;
        p.rshift_bits = 0;
        p.b_const.is_signed = false;
        p.b_is_const = 1;
        p.layer_id = layer_id;
        p.relu_enable = 0;
        ctx.tiu_mul(&p);
    }

    cvk_tiu_add_param_t p1 = {0};
    p1.res_high = NULL;
    p1.res_low = tl_ofmap;
    p1.a_high = tl_working;
    p1.a_low = tl_ifmap;
    p1.b_is_const = 1;
    p1.b_const.val = _fill_constant;
    p1.b_const.is_signed = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_add(&p1);
}

static void tiu_add(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* tl_ifmap, cvk_tl_t* tl_ofmap, cvk_tl_t* tl_working) {

    bool res_is_int8 = true;
    int _fill_constant = 1;

    if (tl_ofmap->fmt == CVK_FMT_BF16) {
        res_is_int8 = false;
        _fill_constant = ctx.convert_fp32_to_bf16(_fill_constant);
    }
    else {
        // int8
        ctx.tiu_zeros(layer_id, tl_working);
    }

    cvk_tiu_mac_param_t p4 = {0};
    p4.res_high = tl_working;
    p4.res_low = tl_ofmap;
    p4.res_is_int8 = res_is_int8;
    p4.a = tl_ifmap;
    p4.b_const.val = _fill_constant;
    p4.b_is_const = 1;
    p4.b_const.is_signed = 1;
    p4.lshift_bits = 0;
    p4.rshift_bits = 0;
    p4.relu_enable = 0;
    ctx.tiu_mac(&p4);
}

static void tiu_requant(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* tl_ofmap, int32_t rshift, const int32_t multiplier) {

    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = tl_ofmap;
    p.a = tl_ofmap;
    p.b_const.val = multiplier;
    p.rshift_bits = rshift;
    p.b_const.is_signed = false;
    p.b_is_const = 1;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
}


static void tiu_copy_dup_w(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* tl_ifmap, cvk_tl_t* tl_ofmap, bool is_broadcast_w) {

    cvk_tl_t tl_src = *tl_ifmap;
    cvk_tl_t tl_dst = *tl_ofmap;

    if (is_broadcast_w) {
        tl_src.stride.w = 0;
    }

    tl_src.shape = tl_dst.shape;

    cvk_tiu_copy_param_t p2 = {0};
    p2.src = &tl_src;
    p2.dst = &tl_dst;
    p2.layer_id = layer_id;
    ctx.tiu_copy(&p2);
}

static void tiu_mul_broadcast(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* _tl_lhs, cvk_tl_t* _tl_rhs, cvk_tl_t* _tl_ofmap,
        uint32_t* rhs_org_shape) {

    assert(_tl_lhs->shape.w == _tl_rhs->shape.w && "rhs(condition) has broadcasted");
    int cn, cc, ch, cw;
    cn = rhs_org_shape[0];
    cc = rhs_org_shape[1];
    ch = rhs_org_shape[2];
    cw = rhs_org_shape[3];

    // org shape means no-tile shape
    cvk_tl_t tl_lhs = *_tl_lhs;
    cvk_tl_t tl_rhs = *_tl_rhs;
    cvk_tl_t tl_ofmap = *_tl_ofmap;

    // rhs as boradcase, for hw limitation that MUST has the same shape
    tl_rhs.shape = tl_lhs.shape;

    // set stride to broadcast, copy from lib/Dialect/TPU/Interpreter/core/where.cpp
    tl_rhs.stride.h = ch * cw == 1 ? 0 : tl_rhs.stride.h;
    tl_rhs.stride.c = cc * ch * cw == 1 ? 0 : tl_rhs.stride.c;
    tl_rhs.stride.n = cn * cc * ch * cw == 1 ? 0 : tl_rhs.stride.n;


    // tl_rhs as condition_shape
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_ofmap;
    p.a = &tl_lhs;
    p.b = &tl_rhs;
    p.b_is_const = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = false;
    ctx.tiu_mul(&p);
}


// flip mean 0/1 flip, e.g: 0/1->1/0
static void tiu_flip_mask(const CviBackendContext &ctx, uint32_t layer_id,
        cvk_tl_t* tl_ifmap, cvk_tl_t* tl_ofmap, cvk_tl_t* tl_working) {

    int _fill_constant = -1;
    if (tl_ofmap->fmt == CVK_FMT_BF16) {
        _fill_constant = ctx.convert_fp32_to_bf16(_fill_constant);
    }
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = tl_ofmap;
    p.a = tl_ifmap;
    p.b_const.val = _fill_constant;
    p.b_const.is_signed = true;
    p.b_is_const = true;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    tiu_add_const(ctx, layer_id, tl_ofmap, tl_ofmap, tl_working, /*fill_constant=*/1);
}

void cvi_backend_tg_where_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_condition, gaddr_t ga_ofmap,
    uint32_t* input_shape, uint32_t* condition_shape,
    int32_t rshift, const int32_t multiplier,
    float fill_constant, cvk_fmt_t fmt) {

    int n, c, h, w;
    n = input_shape[0];
    c = input_shape[1];
    h = input_shape[2];
    w = input_shape[3];

    int cn, cc, ch, cw;
    cn = condition_shape[0];
    cc = condition_shape[1];
    ch = condition_shape[2];
    cw = condition_shape[3];

    assert ((fmt == CVK_FMT_BF16 || fmt == CVK_FMT_I8) && "only support i8/bf16 format");
    bool isBF16 = fmt == CVK_FMT_BF16;

    cvk_tg_shape_t i_s = ctx.tg_shape_t4(n, c, h, w);
    cvk_tg_stride_t i_gstride = ctx.tg_default_stride(i_s, fmt);

    // 1. x = copy(y)
    // 2. x = zero(x)
    // 3. x = x + fill_constant
    // 4. x = x * condition
    // 5. condition = condition * -1
    // 6. condition = condition + 1
    // 7. y = y * condition
    // 8. y = y + x

    std::vector<CviBackendContext::tiling_info_t> tiles;
    int eu_align = 1;
    int blob_num = 2; // x, y
    // set w rather than cw cuz tiu_mul not support set stride_w
    auto condition_size = ctx.lmem_tensor_to_size(cn, cc, ch, w, fmt, eu_align);
    condition_size = condition_size * 2; // 2 means leave origin one
    if (!isBF16) {
        blob_num = blob_num + 1; // extra 1 for working
    }

    ctx.tiling_packing(tiles, n, c, h, w, fmt,
            blob_num, condition_size,
            CviBackendContext::TilingNCH);

    auto _condition_shape = ctx.tl_shape_t4(cn, cc, ch, cw);
    cvk_tl_t *_tl_condition = ctx.lmem_alloc_tensor(_condition_shape, fmt, eu_align);

    // reuse rhs
    ctx.tdma_load(_tl_condition, ga_condition);

    for (auto &tile : tiles) {
        assert((uint32_t)tile.w == (uint32_t)(w) && "not support tile w");
        auto shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
        cvk_tl_t *tl_ifmap = ctx.lmem_alloc_tensor(shape, fmt, eu_align);
        cvk_tl_t *tl_ofmap = ctx.lmem_alloc_tensor(shape, fmt, eu_align);

        auto __condition_shape = _condition_shape;
        __condition_shape.w = tile.w;
        cvk_tl_t *tl_condition = ctx.lmem_alloc_tensor(__condition_shape,
                fmt, eu_align);

        cvk_tl_t *tl_working = NULL;
        if (!isBF16) {
            tl_working = ctx.lmem_alloc_tensor(shape, fmt, eu_align);
        }

        // tl_condition will be dirted later, keep origin one
        tiu_copy_dup_w(ctx, layer_id, _tl_condition, tl_condition, cw == 1);

        // load input
        ctx.tdma_load_stride(tl_ifmap, ga_ifmap + tile.offset, i_gstride);

        // 1. x = copy(y)
        tiu_copy_dup_w(ctx, layer_id, tl_ifmap, tl_ofmap, /*is_broadcast_w=*/false);
        // 2. x = zero(x)
        ctx.tiu_zeros(layer_id, tl_ofmap);

        // 3. x = x + fill_constant
        tiu_add_const(ctx, layer_id, tl_ofmap, tl_ofmap, tl_working, fill_constant);
        // 4. x = x * condition
        tiu_mul_broadcast(ctx, layer_id, tl_ofmap, tl_condition, tl_ofmap, condition_shape);

        // 5. condition = condition * -1
        // 6. condition = condition + 1
        tiu_flip_mask(ctx, layer_id, tl_condition, tl_condition, tl_working);

        // 7. y = y * condition
        tiu_mul_broadcast(ctx, layer_id, tl_ifmap, tl_condition, tl_ifmap,
                condition_shape);

        // 8. y = y + x
        tiu_add(ctx, layer_id, tl_ifmap, tl_ofmap, tl_working);

        if (!isBF16) {
            // quant if int8
            tiu_requant(ctx, layer_id, tl_ofmap, rshift, multiplier);
        }

        ctx.tdma_store_stride(tl_ofmap, ga_ofmap + tile.offset, i_gstride);

        if (tl_working) {
            ctx.lmem_free_tensor(tl_working);
        }
        ctx.lmem_free_tensor(tl_condition);
        ctx.lmem_free_tensor(tl_ofmap);
        ctx.lmem_free_tensor(tl_ifmap);
    }

    ctx.lmem_free_tensor(_tl_condition);
}
