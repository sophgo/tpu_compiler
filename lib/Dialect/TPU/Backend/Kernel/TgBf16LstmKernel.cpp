/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_lstm.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "cvi_backend_lstm_kernel"

#define ASSERT(x) assert(x)

void cvi_backend_tg_bf16_lstm_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                            gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_recurrence,
                                            gaddr_t ga_bias, gaddr_t ga_initial_h, gaddr_t ga_initial_c,
                                            gaddr_t ga_sigmoid_table_data_lut, gaddr_t ga_sigmoid_slope_table_data_lut,
                                            gaddr_t ga_tanh_table_data_lut, gaddr_t ga_tanh_table_slope_data_lut,
                                            gaddr_t ga_output,
                                            int seq_len, int batch_size, int input_size, int hidden_size,
                                            bool do_bias, bool is_bidirectional) {
    // Ref:   https://github.com/onnx/onnx/blob/v1.7.0/onnx/backend/test/case/node/lstm.py#L65-L86
    // weight: Concatenation of weight matrix for update, reset, and hidden gates. shape = [num_directions, 4*hidden_size, input_size]
    // recurrence: Concatenation of recurrence weight matrix for update, reset, and hidden gates
    // bias: Concatenation of Wb[update, reset, hidden gates] and Rb[update, reset, hidden gates], shape = [num_directions, 6*hidden_size]
    // initial_h: [num_directions, batch_size, hidden_size]
    // initial_c: [num_directions, batch_size, hidden_size]
    // Ignore P[ifo]
    // Finish all the operation in one lane
    // TODO:Replace convolution operation with fc operation
    // TODO:Use parallel enable to increase tiu/tdma uRate
    assert(batch_size == 1);
    uint8_t eu_align = 1; // hardware constrainst
    //Load input
    cvk_tl_shape_t reshape_input_shape = {
        static_cast<uint32_t>(batch_size * seq_len), static_cast<uint32_t>(input_size),
        static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    cvk_tl_t *tl_input =
        ctx.lmem_alloc_tensor(reshape_input_shape, CVK_FMT_BF16, eu_align);
    ASSERT(tl_input);
    ctx.tdma_load_bf16(tl_input, ga_input);
    //Load weight (Wi + Wf + Wc + Wo)
    cvk_tl_shape_t  weight_shape = {
        static_cast<uint32_t>(1), static_cast<uint32_t>( 4 * hidden_size),
        static_cast<uint32_t>(1), static_cast<uint32_t>(input_size)};
    cvk_tl_t *tl_weight =
        ctx.lmem_alloc_tensor(weight_shape, CVK_FMT_BF16, 0); //weight EU_ALIGN = False
    ASSERT(tl_weight);
    ctx.tdma_load_bf16(tl_weight, ga_weight);

    //Reshape weight shape/stride to match convolution constraint
    //Just fit the constraint
    tl_weight->shape = {(uint32_t)input_size, (uint32_t)(4 * hidden_size), 1, 1};
    tl_weight->stride = ctx.tl_default_stride(tl_weight->shape, CVK_FMT_BF16, /*eu_align=*/0);

    //Load recurrence (Ri + Rf + Rc + Ro)
    cvk_tl_shape_t reshape_recurrence_shape = {
        static_cast<uint32_t>(1), static_cast<uint32_t>(4 * hidden_size),
        static_cast<uint32_t>(1), static_cast<uint32_t>(hidden_size)};
    cvk_tl_t *tl_recurrence =
        ctx.lmem_alloc_tensor(reshape_recurrence_shape, CVK_FMT_BF16, 0); //weight EU_ALIGN = False
    ASSERT(tl_recurrence);
    ctx.tdma_load_bf16(tl_recurrence, ga_recurrence);

    //Reshape weight shape/stride to match convolution constraint
    //Just fit the constraint
    tl_recurrence->shape = {(uint32_t)hidden_size, (uint32_t)(4 * hidden_size), 1, 1};
    tl_recurrence->stride = ctx.tl_default_stride(tl_recurrence->shape, CVK_FMT_BF16, /*eu_align=*/0);

    //Load bias (Wbi + Wbf + Wbc + Wbo + Rbi + Rbf + Rbc + Rbo)
    cvk_tl_t *tl_wtBias;
    cvk_tl_t *tl_recurrenceBias;
    if(do_bias) {
        //set nstride
        cvk_tl_shape_t reshape_wtBias_shape = {
            static_cast<uint32_t>(2), static_cast<uint32_t>(4 * hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        tl_wtBias =
            ctx.lmem_alloc_tensor(reshape_wtBias_shape, CVK_FMT_BF16, 0); //weight EU_ALIGN = False
        ASSERT(tl_wtBias);
        cvk_tg_stride_t bias_gstride = ctx.tg_default_stride({2, (uint32_t)(4 * hidden_size), 1, 1}, CVK_FMT_BF16);
        bias_gstride.n *= 2;
        ctx.tdma_load_stride_bf16(tl_wtBias, ga_bias, bias_gstride);

        cvk_tl_shape_t reshape_recurrenceBias_shape = {
            static_cast<uint32_t>(2), static_cast<uint32_t>(4 * hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        tl_recurrenceBias =
            ctx.lmem_alloc_tensor(reshape_recurrenceBias_shape, CVK_FMT_BF16, 0); //weight EU_ALIGN = False
        ASSERT(tl_recurrenceBias);
        gaddr_t ga_recurrenceBias = ga_bias + 4 * hidden_size * sizeof(short);
        ctx.tdma_load_stride_bf16(tl_recurrenceBias, ga_recurrenceBias, bias_gstride);
    }
    //Load initial_h
    cvk_tl_shape_t reshape_initial_h_shape = {
            static_cast<uint32_t>(1), static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    cvk_tl_t *tl_initial_h =
            ctx.lmem_alloc_tensor(reshape_initial_h_shape, CVK_FMT_BF16, eu_align);
    ASSERT(tl_initial_h);
    ctx.tdma_load_bf16(tl_initial_h, ga_initial_h);

    //Load initial_c
    cvk_tl_shape_t reshape_initial_c_shape = {
            static_cast<uint32_t>(1), static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    cvk_tl_t *tl_initial_c =
            ctx.lmem_alloc_tensor(reshape_initial_c_shape, CVK_FMT_BF16, eu_align);
    ASSERT(tl_initial_c);
    ctx.tdma_load_bf16(tl_initial_c, ga_initial_c);

    //Load sigmoid table
    int const table_n = 1;
    int const table_c = NPU_NUM;
    int const table_h = 32;
    int const table_w = 8;

    cvk_tl_shape_t table_shape = {
        static_cast<uint32_t>(table_n), static_cast<uint32_t>(table_c),
        static_cast<uint32_t>(table_h), static_cast<uint32_t>(table_w)};

    cvk_tl_t *tl_sigmoid_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_sigmoid_table_answer_slope =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_sigmoid_table_answer);
    ASSERT(tl_sigmoid_table_answer_slope);

    ctx.tdma_load_bf16(tl_sigmoid_table_answer, ga_sigmoid_table_data_lut);
    ctx.tdma_load_bf16(tl_sigmoid_table_answer_slope, ga_sigmoid_slope_table_data_lut);
    //Load tanh table

    cvk_tl_t *tl_tanh_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_tanh_table_answer_slope =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_tanh_table_answer);
    ASSERT(tl_tanh_table_answer_slope);

    ctx.tdma_load_bf16(tl_tanh_table_answer, ga_tanh_table_data_lut);
    ctx.tdma_load_bf16(tl_tanh_table_answer_slope, ga_tanh_table_slope_data_lut);
    //Allocate output buffer
    cvk_tl_shape_t reshape_output_shape = {
            static_cast<uint32_t>(seq_len + 1), static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    cvk_tl_t *tl_output =
            ctx.lmem_alloc_tensor(reshape_output_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
    ASSERT(tl_output);
    //Allocate temp1 =  Xt*W
    cvk_tl_shape_t reshape_temp1_shape = {
            static_cast<uint32_t>(seq_len*batch_size), static_cast<uint32_t>(4 * hidden_size),
            static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    cvk_tl_t *tl_xt_mul_w =
            ctx.lmem_alloc_tensor(reshape_temp1_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
    ASSERT(tl_xt_mul_w);
    //compute Xt * (Wi + Wf + Wc + Wo) + (Wbi + Wbf + Wbc + Wbo)
    cvk_tiu_pt_convolution_param_t param = {0};
    param.ofmap = tl_xt_mul_w;
    param.ifmap = tl_input;
    param.weight = tl_weight;
    param.bias = tl_wtBias;
    param.ins_h =  0;
    param.ins_w = 0;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.relu_enable = 0;
    param.ps32_mode = 0;
    param.w_is_const = 0;
    param.layer_id = layer_id;

    LLVM_DEBUG(llvm::errs() << llvm::format(
                    "    conv:\n"
                    "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                    "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                    "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                    param.ifmap->start_address,
                    param.ifmap->shape.n, param.ifmap->shape.c, param.ifmap->shape.h,
                    param.ifmap->shape.w, param.weight->start_address,
                    param.weight->shape.n, param.weight->shape.c, param.weight->shape.h,
                    param.weight->shape.w, param.ofmap->start_address,
                    param.ofmap->shape.n, param.ofmap->shape.c, param.ofmap->shape.h,
                    param.ofmap->shape.w));

    ctx.tiu_pt_convolution(&param);

    //iteration start
    for(int i = 0; i < seq_len; i++){
        //part 1 = Xt*Wt
        //part 2 = Ht-1 * R
        //tl_it_ot_ft_part1 is part of tl_xt_mul_w

        cvk_tl_t tl_h_t_minus_1;
        tl_h_t_minus_1.start_address = (i == 0) ? tl_initial_h->start_address :  tl_output->start_address + ( i - 1 ) * tl_output->stride.n;  // start of lmem
        tl_h_t_minus_1.fmt = CVK_FMT_BF16;
        tl_h_t_minus_1.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_h_t_minus_1.stride = ctx.tl_default_stride(tl_h_t_minus_1.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_t tl_c_t_minus_1;
        tl_c_t_minus_1.start_address = tl_initial_c->start_address;  // start of lmem
        tl_c_t_minus_1.fmt = CVK_FMT_BF16;
        tl_c_t_minus_1.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_c_t_minus_1.stride = ctx.tl_default_stride(tl_c_t_minus_1.shape, CVK_FMT_BF16, /*eu_align=*/1);

        uint32_t tl_it_ot_ft_part1_startAddr = tl_xt_mul_w->start_address + i * tl_xt_mul_w->stride.n;
        cvk_tl_t tl_it_ot_ft_part1;
        tl_it_ot_ft_part1.start_address = tl_it_ot_ft_part1_startAddr;  // start of lmem
        tl_it_ot_ft_part1.fmt = CVK_FMT_BF16;
        tl_it_ot_ft_part1.shape = {1, (uint32_t)(3 * hidden_size), 1, 1};
        tl_it_ot_ft_part1.stride = ctx.tl_default_stride(tl_it_ot_ft_part1.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_shape_t reshape_h_mul_r_shape = {
                static_cast<uint32_t>(1), static_cast<uint32_t>(4 * hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_h_mul_r =
                ctx.lmem_alloc_tensor(reshape_h_mul_r_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_h_mul_r);
        //compute Ht-1 * (Rz + Rr + Rh) + (Rbz + Rbr + Rbh)
        cvk_tiu_pt_convolution_param_t param = {0};
        param.ofmap = tl_h_mul_r;
        param.ifmap = &tl_h_t_minus_1;
        param.weight = tl_recurrence;
        param.bias = tl_recurrenceBias;
        param.ins_h =  0;
        param.ins_w = 0;
        param.pad_top = 0;
        param.pad_bottom = 0;
        param.pad_left = 0;
        param.pad_right = 0;
        param.stride_h = 1;
        param.stride_w = 1;
        param.dilation_h = 1;
        param.dilation_w = 1;
        param.relu_enable = 0;
        param.ps32_mode = 0;
        param.w_is_const = 0;
        param.layer_id = layer_id;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    conv:\n"
                        "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                        "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                        "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                        param.ifmap->start_address,
                        param.ifmap->shape.n, param.ifmap->shape.c, param.ifmap->shape.h,
                        param.ifmap->shape.w, param.weight->start_address,
                        param.weight->shape.n, param.weight->shape.c, param.weight->shape.h,
                        param.weight->shape.w, param.ofmap->start_address,
                        param.ofmap->shape.n, param.ofmap->shape.c, param.ofmap->shape.h,
                        param.ofmap->shape.w));

        ctx.tiu_pt_convolution(&param);

        cvk_tl_t tl_it_ot_ft_h_mul_r;
        tl_it_ot_ft_h_mul_r.start_address = tl_h_mul_r->start_address;  // start of lmem
        tl_it_ot_ft_h_mul_r.fmt = CVK_FMT_BF16;
        tl_it_ot_ft_h_mul_r.shape = {1, (uint32_t)(3 * hidden_size), 1, 1};
        tl_it_ot_ft_h_mul_r.stride = ctx.tl_default_stride(tl_it_ot_ft_h_mul_r.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_shape_t reshape_zt_rt_lut_index_shape = {
                static_cast<uint32_t>(1), static_cast<uint32_t>(3 * hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_it_ot_ft_lut_index =
                ctx.lmem_alloc_tensor(reshape_zt_rt_lut_index_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_it_ot_ft_lut_index);
        //Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz
        //Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
        cvk_tiu_add_param_t p3 = {0};
        p3.res_high = nullptr;
        p3.res_low = tl_it_ot_ft_lut_index;
        p3.a_high = nullptr;
        p3.a_low = &tl_it_ot_ft_h_mul_r;
        p3.b_is_const = false;
        p3.b.high = nullptr;
        p3.b.low = &tl_it_ot_ft_part1;
        p3.rshift_bits = 0;
        p3.layer_id = layer_id;
        p3.relu_enable = 0;
        ctx.tiu_add(&p3);

        //working
        cvk_tl_shape_t reshape_temp4_shape = {
                static_cast<uint32_t>(2), static_cast<uint32_t>(3 * hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_temp4 =
                ctx.lmem_alloc_tensor(reshape_temp4_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_temp4);

        //output
        cvk_tl_shape_t reshape_it_ft_shape = {
                static_cast<uint32_t>(1), static_cast<uint32_t>(3 * hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_it_ot_ft =
                ctx.lmem_alloc_tensor(reshape_it_ft_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_it_ot_ft);

        // Ignore P[io]
        // line 1 equation it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        // line 2 equation ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
        const int table_thresh_min = -8;
        const int table_thresh_max = 8;

        cvi_backend_tl_lut(
        ctx, layer_id,
        tl_it_ot_ft_lut_index->start_address, tl_it_ot_ft->start_address, tl_temp4->start_address,
        tl_sigmoid_table_answer->start_address, tl_sigmoid_table_answer_slope->start_address,
        table_thresh_min, table_thresh_max, 1, 3 * hidden_size, 1, 1);

        cvk_tl_t tl_it; //get it
        tl_it.start_address = tl_it_ot_ft->start_address;  // start of lmem
        tl_it.fmt = CVK_FMT_BF16;
        tl_it.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_it.stride = ctx.tl_default_stride(tl_it.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_t tl_ot; //get ft
        tl_ot.start_address = tl_it_ot_ft->start_address + ceiling_func(hidden_size, NPU_NUM) * tl_it_ot_ft->stride.c;  // start of lmem
        tl_ot.fmt = CVK_FMT_BF16;
        tl_ot.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_ot.stride = ctx.tl_default_stride(tl_ot.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_t tl_ft; //get ft
        tl_ft.start_address = tl_it_ot_ft->start_address + ceiling_func(2 * hidden_size, NPU_NUM) * tl_it_ot_ft->stride.c;  // start of lmem
        tl_ft.fmt = CVK_FMT_BF16;
        tl_ft.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_ft.stride = ctx.tl_default_stride(tl_ft.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_t tl_ct_h_mul_r;
        tl_ct_h_mul_r.start_address = tl_h_mul_r->start_address + ceiling_func(3 * hidden_size, NPU_NUM) * tl_h_mul_r->stride.c;  // start of lmem
        tl_ct_h_mul_r.fmt = CVK_FMT_BF16;
        tl_ct_h_mul_r.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_ct_h_mul_r.stride = ctx.tl_default_stride(tl_ct_h_mul_r.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_shape_t reshape_ct_lut_index_shape = {
                static_cast<uint32_t>(1), static_cast<uint32_t>(hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_ct_lut_index =
                ctx.lmem_alloc_tensor(reshape_ct_lut_index_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_ct_lut_index);

        uint32_t tl_ct_part1_startAddr = tl_it_ot_ft_part1_startAddr + ceiling_func(3 * hidden_size, NPU_NUM) * tl_xt_mul_w->stride.c;
        cvk_tl_t tl_ct_part1;
        tl_ct_part1.start_address = tl_ct_part1_startAddr ;  // start of lmem
        tl_ct_part1.fmt = CVK_FMT_BF16;
        tl_ct_part1.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_ct_part1.stride = ctx.tl_default_stride(tl_ct_part1.shape, CVK_FMT_BF16, /*eu_align=*/1);


        //Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc
        cvk_tiu_add_param_t p520 = {0};
        p520.res_high = nullptr;
        p520.res_low = tl_ct_lut_index;
        p520.a_high = nullptr;
        p520.a_low = &tl_ct_h_mul_r;
        p520.b_is_const = false;
        p520.b.high = nullptr;
        p520.b.low = &tl_ct_part1;
        p520.rshift_bits = 0;
        p520.layer_id = layer_id;
        p520.relu_enable = 0;
        ctx.tiu_add(&p520);

        //working
        cvk_tl_shape_t reshape_temp6_shape = {
                static_cast<uint32_t>(2), static_cast<uint32_t>(hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_temp6 =
                ctx.lmem_alloc_tensor(reshape_temp6_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_temp6);

        //output
        cvk_tl_shape_t reshape_ct_shape = {
                static_cast<uint32_t>(1), static_cast<uint32_t>(hidden_size),
                static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
        cvk_tl_t *tl_ct =
                ctx.lmem_alloc_tensor(reshape_ct_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_ct);

        //get ct
        cvi_backend_tl_lut(
        ctx, layer_id,
        tl_ct_lut_index->start_address, tl_ct->start_address, tl_temp6->start_address,
        tl_tanh_table_answer->start_address, tl_tanh_table_answer_slope->start_address,
        table_thresh_min, table_thresh_max, 1, hidden_size, 1, 1);

        //Ct = ft (.) Ct-1 + it (.) ct

        //Ct part2 = it (.) Ct-1
        //Dirty tl_c_t_minus_1
        cvk_tiu_mul_param_t ht_part2_mul_param = {0};
        ht_part2_mul_param.res_high = nullptr;
        ht_part2_mul_param.res_low = &tl_c_t_minus_1;
        ht_part2_mul_param.a = &tl_ft;
        ht_part2_mul_param.b = &tl_c_t_minus_1;
        ht_part2_mul_param.b_is_const = 0;
        ht_part2_mul_param.rshift_bits = 0;
        ht_part2_mul_param.layer_id = layer_id;
        ht_part2_mul_param.relu_enable = 0;
        ctx.tiu_mul(&ht_part2_mul_param);

        //Dirty tl_c_t_minus_1 = Ct
        cvk_tiu_mac_param_t p64 = {0};
        p64.res_high = 0;
        p64.res_low = &tl_c_t_minus_1;
        p64.res_is_int8 = 0;
        p64.a = tl_ct;
        p64.b_is_const = 0;
        p64.b = &tl_it;
        p64.lshift_bits = 0; // lshift_bits;
        p64.rshift_bits = 0; // rshift_bits;
        p64.relu_enable = 0;
        p64.layer_id = layer_id;
        ctx.tiu_mac(&p64);

        cvk_tiu_copy_param_t p87 = {0};
        p87.src = &tl_c_t_minus_1;
        p87.dst = tl_ct_lut_index;
        p87.layer_id = layer_id;
        ctx.tiu_copy(&p87);

         //ct_lut_value
        cvk_tl_t *tl_ct_lut_value_index =
                ctx.lmem_alloc_tensor(reshape_ct_shape, CVK_FMT_BF16, 1); //weight EU_ALIGN = False
        ASSERT(tl_ct_lut_value_index);

        cvi_backend_tl_lut(
        ctx, layer_id,
        tl_ct_lut_index->start_address, tl_ct_lut_value_index->start_address, tl_temp6->start_address,
        tl_tanh_table_answer->start_address, tl_tanh_table_answer_slope->start_address,
        table_thresh_min, table_thresh_max, 1, hidden_size, 1, 1);

        cvk_tl_t tl_output_seq;
        tl_output_seq.start_address = tl_output->start_address + i * tl_output->stride.n;  // start of lmem
        tl_output_seq.fmt = CVK_FMT_BF16;
        tl_output_seq.shape = {1, (uint32_t)hidden_size, 1, 1};
        tl_output_seq.stride = ctx.tl_default_stride(tl_output_seq.shape, CVK_FMT_BF16, /*eu_align=*/1);

        //Ht = ot (.) h(Ct)
        cvk_tiu_mul_param_t part1_dot_pro_param = {0};
        part1_dot_pro_param.res_high = nullptr;
        part1_dot_pro_param.res_low = &tl_output_seq;
        part1_dot_pro_param.a = &tl_ot; //rt
        part1_dot_pro_param.b_is_const = 0;
        part1_dot_pro_param.b = tl_ct_lut_value_index;
        part1_dot_pro_param.rshift_bits = 0;
        part1_dot_pro_param.layer_id = layer_id;
        part1_dot_pro_param.relu_enable = 0;
        ctx.tiu_mul(&part1_dot_pro_param);

        if(i == seq_len - 1) {
                //move c_t_minus_1 to output
                cvk_tl_t tl_c_t_minus_1_golden;
                tl_c_t_minus_1_golden.start_address =  tl_output->start_address + seq_len * tl_output->stride.n;  // start of lmem
                tl_c_t_minus_1_golden.fmt = CVK_FMT_BF16;
                tl_c_t_minus_1_golden.shape = {1, (uint32_t)hidden_size, 1, 1};
                tl_c_t_minus_1_golden.stride = ctx.tl_default_stride(tl_c_t_minus_1_golden.shape, CVK_FMT_BF16, /*eu_align=*/1);

                cvk_tiu_copy_param_t p5566 = {0};
                p5566.src = &tl_c_t_minus_1;
                p5566.dst = &tl_c_t_minus_1_golden;
                p5566.layer_id = layer_id;
                ctx.tiu_copy(&p5566);
        }

        //Free memory
        ctx.lmem_free_tensor(tl_ct_lut_value_index);
        // ctx.lmem_free_tensor(tl_ot);
        // ctx.lmem_free_tensor(tl_ot_lut_index);
        ctx.lmem_free_tensor(tl_ct);
        ctx.lmem_free_tensor(tl_temp6);
         ctx.lmem_free_tensor(tl_ct_lut_index);
        ctx.lmem_free_tensor(tl_it_ot_ft);
        ctx.lmem_free_tensor(tl_temp4);
        ctx.lmem_free_tensor(tl_it_ot_ft_lut_index);
        ctx.lmem_free_tensor(tl_h_mul_r);
    }
    //Store output
    ctx.tdma_store_bf16(tl_output, ga_output);
    //free memory
    ctx.lmem_free_tensor(tl_xt_mul_w);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_tanh_table_answer_slope);
    ctx.lmem_free_tensor(tl_tanh_table_answer);
    ctx.lmem_free_tensor(tl_sigmoid_table_answer_slope);
    ctx.lmem_free_tensor(tl_sigmoid_table_answer);
    ctx.lmem_free_tensor(tl_initial_c);
    ctx.lmem_free_tensor(tl_initial_h);
    if(do_bias) {
        ctx.lmem_free_tensor(tl_recurrenceBias);
        ctx.lmem_free_tensor(tl_wtBias);
    }
    ctx.lmem_free_tensor(tl_recurrence);
    ctx.lmem_free_tensor(tl_weight);
    ctx.lmem_free_tensor(tl_input);
 }