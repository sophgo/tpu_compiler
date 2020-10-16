/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_softmax.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "cvi_backend_softmax_kernel"

#define ASSERT(x) assert(x)

unsigned int doSplitHeightBf16softmax2D(const CviBackendContext &ctx, int outerSize, int innerSize) {
    //Default tileN, do not split C/W
    uint8_t eu_align = 1; // hardware constrainst
    int tiledOuterSize = outerSize;
    int bf16_euWorkingOneLane = EU_NUM / 2;
    int parallelC = ceiling_func(innerSize, bf16_euWorkingOneLane);

    cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_BF16);
    int tableSize = ctx.lmem_tensor_to_size(table_shape, CVK_FMT_BF16, eu_align) * 4;

    while(true) {
        if(tiledOuterSize > 4095 - 32) {
            tiledOuterSize--;
            continue;
        }
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(tiledOuterSize,1,1,innerSize);
        int inputSize = ctx.lmem_tensor_to_size(input_shape, CVK_FMT_BF16, eu_align);

        cvk_tl_shape_t enlargeInputShape = ctx.tl_shape_t4(tiledOuterSize,1,1,parallelC * bf16_euWorkingOneLane);
        int enlargeInputSize = ctx.lmem_tensor_to_size(enlargeInputShape, CVK_FMT_BF16, eu_align);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(tiledOuterSize,1,NPU_NUM,1);
        int maxValueSize = ctx.lmem_tensor_to_size(maxValue_shape, CVK_FMT_BF16, eu_align);

        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(tiledOuterSize,parallelC,1,bf16_euWorkingOneLane);
        int parallelInputSize = ctx.lmem_tensor_to_size(parallel_input_shape, CVK_FMT_BF16, eu_align) * 5;
        //parallel_input_shape + lutWorking * 2 + lutResult * 2

        int requiredSize = tableSize + inputSize + enlargeInputSize + parallelInputSize + maxValueSize;
        if(requiredSize < LOCAL_MEM_SIZE) {
            break;
        } else {
            tiledOuterSize--;
        }
    }
    ASSERT(tiledOuterSize && "Can't fit the constraint!");
    return tiledOuterSize;
}

void bf16_softmax_kernel_2d(const CviBackendContext &ctx, uint32_t layer_id,
                            gaddr_t ga_input,
                            gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
                            gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
                            gaddr_t ga_output,
                            int outer_size, int inner_size) {
    unsigned int tiledOutputSize = doSplitHeightBf16softmax2D(ctx, outer_size, inner_size);
    uint8_t eu_align = 1; // hardware constrainst
    int bf16_euWorkingOneLane = EU_NUM / 2;
    int parallelC = ceiling_func(inner_size, bf16_euWorkingOneLane);

    //Load exponential table
    cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_BF16);

    cvk_tl_t *tl_exponential_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_exponential_table_answer_slope =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_exponential_table_answer);
    ASSERT(tl_exponential_table_answer_slope);

    ctx.tdma_load(tl_exponential_table_answer, ga_exponential_table_data_lut);
    ctx.tdma_load(tl_exponential_table_answer_slope, ga_exponential_slope_table_data_lut);
    //Load reciprocal table

    cvk_tl_t *tl_reciprocal_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_reciprocal_mantissa_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_reciprocal_table_answer);
    ASSERT(tl_reciprocal_mantissa_table_answer);

    ctx.tdma_load(tl_reciprocal_table_answer, ga_reciprocal_table_data_lut);
    ctx.tdma_load(tl_reciprocal_mantissa_table_answer, ga_reciprocal_table_mantissa_data_lut);

    int outerSizeStep = ceiling_func(outer_size, tiledOutputSize);
    for(int outerSizeCounter = 0; outerSizeCounter < outerSizeStep; outerSizeCounter++) {
        int outer_pos = outerSizeCounter * tiledOutputSize;
        unsigned int workingOutputSize = std::min(outer_size - outer_pos, (int)tiledOutputSize);

        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(workingOutputSize,1,1,inner_size);
        cvk_tl_t *tl_input =
            ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_input);
        gaddr_t globalSrcAddress = ga_input + outer_pos * inner_size * sizeof(uint16_t);
        ctx.tdma_load(tl_input, globalSrcAddress);

        cvk_tl_t tl_enlargeInput;
        tl_enlargeInput.start_address = tl_input->start_address;  // start of lmem
        tl_enlargeInput.fmt = CVK_FMT_BF16;
        tl_enlargeInput.shape = ctx.tl_shape_t4(workingOutputSize,1,1,parallelC * bf16_euWorkingOneLane);
        tl_enlargeInput.stride = ctx.tl_default_stride(tl_enlargeInput.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(workingOutputSize,1,NPU_NUM,1);
        cvk_tl_t *tl_maxValueBroadcasted =
            ctx.lmem_alloc_tensor(maxValue_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_maxValueBroadcasted);

        cvk_tl_t tl_maxValue;
        tl_maxValue.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
        tl_maxValue.fmt = CVK_FMT_BF16;
        tl_maxValue.shape = {(uint32_t)workingOutputSize, 1, 1, 1};
        tl_maxValue.stride = ctx.tl_default_stride(tl_maxValue.shape, CVK_FMT_BF16, /*eu_align=*/1);

        cvk_tiu_max_pooling_param_t max_pool_param = {0};
        max_pool_param.ofmap = &tl_maxValue;
        max_pool_param.ifmap = tl_input;
        max_pool_param.kh = 1;
        max_pool_param.kw = inner_size;
        max_pool_param.stride_h = 1;
        max_pool_param.stride_w = 1;
        max_pool_param.layer_id = layer_id;
        ctx.tiu_max_pooling(&max_pool_param);

        LLVM_DEBUG(llvm::errs() << llvm::format(
                "  tiu_bf16_max_pooling\n"
                "    ifmap shape (%d, %d, %d, %d)\n"
                "    ofmap shape (%d, %d, %d, %d)\n"
                "    kh %d, kw %d, stride_h %d, stride_w %d\n",
                tl_input->shape.n, tl_input->shape.c, tl_input->shape.h, tl_input->shape.w, tl_maxValue.shape.n,
                tl_maxValue.shape.c, tl_maxValue.shape.h, tl_maxValue.shape.w, 1, inner_size, 1, 1););
        // Broadcast maxValue (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        // (n, 1, NPU_NUM, 1)->(n, NPU_NUM, 1, 1)
        //                 h_str = 0
        {
            // reshape
            cvk_tl_t tl_src;
            tl_src.start_address = tl_maxValue.start_address;  // start of lmem
            tl_src.fmt = CVK_FMT_BF16;
            tl_src.shape = tl_maxValueBroadcasted->shape;
            tl_src.stride = ctx.tl_default_stride(tl_src.shape, CVK_FMT_BF16, /*eu_align=*/1);
            tl_src.stride.h = 0;
            tl_src.stride.n = EU_NUM; //every element = sizeof(BF16), and eu_align  1

            cvk_tl_t tl_dst;
            tl_dst.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
            tl_dst.fmt = CVK_FMT_BF16;
            tl_dst.shape = ctx.tl_shape_t4(workingOutputSize,NPU_NUM,1,1);
            tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, CVK_FMT_BF16, /*eu_align=*/1);

            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = &tl_src;
            p2.dst = &tl_dst;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                            "         L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tdma_l2l_tensor_copy(&p2);
        }
        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(workingOutputSize,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_parallel_input =
            ctx.lmem_alloc_tensor(parallel_input_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_parallel_input);
        //Reshape input(outerSize, 1, 1, innerSize) -> (outerSize, NPU_NUM, 1, innerSize/NPU_NUM)
        {
            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = &tl_enlargeInput;
            p2.dst = tl_parallel_input;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                            "         L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tdma_l2l_tensor_copy(&p2);
        }

        //Input = Input - maxOfInput
        {
            cvk_tl_t tl_reshape_parallel_input;
            tl_reshape_parallel_input.start_address = tl_parallel_input->start_address;  // start of lmem
            tl_reshape_parallel_input.fmt = CVK_FMT_BF16;
            tl_reshape_parallel_input.shape = tl_parallel_input->shape;
            //concate h*w to h and set w = 1. Logcally equal
            tl_reshape_parallel_input.shape.h = tl_parallel_input->shape.h * tl_parallel_input->shape.w;
            tl_reshape_parallel_input.shape.w = 1;
            tl_reshape_parallel_input.stride = ctx.tl_default_stride(tl_reshape_parallel_input.shape, CVK_FMT_BF16, /*eu_align=*/1);

            cvk_tl_t tl_reshape_maxValueBroadcasted;
            tl_reshape_maxValueBroadcasted.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
            tl_reshape_maxValueBroadcasted.fmt = CVK_FMT_BF16;
            tl_reshape_maxValueBroadcasted.shape = tl_reshape_parallel_input.shape;
            tl_reshape_maxValueBroadcasted.stride = ctx.tl_default_stride(tl_reshape_maxValueBroadcasted.shape, CVK_FMT_BF16, /*eu_align=*/1);
            tl_reshape_maxValueBroadcasted.stride.h = 0;//h stride =0
            tl_reshape_maxValueBroadcasted.stride.c = 0;//c stride =0
            tl_reshape_maxValueBroadcasted.stride.n = EU_NUM; //every element = sizeof(BF16)

            cvk_tiu_sub_param_t p5 = {0};
            p5.res_high = 0;
            p5.res_low = &tl_reshape_parallel_input;
            p5.a_high = 0;
            p5.a_low = &tl_reshape_parallel_input;
            p5.b_high = 0;
            p5.b_low = &tl_reshape_maxValueBroadcasted;
            p5.rshift_bits = 0;
            p5.layer_id = layer_id;
            ctx.tiu_sub(&p5);
        }

        cvk_tl_shape_t lut_result_shape = ctx.tl_shape_t4(workingOutputSize,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_lut_result =
            ctx.lmem_alloc_tensor(lut_result_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_lut_result);

        cvk_tl_shape_t lut_working_shape = ctx.tl_shape_t4(workingOutputSize * 2,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_lut_working =
            ctx.lmem_alloc_tensor(lut_working_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_lut_working);
        //lut exponential
        //tl_lut_result = exp(tl_parallel_input)
        {
            const int table_thresh_min = -15;
            const int table_thresh_max = 1;
            cvi_backend_tl_lut(
            ctx, layer_id,
            tl_parallel_input->start_address, tl_lut_result->start_address, tl_lut_working->start_address,
            tl_exponential_table_answer->start_address, tl_exponential_table_answer_slope->start_address,
            table_thresh_min, table_thresh_max, workingOutputSize, parallelC, 1, bf16_euWorkingOneLane);
        }
        //Reshape expValue (outerSize, NPU_NUM, 1, innerSize/NPU_NUM) -> (outerSize, 1, 1, innerSize)
        {
            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = tl_lut_result;
            p2.dst = &tl_enlargeInput;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                            "         L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tdma_l2l_tensor_copy(&p2);
        }

        //Accumulate exponential value
        {
            cvk_tiu_average_pooling_param_t param = {0};
            param.ofmap = &tl_maxValue;
            param.ifmap = tl_input;
            param.kh = 1;
            param.kw = inner_size;
            param.ins_h = 0;
            param.ins_last_h = 0;
            param.ins_w = 0;
            param.ins_last_w = 0;
            param.stride_h = 1;
            param.stride_w = 1;
            //Set this value as inner_size instead of 1  to do accumulate
            //kernel will fill avg_pooling_const / (kh * kw)
            param.avg_pooling_const = ctx.convert_fp32_to_bf16(1.0 * inner_size);
            param.layer_id = layer_id;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                "  tiu_bf16_avg_pooling\n"
                "    ifmap shape (%d, %d, %d, %d)\n"
                "    ofmap shape (%d, %d, %d, %d)\n"
                "    kh %d, kw %d, stride_h %d, stride_w %d\n"
                "    avg_const %f, 0x%x\n",
                tl_input->shape.n, tl_input->shape.c, tl_input->shape.h, tl_input->shape.w, tl_maxValue.shape.n,
                tl_maxValue.shape.c, tl_maxValue.shape.h, tl_maxValue.shape.w, 1, inner_size, 1, 1,
                1.0, param.avg_pooling_const););

            ctx.tiu_average_pooling(&param);
        }

        cvk_tl_t *tl_lut_reciprocal_result =
            ctx.lmem_alloc_tensor(lut_result_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_lut_reciprocal_result);
        //Lut reciprocal value
        {
            cvi_backend_tl_lut_exponential_mul_mantissa(
            ctx, layer_id,
            tl_maxValue.start_address, tl_lut_reciprocal_result->start_address, tl_lut_working->start_address,
            tl_reciprocal_table_answer->start_address, tl_reciprocal_mantissa_table_answer->start_address, workingOutputSize, 1, 1, 1);
        }

        // Broadcast reciprocal value  (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        {
            // reshape
            cvk_tl_t tl_src;
            tl_src.start_address = tl_lut_reciprocal_result->start_address;  // start of lmem
            tl_src.fmt = CVK_FMT_BF16;
            tl_src.shape = tl_maxValueBroadcasted->shape;
            tl_src.stride = ctx.tl_default_stride(tl_src.shape, CVK_FMT_BF16, /*eu_align=*/1);
            tl_src.stride.h = 0;
            tl_src.stride.n = EU_NUM; //every element = sizeof(BF16)

            cvk_tl_t tl_dst;
            tl_dst.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
            tl_dst.fmt = CVK_FMT_BF16;
            tl_dst.shape = ctx.tl_shape_t4(workingOutputSize,NPU_NUM,1,1);
            tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, CVK_FMT_BF16, /*eu_align=*/1);

            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = &tl_src;
            p2.dst = &tl_dst;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                            "         L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tdma_l2l_tensor_copy(&p2);
        }

        //ans = exp(input - maxInput) *  reciprocal value
        {
            cvk_tl_t tl_reshape_maxValueBroadcasted;
            tl_reshape_maxValueBroadcasted.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
            tl_reshape_maxValueBroadcasted.fmt = CVK_FMT_BF16;
            tl_reshape_maxValueBroadcasted.shape = tl_lut_result->shape;
            tl_reshape_maxValueBroadcasted.stride = ctx.tl_default_stride(tl_lut_result->shape, CVK_FMT_BF16, /*eu_align=*/1);
            tl_reshape_maxValueBroadcasted.stride.h = 0;//h stride =0
            tl_reshape_maxValueBroadcasted.stride.c = 0;//c stride =0
            tl_reshape_maxValueBroadcasted.stride.w = 0;//w stride =0
            tl_reshape_maxValueBroadcasted.stride.n = EU_NUM; //every element = sizeof(BF16)

            cvk_tiu_mul_param_t p = {0};
            p.res_high = nullptr;
            p.res_low = tl_lut_result;
            p.a = tl_lut_result;
            p.b = &tl_reshape_maxValueBroadcasted;
            p.b_is_const = 0;
            p.rshift_bits = 0;
            p.layer_id = layer_id;
            p.relu_enable = false;
            ctx.tiu_mul(&p);
        }
        //Store to dram
        {
            cvk_ml_t tl_golden = {0};
            tl_golden.fmt = CVK_FMT_BF16;
            tl_golden.start_address = tl_lut_result->start_address;
            tl_golden.shape = {
                (uint32_t)workingOutputSize,
                (uint32_t)parallelC,
                (uint32_t)bf16_euWorkingOneLane,
                (uint32_t)inner_size}; //n, c, w, col
            tl_golden.stride = ctx.ml_default_stride(tl_golden.shape, tl_golden.fmt, 1);

            cvk_mg_t ts_data = {0};
            ts_data.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_output);
            ts_data.start_address = ga_output + outer_pos * inner_size * sizeof(uint16_t);;
            ts_data.fmt = tl_golden.fmt;
            ts_data.shape = {tl_golden.shape.n, tl_golden.shape.col};
            ts_data.stride = {(uint32_t)(inner_size*sizeof(uint16_t))};

            cvk_tdma_l2g_matrix_copy_param_t p1 = {0};
            p1.src = &tl_golden;
            p1.dst = &ts_data;
            ctx.tdma_l2g_matrix_copy(&p1);
            // ctx.tdma_store_stride(&tl_golden, ga_output,
            //                            {inner_size*sizeof(uint16_t)});// original column width
        }
        ctx.lmem_free_tensor(tl_lut_reciprocal_result);
        ctx.lmem_free_tensor(tl_lut_working);
        ctx.lmem_free_tensor(tl_lut_result);
        ctx.lmem_free_tensor(tl_parallel_input);
        ctx.lmem_free_tensor(tl_maxValueBroadcasted);
        ctx.lmem_free_tensor(tl_input);
    }
    ctx.lmem_free_tensor(tl_reciprocal_mantissa_table_answer);
    ctx.lmem_free_tensor(tl_reciprocal_table_answer);
    ctx.lmem_free_tensor(tl_exponential_table_answer_slope);
    ctx.lmem_free_tensor(tl_exponential_table_answer);
}

unsigned int doSplitHeightBf16softmax4D(const CviBackendContext &ctx, int64_t* shape) {
    //Default tileN, do not split C/W
    uint8_t eu_align = 1; // hardware constrainst
    int c, h, w;
    c = shape[1];
    h = shape[2];
    w = shape[3];
    int tileH = h;

    cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_BF16);
    int tableSize = ctx.lmem_tensor_to_size(table_shape, CVK_FMT_BF16, eu_align) * 4;

    while(true) {
        if(tileH * w > 4095 - 32) {
            tileH--;
            continue;
        }
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(c,tileH * w,1,1);
        int inputSize = ctx.lmem_tensor_to_size(input_shape, CVK_FMT_BF16, eu_align);
        cvk_tl_shape_t input_transposed_shape = ctx.tl_shape_t4(1,tileH * w,1,c);
        int inputTransposedSize = ctx.lmem_tensor_to_size(input_transposed_shape, CVK_FMT_BF16, eu_align) * 5;
        //transposedInput + lutWorking*2 + lutResult * 2

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1,tileH * w,1,1);
        int maxValueSize = ctx.lmem_tensor_to_size(maxValue_shape, CVK_FMT_BF16, eu_align);
        int requiredSize = tableSize + inputSize + inputTransposedSize + maxValueSize;
        if(requiredSize < LOCAL_MEM_SIZE) {
            break;
        } else {
            tileH--;
        }
    }
    ASSERT(tileH && "Can't fit the constraint!");
    return tileH;
}

void bf16_softmax_kernel_4d(const CviBackendContext &ctx, uint32_t layer_id,
                                            gaddr_t ga_input,
                                            gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
                                            gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
                                            gaddr_t ga_output,
                                            int64_t* shape, int axis, int dimension) {
    unsigned int tileH = doSplitHeightBf16softmax4D(ctx, shape);
    uint8_t eu_align = 1; // hardware constrainst
    unsigned int n, c, h, w;
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];

    int hStep = ceiling_func(h, tileH);
    //Load exponential table
    cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_BF16);

    cvk_tl_t *tl_exponential_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_exponential_table_answer_slope =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_exponential_table_answer);
    ASSERT(tl_exponential_table_answer_slope);

    ctx.tdma_load(tl_exponential_table_answer, ga_exponential_table_data_lut);
    ctx.tdma_load(tl_exponential_table_answer_slope, ga_exponential_slope_table_data_lut);
    //Load reciprocal table

    cvk_tl_t *tl_reciprocal_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
    cvk_tl_t *tl_reciprocal_mantissa_table_answer =
        ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

    ASSERT(tl_reciprocal_table_answer);
    ASSERT(tl_reciprocal_mantissa_table_answer);

    ctx.tdma_load(tl_reciprocal_table_answer, ga_reciprocal_table_data_lut);
    ctx.tdma_load(tl_reciprocal_mantissa_table_answer, ga_reciprocal_table_mantissa_data_lut);

    for(int hStepCounter = 0; hStepCounter < hStep; hStepCounter++) {
        int h_pos = hStepCounter * tileH;
        int tileHeightSize = std::min(h - h_pos, tileH);
        //Allocate input
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(c,tileHeightSize * w,1,1);
        cvk_tl_t *tl_input =
            ctx.lmem_alloc_tensor(input_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_input);

        cvk_tl_shape_t input_transposed_shape = ctx.tl_shape_t4(1,tileHeightSize * w,1,c);
        //Allocate transpose input
        cvk_tl_t *tl_transpose_input =
            ctx.lmem_alloc_tensor(input_transposed_shape, CVK_FMT_BF16, eu_align);
        ASSERT(tl_transpose_input);

        for(unsigned int i = 0; i < n; i++) {
            //load input
            gaddr_t inputAddr = ga_input + i * h * w * c * sizeof(uint16_t) + h_pos * w * sizeof(uint16_t);
            cvk_tg_stride_t ifmap_gstride = ctx.tg_default_stride({c, h * w, 1, 1}, CVK_FMT_BF16);
            // original shape
            // ctx.tdma_load(tl_input, inputAddr);
            ctx.tdma_load_stride(tl_input, inputAddr, ifmap_gstride);

            bool doConcateC = tileHeightSize * w > (unsigned int)NPU_NUM;
            if(doConcateC) {
                cvk_tl_t tl_dst;
                tl_dst.start_address = tl_transpose_input->start_address;  // start of lmem
                tl_dst.fmt = tl_input->fmt;
                tl_dst.shape = tl_input->shape;
                int bytesize = tl_input->stride.w;
                int cStride = align_up(c * bytesize, EU_NUM);
                tl_dst.stride = {(uint32_t)bytesize, (uint32_t)cStride, (uint32_t)bytesize, (uint32_t)bytesize};

                cvk_tiu_copy_param_t p2 = {0};
                p2.src = tl_input;
                p2.dst = &tl_dst;
                p2.layer_id = layer_id;

                LLVM_DEBUG(llvm::errs() << llvm::format(
                                "        L2L Reshape:\n"
                                "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                p2.src->start_address, p2.src->shape.n,
                                p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                                p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                                p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                                p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
                ctx.tiu_copy(&p2);
            }
            cvk_tl_t *selected_tl_input = doConcateC ? tl_transpose_input : tl_input;

            cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1,tileHeightSize * w,1,1);
            cvk_tl_t *tl_maxValue =
                ctx.lmem_alloc_tensor(maxValue_shape, CVK_FMT_BF16, eu_align);
            ASSERT(tl_maxValue);

            cvk_tiu_max_pooling_param_t max_pool_param = {0};
            max_pool_param.ofmap = tl_maxValue;
            max_pool_param.ifmap = selected_tl_input;
            max_pool_param.kh = 1;
            max_pool_param.kw = c;
            max_pool_param.stride_h = 1;
            max_pool_param.stride_w = 1;
            max_pool_param.layer_id = layer_id;
            ctx.tiu_max_pooling(&max_pool_param);

            //Input = Input - maxOfInput
            {
                cvk_tl_t tl_reshape_maxValueRefactor;
                tl_reshape_maxValueRefactor.start_address = tl_maxValue->start_address;  // start of lmem
                tl_reshape_maxValueRefactor.fmt = CVK_FMT_BF16;
                tl_reshape_maxValueRefactor.shape = tl_transpose_input->shape;
                tl_reshape_maxValueRefactor.stride = tl_maxValue->stride;
                tl_reshape_maxValueRefactor.stride.w = 0;//w stride =0

                cvk_tiu_sub_param_t p5 = {0};
                p5.res_high = 0;
                p5.res_low = selected_tl_input;
                p5.a_high = 0;
                p5.a_low = selected_tl_input;
                p5.b_high = 0;
                p5.b_low = &tl_reshape_maxValueRefactor;
                p5.rshift_bits = 0;
                p5.layer_id = layer_id;
                ctx.tiu_sub(&p5);
            }
            cvk_tl_shape_t lut_working_shape = ctx.tl_shape_t4(2,tileHeightSize * w,1,c);
            cvk_tl_t *tl_lut_working =
                ctx.lmem_alloc_tensor(lut_working_shape, CVK_FMT_BF16, eu_align);
            ASSERT(tl_lut_working);

            cvk_tl_t *tl_lut_result =
                ctx.lmem_alloc_tensor(input_transposed_shape, CVK_FMT_BF16, eu_align);
            ASSERT(tl_lut_result);
            //lut exponential
            //tl_lut_result = exp(tl_input)
            {
                const int table_thresh_min = -15;
                const int table_thresh_max = 1;
                cvi_backend_tl_lut(
                ctx, layer_id,
                selected_tl_input->start_address, tl_lut_result->start_address, tl_lut_working->start_address,
                tl_exponential_table_answer->start_address, tl_exponential_table_answer_slope->start_address,
                table_thresh_min, table_thresh_max, 1, tileHeightSize * w, 1, c);
            }

            //Accumulate exponential value
            {
                cvk_tiu_average_pooling_param_t param = {0};
                param.ofmap = tl_maxValue;
                param.ifmap = tl_lut_result;
                param.kh = 1;
                param.kw = c;
                param.ins_h = 0;
                param.ins_last_h = 0;
                param.ins_w = 0;
                param.ins_last_w = 0;
                param.stride_h = 1;
                param.stride_w = 1;
                //Set this value as c instead of 1  to do accumulate
                //kernel will fill avg_pooling_const / (kh * kw)
                param.avg_pooling_const = ctx.convert_fp32_to_bf16(1.0 * c);
                param.layer_id = layer_id;

                LLVM_DEBUG(llvm::errs() << llvm::format(
                    "  tiu_bf16_avg_pooling\n"
                    "    ifmap shape (%d, %d, %d, %d)\n"
                    "    ofmap shape (%d, %d, %d, %d)\n"
                    "    kh %d, kw %d, stride_h %d, stride_w %d\n"
                    "    avg_const %f, 0x%x\n",
                    tl_lut_result->shape.n, tl_lut_result->shape.c, tl_lut_result->shape.h, tl_lut_result->shape.w, tl_maxValue->shape.n,
                    tl_maxValue->shape.c, tl_maxValue->shape.h, tl_maxValue->shape.w, 1, c, 1, 1,
                    1.0, param.avg_pooling_const););

                ctx.tiu_average_pooling(&param);
            }

            cvk_tl_t *tl_lut_reciprocal_result =
                ctx.lmem_alloc_tensor(input_transposed_shape, CVK_FMT_BF16, eu_align);
            ASSERT(tl_lut_reciprocal_result);
            //Lut reciprocal value
            {
                cvi_backend_tl_lut_exponential_mul_mantissa(
                ctx, layer_id,
                tl_maxValue->start_address, tl_lut_reciprocal_result->start_address, tl_lut_working->start_address,
                tl_reciprocal_table_answer->start_address, tl_reciprocal_mantissa_table_answer->start_address, 1, tileHeightSize * w, 1, 1);
            }

            //ans = exp(input - maxInput) *  reciprocal value
            {
                cvk_tl_t tl_reshape_reciprocal_value;
                tl_reshape_reciprocal_value.start_address = tl_lut_reciprocal_result->start_address;  // start of lmem
                tl_reshape_reciprocal_value.fmt = CVK_FMT_BF16;
                tl_reshape_reciprocal_value.shape = tl_lut_result->shape;
                tl_reshape_reciprocal_value.stride = ctx.tl_default_stride(tl_reshape_reciprocal_value.shape, CVK_FMT_BF16, /*eu_align=*/1);
                tl_reshape_reciprocal_value.stride.w = 0;//w stride =0

                cvk_tiu_mul_param_t p = {0};
                p.res_high = nullptr;
                p.res_low = tl_lut_result;
                p.a = tl_lut_result;
                p.b = &tl_reshape_reciprocal_value;
                p.b_is_const = 0;
                p.rshift_bits = 0;
                p.layer_id = layer_id;
                p.relu_enable = false;
                ctx.tiu_mul(&p);
            }

            {
                // (1, h*w, 1, c) -> (c, h*w, 1, 1)
                if(doConcateC) {
                    cvk_tl_t tl_dst;
                    tl_dst.start_address = tl_input->start_address;  // start of lmem
                    tl_dst.fmt = tl_lut_result->fmt;
                    tl_dst.shape = tl_lut_result->shape;
                    int bytesize = tl_lut_result->stride.w;
                    tl_dst.stride = {
                        (uint32_t)bytesize,
                        (uint32_t)tl_input->stride.c,
                        (uint32_t)bytesize,
                        (uint32_t)tl_input->stride.n
                    };

                    cvk_tiu_copy_param_t p2 = {0};
                    p2.src = tl_lut_result;
                    p2.dst = &tl_dst;
                    p2.layer_id = layer_id;

                    LLVM_DEBUG(llvm::errs() << llvm::format(
                                    "        L2L Reshape:\n"
                                    "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                    "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                    p2.src->start_address, p2.src->shape.n,
                                    p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                                    p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                                    p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                                    p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
                    ctx.tiu_copy(&p2);
                }

                cvk_tl_t *selected_tl_output = doConcateC ? tl_input : tl_lut_result;
                //store
                // gaddr_t outputAddr = ga_output + i * h * w * c * sizeof(uint16_t);
                gaddr_t outputAddr = ga_output + i * h * w * c * sizeof(uint16_t) + h_pos * w * sizeof(uint16_t);
                cvk_tg_stride_t ofmap_gstride = ctx.tg_default_stride({c, h * w, 1, 1}, CVK_FMT_BF16);
                // // original shape
                ctx.tdma_store_stride(selected_tl_output, outputAddr, ofmap_gstride);
                // ctx.tdma_store(selected_tl_output, outputAddr);
            }
            ctx.lmem_free_tensor(tl_lut_reciprocal_result);
            ctx.lmem_free_tensor(tl_lut_result);
            ctx.lmem_free_tensor(tl_lut_working);
            ctx.lmem_free_tensor(tl_maxValue);
        }
        ctx.lmem_free_tensor(tl_transpose_input);
        ctx.lmem_free_tensor(tl_input);
    }
    ctx.lmem_free_tensor(tl_reciprocal_mantissa_table_answer);
    ctx.lmem_free_tensor(tl_reciprocal_table_answer);
    ctx.lmem_free_tensor(tl_exponential_table_answer_slope);
    ctx.lmem_free_tensor(tl_exponential_table_answer);
}

void cvi_backend_tg_bf16_softmax_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                        gaddr_t ga_input,
                                        gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
                                        gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
                                        gaddr_t ga_output,
                                        int64_t* shape, int axis, int dimension) {
    int outer_size, inner_size;
    bool doTranspose = false;
    if (dimension == 2) {
        outer_size = shape[0];
        inner_size = shape[1];
    } else if (dimension == 4) {
        assert(axis == 1 && "Support only axis = 1 (Align c)");
        if(shape[2] * shape[3] == 1) {
            outer_size = shape[0]; //n
            inner_size = shape[1]; //c
        } else {
            // batchStep = shape[0];
            outer_size = shape[0] * shape[2] * shape[3]; //n
            inner_size = shape[1]; //c
            doTranspose = true;
        }
    } else if (dimension == 3) {
        assert(axis == 2 && "Support only axis = 2");
        outer_size = shape[0] * shape[1]; //c * h
        inner_size = shape[2]; //w
    }
    if(doTranspose) {
        bf16_softmax_kernel_4d(ctx, layer_id,
                               ga_input,
                               ga_exponential_table_data_lut, ga_exponential_slope_table_data_lut,
                               ga_reciprocal_table_data_lut, ga_reciprocal_table_mantissa_data_lut,
                               ga_output,
                               shape, axis, dimension);
    } else {
        bf16_softmax_kernel_2d(ctx, layer_id,
                               ga_input,
                               ga_exponential_table_data_lut, ga_exponential_slope_table_data_lut,
                               ga_reciprocal_table_data_lut, ga_reciprocal_table_mantissa_data_lut,
                               ga_output,
                               outer_size, inner_size);
    }
}