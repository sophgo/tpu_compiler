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
#include "TgBf16SoftmaxKernel.hpp"

#define DEBUG_TYPE "cvi_backend_softmax_kernel"

#define ASSERT(x) assert(x)

void TgSoftmaxKernel::matrixToTensor(cvk_tl_t *tensor, const cvk_ml_t &matrix) {
  cvk_tl_shape_t shape = {matrix.shape.n, matrix.shape.c, 1, matrix.shape.w};
  ctx.lmem_init_tensor(tensor, shape, fmt, 1);
  tensor->start_address = matrix.start_address;
}

unsigned int TgSoftmaxKernel::doSplitHeightBf16softmax2DParallelInnerSize() {
    //Default tileN, do not split C/W
    uint8_t eu_align = 1; // hardware constrainst
    int tiledOuterSize = outer_size;
    int bf16_euWorkingOneLane = ctx.tiu_eu_num(fmt);
    int parallelC = ceiling_func(inner_size, bf16_euWorkingOneLane);

    int tableSize = ctx.lmem_tensor_to_size(table_shape, fmt, eu_align) * 4;

    while(true) {
        if(tiledOuterSize > 4095 - 32) {
            tiledOuterSize--;
            continue;
        }

        cvk_tl_shape_t enlargeInputShape = ctx.tl_shape_t4(tiledOuterSize,1,1,parallelC * bf16_euWorkingOneLane);
        int enlargeInputSize = ctx.lmem_tensor_to_size(enlargeInputShape, fmt, eu_align);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(tiledOuterSize,1,NPU_NUM,1);
        int maxValueSize = ctx.lmem_tensor_to_size(maxValue_shape, fmt, eu_align);

        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(tiledOuterSize,parallelC,1,bf16_euWorkingOneLane);
        int parallelInputSize = ctx.lmem_tensor_to_size(parallel_input_shape, fmt, eu_align) * 5;
        //parallel_input_shape + lutWorking * 2 + lutResult * 2

        int requiredSize = tableSize + enlargeInputSize + parallelInputSize + maxValueSize;
        LLVM_DEBUG(llvm::dbgs() << llvm::format(
                            "        Size:\n"
                            "         tableSize 0x%lx, enlargeInputSize 0x%lx, maxValue_shape 0x%lx parallel_input_shape 0x%lx,\n", tableSize
                            , enlargeInputSize, maxValueSize, parallelInputSize
                           ));
        if(requiredSize < LOCAL_MEM_SIZE) {
            break;
        } else {
            tiledOuterSize--;
        }
    }
    return tiledOuterSize;
}

unsigned int TgSoftmaxKernel::doSplitHeightBf16softmax2DParallelOuterSize() {
    //Default tileN, do not split C/W
    uint8_t eu_align = 1; // hardware constrainst
    int tiledOuterSize = outer_size;

    int tableSize = ctx.lmem_tensor_to_size(table_shape, fmt, eu_align) * 4;

    while(true) {
        if(tiledOuterSize > 4095 - 32) {
            tiledOuterSize--;
            continue;
        }
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(1 , tiledOuterSize, 1, inner_size);
        int inputSize = ctx.lmem_tensor_to_size(input_shape, fmt, eu_align);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1, tiledOuterSize, 1, 1);
        int maxValueSize = ctx.lmem_tensor_to_size(maxValue_shape, fmt, eu_align);

        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(1 , tiledOuterSize, 1, inner_size);
        int parallelInputSize = ctx.lmem_tensor_to_size(parallel_input_shape, fmt, eu_align) * 5;
        //parallel_input_shape + lutWorking * 2 + lutResult * 2

        int requiredSize = tableSize + inputSize + parallelInputSize + maxValueSize;
        LLVM_DEBUG(llvm::dbgs() << llvm::format(
                            "        Size:\n"
                            "         tableSize 0x%lx, inputSize 0x%lx, maxValueSize 0x%lx parallel_input_shape 0x%lx,\n", tableSize
                            , inputSize, maxValueSize, parallelInputSize
                           ));
        if(requiredSize < LOCAL_MEM_SIZE) {
            break;
        } else {
            tiledOuterSize--;
        }
    }
    return tiledOuterSize;
}

void TgSoftmaxKernel::softmaxLargeSizeHandler() {
    const unsigned int tiledOutputSize = 1;
    uint8_t eu_align = 1; // hardware constrainst
    int sizePerLane = ceiling_func(inner_size, NPU_NUM);

    int outerSizeStep = ceiling_func(outer_size, tiledOutputSize);
    for(int outerSizeCounter = 0; outerSizeCounter < outerSizeStep; outerSizeCounter++) {
        int outer_pos = outerSizeCounter * tiledOutputSize;
        unsigned int workingOutputSize = std::min(outer_size - outer_pos, (int)tiledOutputSize);

        cvk_ml_shape_t input_shape = {
                (uint32_t)workingOutputSize,
                (uint32_t)NPU_NUM,
                (uint32_t)ceiling_func(inner_size, NPU_NUM),
                (uint32_t)inner_size}; //n, c, w, col
        cvk_ml_t *ml_input =
            ctx.lmem_alloc_matrix(input_shape, fmt, eu_align);
        ASSERT(ml_input);

        //init to zero
        cvk_tl_t tl_input;
        matrixToTensor(&tl_input, *ml_input);
        ctx.tiu_zeros(layer_id, &tl_input);

        //load
        gaddr_t globalSrcAddress = ga_input + outer_pos * inner_size * sizeof(uint16_t);
        ctx.tdma_load(ml_input, globalSrcAddress);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1,tl_input.shape.c, 1, tl_input.shape.c);
        cvk_tl_t *tl_maxValueBroadcasted =
            ctx.lmem_alloc_tensor(maxValue_shape, fmt, eu_align);
        ASSERT(tl_maxValueBroadcasted);

        cvk_tl_t tl_perCMaxValue;
        tl_perCMaxValue.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
        tl_perCMaxValue.fmt = fmt;
        tl_perCMaxValue.shape = {1, tl_input.shape.c, 1, 1};
        tl_perCMaxValue.stride = ctx.tl_default_stride(tl_perCMaxValue.shape, fmt, /*eu_align=*/1);

        cvk_tl_t tl_concatMaxValue;
        tl_concatMaxValue.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
        tl_concatMaxValue.fmt = fmt;
        tl_concatMaxValue.shape = {1, 1, tl_input.shape.c, 1};
        tl_concatMaxValue.stride = ctx.tl_default_stride(tl_concatMaxValue.shape, fmt, /*eu_align=*/1);

        cvk_tl_t tl_maxValue;
        tl_maxValue.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
        tl_maxValue.fmt = fmt;
        tl_maxValue.shape = {1, 1, 1, 1};
        tl_maxValue.stride = ctx.tl_default_stride(tl_maxValue.shape, fmt, /*eu_align=*/1);

        // Calculate per lane max value
        max_per_lane_value(&tl_input, &tl_perCMaxValue);

        // Concate per lane max value
        cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
        p2.src = &tl_perCMaxValue;
        p2.dst = &tl_concatMaxValue;

        LLVM_DEBUG(llvm::dbgs() << llvm::format(
                        "         L2L Reshape:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                        p2.src->start_address, p2.src->shape.n,
                        p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                        p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                        p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                        p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
        ctx.tdma_l2l_tensor_copy(&p2);

        // Get max value
        max_per_lane_value(&tl_concatMaxValue, &tl_maxValue);

        // Broadcast maxValue (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        // (n, 1, NPU_NUM, 1)->(n, NPU_NUM, 1, 1)
        //                 h_str = 0
        broadcast_one_data_to_all_lane(&tl_maxValue, tl_maxValueBroadcasted);

        //Input = Input - maxOfInput
        every_input_operate_one_specific_data(&tl_input, tl_maxValueBroadcasted, Sub, false);

        cvk_tl_shape_t lut_result_shape = tl_input.shape;
        cvk_tl_t *tl_lut_result =
            ctx.lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
        ASSERT(tl_lut_result);

        cvk_tl_shape_t lut_working_shape = tl_input.shape;
        lut_working_shape.n *= 2; // Allocate twice of input as working space
        cvk_tl_t *tl_lut_working =
            ctx.lmem_alloc_tensor(lut_working_shape, fmt, eu_align);
        ASSERT(tl_lut_working);
        //lut exponential
        //tl_lut_result = exp(tl_parallel_input)
        exponential(&tl_input, tl_lut_result, tl_lut_working);

        //Accumulate exponential value
        {
            //Calculate per lane exponential value
            accumulate_per_lane_value(tl_lut_result, &tl_perCMaxValue);

            // Concate per lane accumulator value
            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = &tl_perCMaxValue;
            p2.dst = &tl_concatMaxValue;

            LLVM_DEBUG(llvm::dbgs() << llvm::format(
                            "         L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tdma_l2l_tensor_copy(&p2);

            //Accumulate per lane exponential value
            accumulate_per_lane_value(&tl_concatMaxValue, &tl_maxValue);
        }

        cvk_tl_t *tl_lut_reciprocal_result =
            ctx.lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
        ASSERT(tl_lut_reciprocal_result);
        //Lut reciprocal value
        reciprocal(&tl_maxValue, tl_lut_reciprocal_result, tl_lut_working);
        // Broadcast reciprocal value  (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        broadcast_one_data_to_all_lane(tl_lut_reciprocal_result, tl_maxValueBroadcasted);

        //ans = exp(input - maxInput) *  reciprocal value
        every_input_operate_one_specific_data(tl_lut_result, tl_maxValueBroadcasted, Mul, false);

        //Store to dram
        {
            cvk_ml_t tl_golden = {0};
            tl_golden.fmt = fmt;
            tl_golden.start_address = tl_lut_result->start_address;
            tl_golden.shape = {
                (uint32_t)workingOutputSize,
                (uint32_t)NPU_NUM,
                (uint32_t)sizePerLane,
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
        ctx.lmem_free_tensor(tl_maxValueBroadcasted);
        ctx.lmem_free_matrix(ml_input);
    }
}

void TgSoftmaxKernel::bf16_softmax_kernel_2d_parallel_inner_size() {
    unsigned int tiledOutputSize = doSplitHeightBf16softmax2DParallelInnerSize();
    uint8_t eu_align = 1; // hardware constrainst
    int bf16_euWorkingOneLane = ctx.tiu_eu_num(fmt);
    int parallelC = ceiling_func(inner_size, bf16_euWorkingOneLane);
    bool isInnerSizeBiggerHWConstraint = inner_size > MAX_WIDTH;
    int outerSizeStep = ceiling_func(outer_size, tiledOutputSize);

    for(int outerSizeCounter = 0; outerSizeCounter < outerSizeStep; outerSizeCounter++) {
        int outer_pos = outerSizeCounter * tiledOutputSize;
        unsigned int workingOutputSize = std::min(outer_size - outer_pos, (int)tiledOutputSize);

        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(workingOutputSize,1,1,inner_size);
        cvk_tl_t *tl_input =
            ctx.lmem_alloc_tensor(input_shape, fmt, eu_align);
        ASSERT(tl_input);
        gaddr_t globalSrcAddress = ga_input + outer_pos * inner_size * sizeof(uint16_t);
        ctx.tdma_load(tl_input, globalSrcAddress);

        cvk_tl_t tl_enlargeInput = {};
        tl_enlargeInput.start_address = tl_input->start_address;  // start of lmem
        tl_enlargeInput.fmt = fmt;
        tl_enlargeInput.shape = ctx.tl_shape_t4(workingOutputSize, 1, 1, parallelC * bf16_euWorkingOneLane);
        tl_enlargeInput.stride = ctx.tl_default_stride(tl_enlargeInput.shape, fmt, /*eu_align=*/1);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(workingOutputSize,1,NPU_NUM,1);
        cvk_tl_t *tl_maxValueBroadcasted =
            ctx.lmem_alloc_tensor(maxValue_shape, fmt, eu_align);
        ASSERT(tl_maxValueBroadcasted);

        cvk_tl_t tl_maxValue;
        tl_maxValue.start_address = tl_maxValueBroadcasted->start_address;  // start of lmem
        tl_maxValue.fmt = fmt;
        tl_maxValue.shape = {(uint32_t)workingOutputSize, 1, 1, 1};
        tl_maxValue.stride = ctx.tl_default_stride(tl_maxValue.shape, fmt, /*eu_align=*/1);

        if(isInnerSizeBiggerHWConstraint) {
            cvk_tl_shape_t maxValueTemp_shape = ctx.tl_shape_t4(workingOutputSize,1,1,EU_NUM);
            cvk_tl_t *tl_maxValueBroadcastedTemp =
                ctx.lmem_alloc_tensor(maxValueTemp_shape, fmt, eu_align);
            ASSERT(tl_maxValueBroadcastedTemp);

            cvk_tl_t tl_currentInput;
            tl_currentInput = *tl_input;

            const int stepSize = MAX_WIDTH / 16 * 16; //Align 16B
            int innerSizeStep = ceiling_func(inner_size, stepSize);
            for(int innerStepTimes = 0; innerStepTimes < innerSizeStep; innerStepTimes++) {
                int inner_pos = innerStepTimes * stepSize;
                unsigned int workingInnerSize = std::min(inner_size - inner_pos, (int)stepSize);
                tl_currentInput.start_address = tl_input->start_address + inner_pos;
                tl_currentInput.shape = ctx.tl_shape_t4(workingOutputSize, 1, 1, workingInnerSize);

                max_per_lane_value(&tl_currentInput, &tl_maxValue);

                cvk_tl_t tl_maxValuePos;
                tl_maxValuePos.start_address = tl_maxValueBroadcastedTemp->start_address + innerStepTimes * sizeof(uint16_t);  // start of lmem
                tl_maxValuePos.fmt = fmt;
                tl_maxValuePos.shape = ctx.tl_shape_t4(workingOutputSize,1,1,1);
                tl_maxValuePos.stride = ctx.tl_default_stride(tl_maxValuePos.shape, fmt, /*eu_align=*/1);

                cvk_tiu_copy_param_t p_copy_max = {0};
                p_copy_max.src = &tl_maxValue;
                p_copy_max.dst = &tl_maxValuePos;
                p_copy_max.layer_id = layer_id;

                LLVM_DEBUG(llvm::dbgs() << llvm::format(
                                "        L2L Reshape:\n"
                                "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                p_copy_max.src->start_address, p_copy_max.src->shape.n,
                                p_copy_max.src->shape.c, p_copy_max.src->shape.h, p_copy_max.src->shape.w, p_copy_max.src->stride.n,
                                p_copy_max.src->stride.c, p_copy_max.src->stride.h, p_copy_max.src->stride.w, p_copy_max.dst->start_address,
                                p_copy_max.dst->shape.n, p_copy_max.dst->shape.c, p_copy_max.dst->shape.h, p_copy_max.dst->shape.w,
                                p_copy_max.dst->stride.n, p_copy_max.dst->stride.c, p_copy_max.dst->stride.h, p_copy_max.dst->stride.w));
                ctx.tiu_copy(&p_copy_max);
            }
            cvk_tl_t tl_maxValueTemp;
            tl_maxValueTemp.start_address = tl_maxValueBroadcastedTemp->start_address;  // start of lmem
            tl_maxValueTemp.fmt = fmt;
            tl_maxValueTemp.shape = ctx.tl_shape_t4(workingOutputSize,1,1,innerSizeStep);
            tl_maxValueTemp.stride = tl_maxValueBroadcastedTemp->stride;

            max_per_lane_value(&tl_maxValueTemp, &tl_maxValue);
            ctx.lmem_free_tensor(tl_maxValueBroadcastedTemp);
        } else {
            max_per_lane_value(tl_input, &tl_maxValue);
        }
        // Broadcast maxValue (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        // (n, 1, NPU_NUM, 1)->(n, NPU_NUM, 1, 1)
        //                 h_str = 0
        broadcast_one_data_to_all_lane(&tl_maxValue, tl_maxValueBroadcasted);

        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(workingOutputSize,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_parallel_input =
            ctx.lmem_alloc_tensor(parallel_input_shape, fmt, eu_align);
        ASSERT(tl_parallel_input);
        //Reshape input(outerSize, 1, 1, innerSize) -> (outerSize, NPU_NUM, 1, innerSize/NPU_NUM)
        {
            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = &tl_enlargeInput;
            p2.dst = tl_parallel_input;

            LLVM_DEBUG(llvm::dbgs() << llvm::format(
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
        every_input_operate_one_specific_data(tl_parallel_input, tl_maxValueBroadcasted, Sub, false);

        cvk_tl_shape_t lut_result_shape = ctx.tl_shape_t4(workingOutputSize,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_lut_result =
            ctx.lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
        ASSERT(tl_lut_result);

        cvk_tl_shape_t lut_working_shape = ctx.tl_shape_t4(workingOutputSize * 2,parallelC,1,bf16_euWorkingOneLane);
        cvk_tl_t *tl_lut_working =
            ctx.lmem_alloc_tensor(lut_working_shape, fmt, eu_align);
        ASSERT(tl_lut_working);
        //lut exponential
        //tl_lut_result = exp(tl_parallel_input)
        exponential(tl_parallel_input, tl_lut_result, tl_lut_working);

        //Reshape expValue (outerSize, NPU_NUM, 1, innerSize/NPU_NUM) -> (outerSize, 1, 1, innerSize)
        {
            cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
            p2.src = tl_lut_result;
            p2.dst = &tl_enlargeInput;

            LLVM_DEBUG(llvm::dbgs() << llvm::format(
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
            if(isInnerSizeBiggerHWConstraint) {
                cvk_tl_shape_t accValueTemp_shape = ctx.tl_shape_t4(workingOutputSize,1,1,EU_NUM);
                cvk_tl_t *tl_accValue =
                    ctx.lmem_alloc_tensor(accValueTemp_shape, fmt, eu_align);
                ASSERT(tl_accValue);

                cvk_tl_t tl_currentInput;
                tl_currentInput = *tl_input;

                const int stepSize = MAX_WIDTH / 16 * 16; //Align 16B
                int innerSizeStep = ceiling_func(inner_size, stepSize);
                for(int innerStepTimes = 0; innerStepTimes < innerSizeStep; innerStepTimes++) {
                    int inner_pos = innerStepTimes * stepSize;
                    unsigned int workingInnerSize = std::min(inner_size - inner_pos, (int)stepSize);
                    tl_currentInput.start_address = tl_input->start_address + inner_pos;
                    tl_currentInput.shape = ctx.tl_shape_t4(workingOutputSize, 1, 1, workingInnerSize);


                    accumulate_per_lane_value(&tl_currentInput, &tl_maxValue);

                    cvk_tl_t tl_accValuePos;
                    tl_accValuePos.start_address = tl_accValue->start_address + innerStepTimes * sizeof(uint16_t);  // start of lmem
                    tl_accValuePos.fmt = fmt;
                    tl_accValuePos.shape = ctx.tl_shape_t4(workingOutputSize,1,1,1);
                    tl_accValuePos.stride = ctx.tl_default_stride(tl_accValuePos.shape, fmt, /*eu_align=*/1);

                    cvk_tiu_copy_param_t p_copy_acc = {0};
                    p_copy_acc.src = &tl_maxValue;
                    p_copy_acc.dst = &tl_accValuePos;
                    p_copy_acc.layer_id = layer_id;

                    LLVM_DEBUG(llvm::dbgs() << llvm::format(
                                    "        L2L Reshape:\n"
                                    "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                    "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                    p_copy_acc.src->start_address, p_copy_acc.src->shape.n,
                                    p_copy_acc.src->shape.c, p_copy_acc.src->shape.h, p_copy_acc.src->shape.w, p_copy_acc.src->stride.n,
                                    p_copy_acc.src->stride.c, p_copy_acc.src->stride.h, p_copy_acc.src->stride.w, p_copy_acc.dst->start_address,
                                    p_copy_acc.dst->shape.n, p_copy_acc.dst->shape.c, p_copy_acc.dst->shape.h, p_copy_acc.dst->shape.w,
                                    p_copy_acc.dst->stride.n, p_copy_acc.dst->stride.c, p_copy_acc.dst->stride.h, p_copy_acc.dst->stride.w));
                    ctx.tiu_copy(&p_copy_acc);
                }
                cvk_tl_t tl_accValueTemp;
                tl_accValueTemp.start_address = tl_accValue->start_address;  // start of lmem
                tl_accValueTemp.fmt = fmt;
                tl_accValueTemp.shape = ctx.tl_shape_t4(workingOutputSize,1,1,innerSizeStep);
                tl_accValueTemp.stride = ctx.tl_default_stride(tl_accValueTemp.shape, fmt, /*eu_align=*/1);

                accumulate_per_lane_value(&tl_accValueTemp, &tl_maxValue);
                ctx.lmem_free_tensor(tl_accValue);
            } else {
                accumulate_per_lane_value(tl_input, &tl_maxValue);
            }
        }

        cvk_tl_t *tl_lut_reciprocal_result =
            ctx.lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
        ASSERT(tl_lut_reciprocal_result);
        //Lut reciprocal value
        reciprocal(&tl_maxValue, tl_lut_reciprocal_result, tl_lut_working);

        // Broadcast reciprocal value  (n, 1, 1, 1) -> (n, NPU_NUM, 1, 1)
        broadcast_one_data_to_all_lane(tl_lut_reciprocal_result, tl_maxValueBroadcasted);

        //ans = exp(input - maxInput) *  reciprocal value
        every_input_operate_one_specific_data(tl_lut_result, tl_maxValueBroadcasted, Mul, false);
        //Store to dram
        {
            cvk_ml_t tl_golden = {0};
            tl_golden.fmt = fmt;
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
}

void TgSoftmaxKernel::bf16_softmax_kernel_2d_parallel_outer_size() {
    unsigned int tiledOutputSize = doSplitHeightBf16softmax2DParallelOuterSize();
    uint8_t eu_align = 1; // hardware constrainst
    bool isInnerSizeBiggerHWConstraint = inner_size > MAX_WIDTH;
    int outerSizeStep = ceiling_func(outer_size, tiledOutputSize);

    for(int outerSizeCounter = 0; outerSizeCounter < outerSizeStep; outerSizeCounter++) {
        int outer_pos = outerSizeCounter * tiledOutputSize;
        unsigned int workingOutputSize = std::min(outer_size - outer_pos, (int)tiledOutputSize);

        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, inner_size);
        cvk_tl_t *tl_input =
            ctx.lmem_alloc_tensor(input_shape, fmt, eu_align);
        ASSERT(tl_input);
        gaddr_t globalSrcAddress = ga_input + outer_pos * inner_size * sizeof(uint16_t);
        ctx.tdma_load(tl_input, globalSrcAddress);

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, 1);
        cvk_tl_t *tl_maxValue =
            ctx.lmem_alloc_tensor(maxValue_shape, fmt, eu_align);
        ASSERT(tl_maxValue);

        if(isInnerSizeBiggerHWConstraint) {
            cvk_tl_shape_t maxValueTemp_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, EU_NUM);
            cvk_tl_t *tl_maxValueBroadcastedTemp =
                ctx.lmem_alloc_tensor(maxValueTemp_shape, fmt, eu_align);
            ASSERT(tl_maxValueBroadcastedTemp);

            cvk_tl_t tl_currentInput;
            tl_currentInput = *tl_input;

            const int stepSize = MAX_WIDTH / 16 * 16; //Align 16B
            int innerSizeStep = ceiling_func(inner_size, stepSize);
            for(int innerStepTimes = 0; innerStepTimes < innerSizeStep; innerStepTimes++) {
                int inner_pos = innerStepTimes * stepSize;
                unsigned int workingInnerSize = std::min(inner_size - inner_pos, (int)stepSize);
                tl_currentInput.start_address = tl_input->start_address + inner_pos;
                tl_currentInput.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, workingInnerSize);

                max_per_lane_value(&tl_currentInput, tl_maxValue);

                cvk_tl_t tl_maxValuePos;
                tl_maxValuePos.start_address = tl_maxValueBroadcastedTemp->start_address + innerStepTimes * sizeof(uint16_t);  // start of lmem
                tl_maxValuePos.fmt = fmt;
                tl_maxValuePos.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, 1);
                tl_maxValuePos.stride = ctx.tl_default_stride(tl_maxValuePos.shape, fmt, /*eu_align=*/1);

                cvk_tiu_copy_param_t p_copy_max = {0};
                p_copy_max.src = tl_maxValue;
                p_copy_max.dst = &tl_maxValuePos;
                p_copy_max.layer_id = layer_id;

                LLVM_DEBUG(llvm::dbgs() << llvm::format(
                                "        L2L Reshape:\n"
                                "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                p_copy_max.src->start_address, p_copy_max.src->shape.n,
                                p_copy_max.src->shape.c, p_copy_max.src->shape.h, p_copy_max.src->shape.w, p_copy_max.src->stride.n,
                                p_copy_max.src->stride.c, p_copy_max.src->stride.h, p_copy_max.src->stride.w, p_copy_max.dst->start_address,
                                p_copy_max.dst->shape.n, p_copy_max.dst->shape.c, p_copy_max.dst->shape.h, p_copy_max.dst->shape.w,
                                p_copy_max.dst->stride.n, p_copy_max.dst->stride.c, p_copy_max.dst->stride.h, p_copy_max.dst->stride.w));
                ctx.tiu_copy(&p_copy_max);
            }
            cvk_tl_t tl_maxValueTemp;
            tl_maxValueTemp.start_address = tl_maxValueBroadcastedTemp->start_address;  // start of lmem
            tl_maxValueTemp.fmt = fmt;
            tl_maxValueTemp.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, innerSizeStep);
            tl_maxValueTemp.stride = tl_maxValueBroadcastedTemp->stride;


            max_per_lane_value(&tl_maxValueTemp, tl_maxValue);
            ctx.lmem_free_tensor(tl_maxValueBroadcastedTemp);
        } else {
            max_per_lane_value(tl_input, tl_maxValue);
        }
        cvk_tl_shape_t parallel_input_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, inner_size);
        cvk_tl_t *tl_parallel_input =
            ctx.lmem_alloc_tensor(parallel_input_shape, fmt, eu_align);
        ASSERT(tl_parallel_input);

        //Input = Input - maxOfInput
        every_input_operate_one_specific_data(tl_input, tl_maxValue, Sub, true);

        cvk_tl_shape_t lut_result_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, inner_size);
        cvk_tl_t *tl_lut_result =
            ctx.lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
        ASSERT(tl_lut_result);

        cvk_tl_shape_t lut_working_shape = ctx.tl_shape_t4(1, workingOutputSize * 2, 1, inner_size);
        cvk_tl_t *tl_lut_working =
            ctx.lmem_alloc_tensor(lut_working_shape, fmt, eu_align);
        ASSERT(tl_lut_working);
        //lut exponential
        //tl_lut_result = exp(tl_parallel_input)
        exponential(tl_input, tl_lut_result, tl_lut_working);
        {
            if(isInnerSizeBiggerHWConstraint) {
                cvk_tl_shape_t accValueTemp_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, EU_NUM);
                cvk_tl_t *tl_accValue =
                    ctx.lmem_alloc_tensor(accValueTemp_shape, fmt, eu_align);
                ASSERT(tl_accValue);

                cvk_tl_t tl_currentInput;
                tl_currentInput = *tl_input;

                const int stepSize = MAX_WIDTH / 16 * 16; //Align 16B
                int innerSizeStep = ceiling_func(inner_size, stepSize);
                for(int innerStepTimes = 0; innerStepTimes < innerSizeStep; innerStepTimes++) {
                    int inner_pos = innerStepTimes * stepSize;
                    unsigned int workingInnerSize = std::min(inner_size - inner_pos, (int)stepSize);
                    tl_currentInput.start_address = tl_input->start_address + inner_pos;
                    tl_currentInput.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, workingInnerSize);

                    accumulate_per_lane_value(&tl_currentInput, tl_maxValue);

                    cvk_tl_t tl_accValuePos;
                    tl_accValuePos.start_address = tl_accValue->start_address + innerStepTimes * sizeof(uint16_t);  // start of lmem
                    tl_accValuePos.fmt = fmt;
                    tl_accValuePos.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, 1);
                    tl_accValuePos.stride = ctx.tl_default_stride(tl_accValuePos.shape, fmt, /*eu_align=*/1);

                    cvk_tiu_copy_param_t p_copy_acc = {0};
                    p_copy_acc.src = tl_maxValue;
                    p_copy_acc.dst = &tl_accValuePos;
                    p_copy_acc.layer_id = layer_id;

                    LLVM_DEBUG(llvm::dbgs() << llvm::format(
                                    "        L2L Reshape:\n"
                                    "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                    "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                    p_copy_acc.src->start_address, p_copy_acc.src->shape.n,
                                    p_copy_acc.src->shape.c, p_copy_acc.src->shape.h, p_copy_acc.src->shape.w, p_copy_acc.src->stride.n,
                                    p_copy_acc.src->stride.c, p_copy_acc.src->stride.h, p_copy_acc.src->stride.w, p_copy_acc.dst->start_address,
                                    p_copy_acc.dst->shape.n, p_copy_acc.dst->shape.c, p_copy_acc.dst->shape.h, p_copy_acc.dst->shape.w,
                                    p_copy_acc.dst->stride.n, p_copy_acc.dst->stride.c, p_copy_acc.dst->stride.h, p_copy_acc.dst->stride.w));
                    ctx.tiu_copy(&p_copy_acc);
                }
                cvk_tl_t tl_accValueTemp;
                tl_accValueTemp.start_address = tl_accValue->start_address;  // start of lmem
                tl_accValueTemp.fmt = fmt;
                tl_accValueTemp.shape = ctx.tl_shape_t4(1, workingOutputSize, 1, innerSizeStep);
                tl_accValueTemp.stride = ctx.tl_default_stride(tl_accValueTemp.shape, fmt, /*eu_align=*/1);

                accumulate_per_lane_value(&tl_accValueTemp, tl_maxValue);

                ctx.lmem_free_tensor(tl_accValue);
            } else {
                accumulate_per_lane_value(tl_lut_result, tl_maxValue);
            }
        }

        cvk_tl_shape_t lut_reciprocal_result_shape = ctx.tl_shape_t4(1, workingOutputSize, 1, 1);
        cvk_tl_t *tl_lut_reciprocal_result =
            ctx.lmem_alloc_tensor(lut_reciprocal_result_shape, fmt, eu_align);
        ASSERT(tl_lut_reciprocal_result);
        //Lut reciprocal value
        reciprocal(tl_maxValue, tl_lut_reciprocal_result, tl_lut_working);

        //ans = exp(input - maxInput) *  reciprocal value
        every_input_operate_one_specific_data(tl_lut_result, tl_lut_reciprocal_result, Mul, true);
        //Store to dram
        {
            //store
            // gaddr_t outputAddr = ga_output + i * h * w * c * sizeof(uint16_t);
            gaddr_t outputAddr = ga_output + outer_pos * inner_size * sizeof(uint16_t);
            cvk_tg_stride_t ofmap_gstride = ctx.tg_default_stride({1, (uint32_t)workingOutputSize, 1, (uint32_t)inner_size}, fmt);
            // // original shape
            ctx.tdma_store_stride(tl_lut_result, outputAddr, ofmap_gstride);
        }
        ctx.lmem_free_tensor(tl_lut_reciprocal_result);
        ctx.lmem_free_tensor(tl_lut_working);
        ctx.lmem_free_tensor(tl_lut_result);
        ctx.lmem_free_tensor(tl_parallel_input);
        ctx.lmem_free_tensor(tl_maxValue);
        ctx.lmem_free_tensor(tl_input);
    }
}

void TgSoftmaxKernel::bf16_softmax_kernel_2d() {
    //This constraint is temporarily used.
    //Set uRate ~= 75%
    bool isParallelOuterSize = (outer_size >= NPU_NUM * 3 / 4) ? true : false;
    unsigned int tiledParallelInnerOutputSize = doSplitHeightBf16softmax2DParallelInnerSize();
    unsigned int tiledParallelOuterOutputSize = doSplitHeightBf16softmax2DParallelOuterSize();
    bool isSizeTooLargeToHandle = (tiledParallelInnerOutputSize == 0) || (tiledParallelOuterOutputSize == 0);
    if(!isSizeTooLargeToHandle){
        if(isParallelOuterSize) {
            bf16_softmax_kernel_2d_parallel_outer_size();
        } else {
            bf16_softmax_kernel_2d_parallel_inner_size();
        }
    } else {
        softmaxLargeSizeHandler();
    }
}

int TgSoftmaxKernel::doSplitHeightBf16softmax4D() {
    //Default tileN, do not split C/W
    uint8_t eu_align = 1; // hardware constrainst
    int tileH = h;

    int tableSize = ctx.lmem_tensor_to_size(table_shape, fmt, eu_align) * 4;

    while(true) {
        if(tileH * w > 4095 - 32) {
            tileH--;
            continue;
        }
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(c,tileH * w,1,1);
        int inputSize = ctx.lmem_tensor_to_size(input_shape, fmt, eu_align);
        cvk_tl_shape_t input_transposed_shape = ctx.tl_shape_t4(1,tileH * w,1,c);
        int inputTransposedSize = ctx.lmem_tensor_to_size(input_transposed_shape, fmt, eu_align) * 5;
        //transposedInput + lutWorking*2 + lutResult * 2

        cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1,tileH * w,1,1);
        int maxValueSize = ctx.lmem_tensor_to_size(maxValue_shape, fmt, eu_align);
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

void TgSoftmaxKernel::bf16_softmax_kernel_4d() {
    int tileH = doSplitHeightBf16softmax4D();
    uint8_t eu_align = 1; // hardware constrainst

    int hStep = ceiling_func(h, tileH);
    for(int hStepCounter = 0; hStepCounter < hStep; hStepCounter++) {
        int h_pos = hStepCounter * tileH;
        int tileHeightSize = std::min(h - h_pos, tileH);
        //Allocate input
        cvk_tl_shape_t input_shape = ctx.tl_shape_t4(c,tileHeightSize * w,1,1);
        cvk_tl_t *tl_input =
            ctx.lmem_alloc_tensor(input_shape, fmt, eu_align);
        ASSERT(tl_input);

        cvk_tl_shape_t input_transposed_shape = ctx.tl_shape_t4(1,tileHeightSize * w,1,c);
        //Allocate transpose input
        cvk_tl_t *tl_transpose_input =
            ctx.lmem_alloc_tensor(input_transposed_shape, fmt, eu_align);
        ASSERT(tl_transpose_input);

        for(int i = 0; i < n; i++) {
            //load input
            gaddr_t inputAddr = ga_input + i * h * w * c * sizeof(uint16_t) + h_pos * w * sizeof(uint16_t);
            cvk_tg_stride_t ifmap_gstride = ctx.tg_default_stride({(uint32_t)c,(uint32_t) h * w, 1, 1}, fmt);
            // original shape
            // ctx.tdma_load(tl_input, inputAddr);
            ctx.tdma_load_stride(tl_input, inputAddr, ifmap_gstride);

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

            LLVM_DEBUG(llvm::dbgs() << llvm::format(
                            "        L2L Reshape:\n"
                            "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                            "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                            p2.src->start_address, p2.src->shape.n,
                            p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                            p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                            p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                            p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
            ctx.tiu_copy(&p2);

            cvk_tl_shape_t maxValue_shape = ctx.tl_shape_t4(1,tileHeightSize * w,1,1);
            cvk_tl_t *tl_maxValue =
                ctx.lmem_alloc_tensor(maxValue_shape, fmt, eu_align);
            ASSERT(tl_maxValue);

            max_per_lane_value(tl_transpose_input, tl_maxValue);

            //Input = Input - maxOfInput
            every_input_operate_one_specific_data(tl_transpose_input, tl_maxValue, Sub, true);

            cvk_tl_shape_t lut_working_shape = ctx.tl_shape_t4(2,tileHeightSize * w,1,c);
            cvk_tl_t *tl_lut_working =
                ctx.lmem_alloc_tensor(lut_working_shape, fmt, eu_align);
            ASSERT(tl_lut_working);

            cvk_tl_t *tl_lut_result =
                ctx.lmem_alloc_tensor(input_transposed_shape, fmt, eu_align);
            ASSERT(tl_lut_result);
            //lut exponential
            //tl_lut_result = exp(tl_input)
            exponential(tl_transpose_input, tl_lut_result, tl_lut_working);

            accumulate_per_lane_value(tl_lut_result, tl_maxValue);

            cvk_tl_t *tl_lut_reciprocal_result =
                ctx.lmem_alloc_tensor(input_transposed_shape, fmt, eu_align);
            ASSERT(tl_lut_reciprocal_result);
            //Lut reciprocal value
            reciprocal(tl_maxValue, tl_lut_reciprocal_result, tl_lut_working);

            //ans = exp(input - maxInput) *  reciprocal value
            //Every c multiply its reciprocal value
            {
                cvk_tl_t tl_reshape_reciprocal_value;
                tl_reshape_reciprocal_value.start_address = tl_lut_reciprocal_result->start_address;  // start of lmem
                tl_reshape_reciprocal_value.fmt = fmt;
                tl_reshape_reciprocal_value.shape = tl_lut_result->shape;
                tl_reshape_reciprocal_value.stride = ctx.tl_default_stride(tl_reshape_reciprocal_value.shape, fmt, /*eu_align=*/1);
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

                LLVM_DEBUG(llvm::dbgs() << llvm::format(
                                "        L2L Reshape:\n"
                                "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                                "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                                p2.src->start_address, p2.src->shape.n,
                                p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                                p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                                p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                                p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
                ctx.tiu_copy(&p2);

                cvk_tl_t *selected_tl_output = tl_input;
                //store
                // gaddr_t outputAddr = ga_output + i * h * w * c * sizeof(uint16_t);
                gaddr_t outputAddr = ga_output + i * h * w * c * sizeof(uint16_t) + h_pos * w * sizeof(uint16_t);
                cvk_tg_stride_t ofmap_gstride = ctx.tg_default_stride({(uint32_t)c, (uint32_t)h * w, 1, 1}, fmt);
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
}

void TgSoftmaxKernel::accumulate_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out) {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = tl_out;
    param.ifmap = tl_in;
    param.kh = tl_in->shape.h;
    param.kw = tl_in->shape.w;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    //kernel will fill avg_pooling_const / (kh * kw)
    param.avg_pooling_const = ctx.convert_fp32_to_bf16(1.0 * tl_in->shape.h * tl_in->shape.w);
    param.layer_id = layer_id;
    param.ins_val = 0;
    param.ins_fp = param.avg_pooling_const;

    LLVM_DEBUG(llvm::dbgs() << llvm::format(
        "  tiu_bf16_avg_pooling\n"
        "    ifmap shape (%d, %d, %d, %d)\n"
        "    ofmap shape (%d, %d, %d, %d)\n"
        "    kh %d, kw %d, stride_h %d, stride_w %d\n"
        "    avg_const %f, 0x%x\n",
        tl_in->shape.n, tl_in->shape.c, tl_in->shape.h, tl_in->shape.w, tl_out->shape.n,
        tl_out->shape.c, tl_out->shape.h, tl_out->shape.w, tl_in->shape.h, tl_in->shape.w, 1, 1,
        ctx.convert_fp32_to_bf16(1.0 * tl_in->shape.h * tl_in->shape.w), param.avg_pooling_const););

    ctx.tiu_average_pooling(&param);
}

void TgSoftmaxKernel::every_input_operate_one_specific_data(cvk_tl_t *tl_in_out, cvk_tl_t *tl_sub, OperateMode operate, bool isParallelInLane) {
    cvk_tl_t tl_reshape_parallel_input;
    tl_reshape_parallel_input.start_address = tl_in_out->start_address;  // start of lmem
    tl_reshape_parallel_input.fmt = fmt;
    tl_reshape_parallel_input.shape = tl_in_out->shape;
    tl_reshape_parallel_input.shape.h = tl_in_out->shape.h * tl_in_out->shape.w;
    tl_reshape_parallel_input.shape.w = 1;
    tl_reshape_parallel_input.stride = ctx.tl_default_stride(tl_reshape_parallel_input.shape, fmt, /*eu_align=*/1);

    cvk_tl_t tl_reshape_maxValueBroadcasted;
    tl_reshape_maxValueBroadcasted.start_address = tl_sub->start_address;  // start of lmem
    tl_reshape_maxValueBroadcasted.fmt = fmt;
    tl_reshape_maxValueBroadcasted.shape = tl_reshape_parallel_input.shape;
    tl_reshape_maxValueBroadcasted.stride = ctx.tl_default_stride(tl_reshape_maxValueBroadcasted.shape, fmt, /*eu_align=*/1);
    tl_reshape_maxValueBroadcasted.stride.h = 0;//h stride =0
    if(!isParallelInLane) {
        tl_reshape_maxValueBroadcasted.stride.c = 0;//c stride =0
        tl_reshape_maxValueBroadcasted.stride.n = EU_NUM;
    } else {
        tl_reshape_maxValueBroadcasted.stride.c = EU_NUM;//c stride =0
        tl_reshape_maxValueBroadcasted.stride.n = EU_NUM;//n stride =0
    }

    if(operate == Sub) {
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
    } else if (operate == Mul) {
        cvk_tiu_mul_param_t p = {0};
        p.res_high = nullptr;
        p.res_low = &tl_reshape_parallel_input;
        p.a = &tl_reshape_parallel_input;
        p.b = &tl_reshape_maxValueBroadcasted;
        p.b_is_const = 0;
        p.rshift_bits = 0;
        p.layer_id = layer_id;
        p.relu_enable = false;
        ctx.tiu_mul(&p);
    } else {
        ASSERT(0 && "Not supported operating mode");
    }
}

void TgSoftmaxKernel::broadcast_one_data_to_all_lane(cvk_tl_t *tl_in, cvk_tl_t *tl_out) {
    // reshape
    cvk_tl_t tl_src = {};
    tl_src.start_address = tl_in->start_address;  // start of lmem
    tl_src.fmt = fmt;
    tl_src.shape = {tl_in->shape.n, 1, (uint32_t)NPU_NUM, 1};
    tl_src.stride = ctx.tl_default_stride(tl_src.shape, fmt, /*eu_align=*/1);
    tl_src.stride.h = 0;
    tl_src.stride.n = EU_NUM; //every element = sizeof(BF16), and eu_align  1

    cvk_tl_t tl_dst = {};
    tl_dst.start_address = tl_out->start_address;  // start of lmem
    tl_dst.fmt = fmt;
    tl_dst.shape = {tl_out->shape.n, (uint32_t)NPU_NUM, 1, 1};
    tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/1);

    cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
    p2.src = &tl_src;
    p2.dst = &tl_dst;

    LLVM_DEBUG(llvm::dbgs() << llvm::format(
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

void TgSoftmaxKernel::max_per_lane_value(cvk_tl_t *tl_in, cvk_tl_t *tl_out) {
    cvk_tiu_max_pooling_param_t max_pool_param = {0};
    max_pool_param.ofmap = tl_out;
    max_pool_param.ifmap = tl_in;
    max_pool_param.kh = tl_in->shape.h;
    max_pool_param.kw = tl_in->shape.w;
    max_pool_param.stride_h = 1;
    max_pool_param.stride_w = 1;
    max_pool_param.layer_id = layer_id;
    max_pool_param.ins_val = -128;
    max_pool_param.ins_fp = 0xff7f;
    ctx.tiu_max_pooling(&max_pool_param);
    LLVM_DEBUG(llvm::dbgs() << llvm::format(
                    "  tiu_bf16_max_pooling\n"
                    "    ifmap shape (%d, %d, %d, %d)\n"
                    "    ofmap shape (%d, %d, %d, %d)\n"
                    "    kh %d, kw %d, stride_h %d, stride_w %d\n",
                    tl_in->shape.n, tl_in->shape.c, tl_in->shape.h, tl_in->shape.w, tl_out->shape.n,
                    tl_out->shape.c, tl_out->shape.h, tl_out->shape.w, tl_in->shape.h, tl_in->shape.w, 1, 1););
}

void TgSoftmaxKernel::selectSoftmaxMode(int64_t* shape) {
    if (dimension == 2) {
        outer_size = shape[0];
        inner_size = shape[1];
        n = shape[0];
        c = shape[1];
        functionMode = Softmax2D;
    } else if (dimension == 4) {
        assert(axis == 1 && "Support only axis = 1 (Align c)");
        n = shape[0];
        c = shape[1];
        h = shape[2];
        w = shape[3];
        if(shape[2] * shape[3] == 1) {
            outer_size = shape[0]; //n
            inner_size = shape[1]; //c
            functionMode = Softmax2D;
        } else {
            // batchStep = shape[0];
            outer_size = shape[0] * shape[2] * shape[3]; //n
            inner_size = shape[1]; //c
            functionMode = Softmax4D;
        }
    } else if (dimension == 3) {
        assert(axis == 2 && "Support only axis = 2");
        outer_size = shape[0] * shape[1]; //c * h
        inner_size = shape[2]; //w
        n = shape[0];
        c = shape[1];
        h = shape[2];
        functionMode = Softmax2D;
    }
    if (inner_size == 1) {
        // strange model
        // Because inner_size == 1, the probability is 100%(1)
        fillOneAsGolden();
        return;
    }
}

void TgSoftmaxKernel::fillOneAsGolden() {
    cvk_tdma_l2g_tensor_fill_constant_param_t p = {0};
    cvk_tg_t dst;
    dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_output);
    dst.start_address = ga_output;
    dst.fmt = fmt;
    dst.shape = ctx.tg_shape_t4(1, 1, 1, outer_size);
    dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);
    p.constant = ctx.convert_fp32_to_bf16(1.0f);
    p.dst = &dst;
    ctx.tdma_l2g_tensor_fill_constant(&p);
}

void TgSoftmaxKernel::schedule() {
    if(functionMode == Softmax4D) {
        bf16_softmax_kernel_4d();
    } else {
        bf16_softmax_kernel_2d();
    }
    free_table();
}

void TgSoftmaxKernel::exponential(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work) {
    const int table_thresh_min = -15;
    const int table_thresh_max = 1;
    cvi_backend_tl_lut(
        ctx, layer_id,
        tl_in->start_address, tl_out->start_address, tl_work->start_address,
        tl_exponential_table_answer->start_address, tl_exponential_table_answer_slope->start_address,
        table_thresh_min, table_thresh_max, tl_in->shape.n, tl_in->shape.c, tl_in->shape.h, tl_in->shape.w);
}

void TgSoftmaxKernel::reciprocal(cvk_tl_t *tl_in, cvk_tl_t *tl_out, cvk_tl_t *tl_work) {
    cvi_backend_tl_lut_exponential_mul_mantissa(
        ctx, layer_id,
        tl_in->start_address, tl_out->start_address, tl_work->start_address,
        tl_reciprocal_table_answer->start_address, tl_reciprocal_mantissa_table_answer->start_address, tl_in->shape.n, tl_in->shape.c, tl_in->shape.h, tl_in->shape.w);
}

void TgSoftmaxKernel::init(uint32_t layer_id,
          gaddr_t ga_input,
          gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
          gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
          gaddr_t ga_output,
          int64_t* shape, int axis, int dimension) {
    this->layer_id = layer_id;
    this->ga_input = ga_input;
    this->ga_exponential_table_data_lut = ga_exponential_table_data_lut;
    this->ga_exponential_slope_table_data_lut = ga_exponential_slope_table_data_lut;
    this->ga_reciprocal_table_data_lut = ga_reciprocal_table_data_lut;
    this->ga_reciprocal_table_mantissa_data_lut = ga_reciprocal_table_mantissa_data_lut;
    this->ga_output = ga_output;
    this->axis = axis;
    this->dimension = dimension;
    this->fmt = CVK_FMT_BF16;
    this->fmt_size = ctx.bytesize_of_fmt(fmt);
    selectSoftmaxMode(shape);
    init_table();
}

void TgSoftmaxKernel::init_table() {
  uint8_t eu_align = 1; // hardware constrainst
  //Load exponential table
  this->table_shape = ctx.lut_table_shape(fmt);


  tl_exponential_table_answer =
      ctx.lmem_alloc_tensor(table_shape, fmt, eu_align);
  tl_exponential_table_answer_slope =
      ctx.lmem_alloc_tensor(table_shape, fmt, eu_align);

  ASSERT(tl_exponential_table_answer);
  ASSERT(tl_exponential_table_answer_slope);

  ctx.tdma_load(tl_exponential_table_answer, ga_exponential_table_data_lut);
  ctx.tdma_load(tl_exponential_table_answer_slope, ga_exponential_slope_table_data_lut);
  //Load reciprocal table

  tl_reciprocal_table_answer =
      ctx.lmem_alloc_tensor(table_shape, fmt, eu_align);
  tl_reciprocal_mantissa_table_answer =
      ctx.lmem_alloc_tensor(table_shape, fmt, eu_align);

  ASSERT(tl_reciprocal_table_answer);
  ASSERT(tl_reciprocal_mantissa_table_answer);

  ctx.tdma_load(tl_reciprocal_table_answer, ga_reciprocal_table_data_lut);
  ctx.tdma_load(tl_reciprocal_mantissa_table_answer, ga_reciprocal_table_mantissa_data_lut);
}

void TgSoftmaxKernel::free_table() {
    ctx.lmem_free_tensor(tl_reciprocal_mantissa_table_answer);
    ctx.lmem_free_tensor(tl_reciprocal_table_answer);
    ctx.lmem_free_tensor(tl_exponential_table_answer_slope);
    ctx.lmem_free_tensor(tl_exponential_table_answer);
}

void cvi_backend_tg_bf16_softmax_kernel(
                const CviBackendContext &ctx, uint32_t layer_id,
                gaddr_t ga_input,
                gaddr_t ga_exponential_table_data_lut, gaddr_t ga_exponential_slope_table_data_lut,
                gaddr_t ga_reciprocal_table_data_lut, gaddr_t ga_reciprocal_table_mantissa_data_lut,
                gaddr_t ga_output,
                int64_t* shape, int axis, int dimension) {

  TgSoftmaxKernel kernel(ctx);
  kernel.init(layer_id,
              ga_input,
              ga_exponential_table_data_lut, ga_exponential_slope_table_data_lut,
              ga_reciprocal_table_data_lut, ga_reciprocal_table_mantissa_data_lut,
              ga_output,
              shape, axis, dimension);

  kernel.schedule();
}