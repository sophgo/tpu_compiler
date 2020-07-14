#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

###############################################################################
# Mix-Precison 1: mix-bf16-broadcastmul + mix-bf16-sigmoid mix-bf16-eltwisemul
###############################################################################
if [ $DO_QUANT_MIX -eq 1 ] && [ $BATCH_SIZE -eq 1 ]; then
    gen_data_list.py \
    $DATASET_PATH/imagenet/img_val_extracted \
    10 \
    cali_list_imagenet.txt

    cvi_mix_precision.py \
        ${NET}_cali.mlir \
        cali_list_imagenet.txt \
        ${NET}_mix_precision_bf16_table \
        --image_resize_dims ${IMAGE_RESIZE_DIMS} \
        --net_input_dims ${NET_INPUT_DIMS} \
        --raw_scale ${RAW_SCALE} \
        --mean ${MEAN} \
        --std ${STD} \
        --input_scale ${INPUT_SCALE} \
        --input_num=1 \
        --number_bf16=10

    mlir-opt \
        --tpu-quant \
        --quant-int8-mix-bf16-layers-from-file ${NET}_mix_precision_bf16_table \
        --tpu-op-info-filename ${NET}_op_info_mix.csv \
        --print-tpu-op-info \
        ${NET}_cali.mlir \
        -o ${NET}_mix_precision.mlir

    mlir-tpu-interpreter ${NET}_mix_precision.mlir \
        --tensor-in ${NET}_in_fp32.npz \
        --tensor-out ${NET}_out_mix.npz \
        --dump-all-tensor=${NET}_tensor_all_mix.npz

    cvi_npz_tool.py compare \
        ${NET}_tensor_all_mix.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_mix.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_MULTIPLER -vv \
        --stats_int8_tensor
fi

# VERDICT
echo $0 PASSED
