#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

###############################################################################
# Mix-Precison 1: mix-bf16-broadcastmul + mix-bf16-sigmoid mix-bf16-eltwisemul
###############################################################################
if [ $DO_QUANT_MIX -eq 1 ] && [ $BATCH_SIZE -eq 1 ]; then

    # imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
    # wider    : --dataset $DATASET_PATH/widerface/WIDER_val
    # gen_data_list.py \
    #     $DATASET_PATH/imagenet/img_val_extracted \
    #     10 \
    #     cali_list_imagenet.txt

    echo $REGRESSION_PATH/data/cat.jpg > cali_list_imagenet.txt

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
        --number_bf16=$MIX_PRECISION_BF16_LAYER_NUM

    mlir-opt \
        --assign-chip-name \
        --chipname ${SET_CHIP_NAME} \
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
        --tolerance=$TOLERANCE_MIX_PRECISION -vv

    mlir-opt \
        --tpu-lower --reorder-op \
        ${NET}_mix_precision.mlir \
        -o ${NET}_mix_tg.mlir

    mlir-opt \
        --group-ops \
        ${NET}_mix_tg.mlir \
        -o ${NET}_mix_lg.mlir

    mlir-opt \
        --assign-weight-address \
        --tpu-weight-address-align=16 \
        --tpu-weight-map-filename=${NET}_weight_map_mix.csv \
        --tpu-weight-bin-filename=weight_mix.bin \
        --assign-neuron-address \
        --tpu-neuron-address-align=64 \
        --tpu-neuron-map-filename=${NET}_neuron_map_imx.csv \
        ${NET}_mix_lg.mlir \
        -o ${NET}_mix_addr.mlir

    mlir-opt \
        --divide-ops-to-func \
        ${NET}_mix_addr.mlir \
        -o ${NET}_mix_addr_func.mlir

    mlir-translate \
        --mlir-to-cvimodel \
        --weight-file weight_mix.bin \
        ${NET}_mix_addr_func.mlir \
        -o ${NET}_mix.cvimodel

    model_runner \
        --dump-all-tensors \
        --input ${NET}_in_fp32.npz \
        --model ${NET}_mix.cvimodel \
        --batch-num $BATCH_SIZE \
        --output ${NET}_cmdbuf_out_all_mix.npz

    cvi_npz_tool.py compare \
        ${NET}_cmdbuf_out_all_mix.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_mix.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance=$TOLERANCE_MIX_PRECISION -vv

fi

# VERDICT
echo $0 PASSED
