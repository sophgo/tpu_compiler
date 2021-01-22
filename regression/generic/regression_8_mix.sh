#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ $BATCH_SIZE -ne 1 ]; then
  exit 0
fi

###############################################################################
# Mix-Precison 1: mix-bf16-broadcastmul + mix-bf16-sigmoid mix-bf16-eltwisemul
###############################################################################

# imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
# wider    : --dataset $DATASET_PATH/widerface/WIDER_val
# gen_data_list.py \
#     $DATASET_PATH/imagenet/img_val_extracted \
#     10 \
#     cali_list_imagenet.txt

echo $REGRESSION_PATH/data/cat.jpg > cali_list_imagenet.txt

tpuc-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    -o ${NET}_cali.mlir

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

tpuc-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --quant-int8-mix-bf16-layers-from-file ${NET}_mix_precision_bf16_table \
    --tpu-op-info-filename ${NET}_op_info_mix.csv \
    --print-tpu-op-info \
    ${NET}_cali.mlir \
    -o ${NET}_mix.mlir

tpuc-interpreter ${NET}_mix.mlir \
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

$DIR/../mlir_to_cvimodel.sh \
   -i ${NET}_mix.mlir \
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

# VERDICT
echo $0 PASSED
