#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CHECK_NON_OPT_VERSION=0

if [ $DO_FUSE_PREPROCESS -eq 1 ]; then
    # make image data only resize
    cvi_preprocess.py  \
      --image_file $REGRESSION_PATH/data/cat.jpg \
      --net_input_dims ${IMAGE_RESIZE_DIMS} \
      --image_resize_dims ${IMAGE_RESIZE_DIMS} \
      --raw_scale 255 \
      --mean 0,0,0 \
      --std 1,1,1 \
      --input_scale 1 \
      --npz_name ${NET}_only_resize_in_fp32.npz \
      --input_name input

    cvi_model_convert.py \
      --model_path $MODEL_DEF \
      --model_dat=$MODEL_DAT \
      --model_name ${NET} \
      --model_type $MODEL_TYPE \
      --batch_size $BATCH_SIZE \
      --image_resize_dims ${IMAGE_RESIZE_DIMS} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --raw_scale ${RAW_SCALE} \
      --mean ${MEAN} \
      --std ${STD} \
      --batch_size $BATCH_SIZE \
      --input_scale ${INPUT_SCALE} \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --convert_preprocess 1 \
      --mlir_file_path ${NET}_fused_preprocess.mlir

fi

# VERDICT
echo $0 PASSED
