#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

if [ -z $CALI_TABLE ]; then
  echo "empty CALI_TABLE"
  exit 1
fi

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  # imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
  # wider    : --dataset $DATASET_PATH/widerface/WIDER_val
  gen_data_list.py \
      $DATASET_PATH/imagenet/img_val_extracted \
      $CALIBRATION_IMAGE_COUNT \
      cali_list_imagenet.txt
  if [ $DO_PREPROCESS -ne 1 ]; then
    run_calibration.py \
        ${NET}_opt.mlir \
        cali_list_imagenet.txt \
        --output_file=${CALI_TABLE} \
        --image_resize_dims ${IMAGE_RESIZE_DIMS} \
        --net_input_dims ${NET_INPUT_DIMS} \
        --raw_scale ${RAW_SCALE} \
        --mean ${MEAN} \
        --std ${STD} \
        --input_scale ${INPUT_SCALE} \
        --input_num=${CALIBRATION_IMAGE_COUNT}
  else
    run_calibration.py \
        ${NET}_opt.mlir \
        cali_list_imagenet.txt \
        --output_file=${CALI_TABLE_PREPROCESS} \
        --image_resize_dims ${IMAGE_RESIZE_DIMS} \
        --net_input_dims ${NET_INPUT_DIMS} \
        --raw_scale 255 \
        --mean 0,0,0 \
        --std 1,1,1 \
        --input_scale 1 \
        --input_num=${CALIBRATION_IMAGE_COUNT}
  fi
fi

if [ ! -f $CALI_TABLE ]; then
  echo "CALI_TABLE=$CALI_TABLE not exist"
  exit 1
fi

# import calibration table
if [ $DO_PREPROCESS -ne 1 ]; then
  mlir-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir
else
  mlir-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE_PREPROCESS} \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir
fi

echo $0 PASSED
