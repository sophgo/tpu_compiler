#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

COMPARE_ALL=1

if [ -z $CALI_TABLE ]; then
  echo "empty CALI_TABLE"
  exit 1
fi

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  # imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
  # wider    : --dataset $DATASET_PATH/widerface/WIDER_val
  DATASET=$DATASET_PATH/imagenet/img_val_extracted
  if [ $NET = "yolo_v4" ] || [ $NET = "yolo_v4_tiny" ]; then
     DATASET=$DATASET_PATH/coco/val2017
  elif [ $NET = "bisenetv2" ]; then
     DATASET=$DATASET_PATH/cityscapes/CityScapes_calibration
  fi

  gen_data_list.py \
      $DATASET \
      $CALIBRATION_IMAGE_COUNT \
      cali_list_imagenet.txt

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
fi

if [ ! -f $CALI_TABLE ]; then
  echo "CALI_TABLE=$CALI_TABLE not exist"
  exit 1
fi

# import calibration table
mlir-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir

echo $0 PASSED
