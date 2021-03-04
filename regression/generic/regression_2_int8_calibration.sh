#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ -z $CALI_TABLE ]; then
  echo "empty CALI_TABLE"
  exit 1
fi

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
    ${NET}_opt_fp32.mlir \
    --dataset=$DATASET \
    --input_num=${CALIBRATION_IMAGE_COUNT} \
    --calibration_table=${CALI_TABLE}

if [ ! -f $CALI_TABLE ]; then
  echo "CALI_TABLE=$CALI_TABLE not exist"
  exit 1
fi

echo $0 PASSED
