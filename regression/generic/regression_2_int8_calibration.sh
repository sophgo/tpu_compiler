#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=0

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  # imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
  # wider    : --dataset $DATASET_PATH/widerface/WIDER_val
  python $TPU_PYTHON_PATH/dataset_util/gen_dataset_img_list.py \
      --dataset $DATASET_PATH/imagenet/img_val_extracted \
      --count $CALIBRATION_IMAGE_COUNT \
      --output_img_list cali_list_imagenet.txt

  python $TPU_PYTHON_PATH/run_calibration.py \
    ${NET}_opt.mlir \
    cali_list_imagenet.txt \
    --output_file=${CALI_TABLE} \
    --net_input_dims $NET_INPUT_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --input_scale $INPUT_SCALE \
    --input_num=$CALIBRATION_IMAGE_COUNT
fi

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir

echo $0 PASSED
