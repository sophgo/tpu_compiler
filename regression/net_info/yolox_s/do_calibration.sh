#!/bin/bash
set -e
export NET=yolox_s
source $REGRESSION_PATH/generic/generic_models.sh
CALI_PATH=$REGRESSION_PATH/$NET
OUT_PATH=$REGRESSION_PATH/generic/regression_out/${SET_CHIP_NAME}/${NET}_bs1

pushd $OUT_PATH
run_calibration.py \
      ${NET}_opt.mlir \
      --dataset=$DATASET_PATH/coco/val2017/ \
      --input_num=${CALIBRATION_IMAGE_COUNT} \
      -o ${NET}_calibration_table
popd