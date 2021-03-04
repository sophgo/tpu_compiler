#!/bin/bash
set -e
export NET=mobilenet_ssd
source $REGRESSION_PATH/generic/generic_models.sh
CALI_PATH=$REGRESSION_PATH/$NET
OUT_PATH=$REGRESSION_PATH/regression_out/${NET}_bs1

pushd $OUT_PATH
gen_data_list.py \
         $DATASET_PATH/VOCdevkit/VOC2012/JPEGImages \
         $CALIBRATION_IMAGE_COUNT \
         cali_list_voc2012.txt

run_calibration.py \
      ${NET}_opt.mlir \
      cali_list_voc2012.txt \
      --output_file=${CALI_TABLE} \
      --input_num=${CALIBRATION_IMAGE_COUNT}
popd