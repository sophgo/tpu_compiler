#!/bin/bash
set -e
export NET=segnet
source $REGRESSION_PATH/generic/generic_models.sh
CALI_PATH=$REGRESSION_PATH/$NET
OUT_PATH=$REGRESSION_PATH/regression_out/${NET}_bs1

pushd $OUT_PATH

run_calibration.py \
      ${NET}_opt.mlir \
      ${CALI_PATH}/cali_list.txt \
      --output_file=${CALI_TABLE} \
      --input_num=101
popd