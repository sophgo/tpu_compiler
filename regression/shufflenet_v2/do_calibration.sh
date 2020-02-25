#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
CALI_PATH=$DIR/../../externals/calibration_tool
source $DIR/../../envsetup.sh
mkdir -p lib
cp $CALI_PATH/build/calibration_math.so ./lib/

python $CALI_PATH/run_calibration.py shufflenet shufflenet_opt.mlir ./input.txt --input_num=1000 --out_path ./data
