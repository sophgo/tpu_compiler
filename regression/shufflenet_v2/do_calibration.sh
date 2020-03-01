#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
CALI_PATH=$DIR/../../externals/calibration_tool
source $DIR/../../envsetup.sh
mkdir -p lib
cp $CALI_PATH/build/calibration_math.so ./lib/

#input txt
#python3 gen_data_list.py ~/data/dataset/imagenet/img_val_extracted/val 5000 input.txt

python $CALI_PATH/run_calibration.py shufflenet_v2 shufflenet_opt.mlir ./data/input.txt \
            --input_num=1000 \
            --out_path ./data
