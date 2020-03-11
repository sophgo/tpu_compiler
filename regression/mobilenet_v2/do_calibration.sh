#!/bin/bash
set -e

source $MLIR_SRC_PATH/envsetup.sh
CALI_PATH=$REGRESSION_PATH/mobilenet_v2

#input txt
#python3 gen_data_list.py ~/data/dataset/imagenet/img_val_extracted/val 1000 input.txt

python $MLIR_SRC_PATH/python/calibration/run_calibration.py \
    $CALI_PATH/mobilenet_v2/mobilenet_v2_preprocess_opt.mlir \
    $CALI_PATH/data/cali_list_imagenet_1000.txt \
    --output_file=$CALI_PATH/data/mobilenet_v2_preprocess_threshold_table \
    --net_input_dims 224,224 \
    --raw_scale 1.0 \
    --input_num=1000
