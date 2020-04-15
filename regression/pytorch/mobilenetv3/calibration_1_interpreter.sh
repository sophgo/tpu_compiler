#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1

#input txt
gen_data_list.py $DATASET_PATH/imagenet/img_val_extracted/val 1000 cali_list_imagenet_1000.txt

run_calibration.py \
    ${NET}_opt.mlir \
    cali_list_imagenet_1000.txt \
    --output_file=${NET}_preprocess_calibration_table \
    --net_input_dims 224,224 \
    --raw_scale 1.0 \
    --histogram_bin_num 65536 \
    --input_num=1000
