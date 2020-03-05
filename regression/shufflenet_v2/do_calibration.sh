#!/bin/bash
set -e

source $MLIR_SRC_PATH/envsetup.sh
CALI_PATH=$REGRESSION_PATH/shufflenet_v2

#input txt
#python3 gen_data_list.py ~/data/dataset/imagenet/img_val_extracted/val 5000 input.txt

  python $TPU_PYTHON_PATH/run_calibration.py \
    $CALI_PATH/shufflenet/shufflenet_opt.mlir \
    $CALI_PATH/data/input.txt \
    --output_file=$CALIPATH/data/shufflenet_v2_threshold_table \
    --net_input_dims 224,224 \
    --raw_scale 1.0 \
    --input_num=1000