#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh


# assuming run after run regression_XXX.sh
./eval_arcface.py \
    --model=arcface_res50_quant_int8_multiplier.mlir \
    --dataset=$DATASET_PATH/lfw/lfw_aligned \
    --pairs=$DATASET_PATH/lfw/pairs.txt \
    --show=True
