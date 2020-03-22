#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# assuming run after run regression_XXX.sh
./eval_bmface_v3.py \
    --model=bmface_v3_quant_int8_multiplier.mlir \
    --dataset=$DATASET_PATH/lfw/bmface_preprocess/bmface_LFW \
    --pairs=$DATASET_PATH/lfw/pairs.txt \
    --show=True
