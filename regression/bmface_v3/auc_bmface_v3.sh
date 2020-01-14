#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh


# assuming run after run regression_XXX.sh
./eval_bmface_v3.py \
    --model=bmface-v3_opt.mlir \
    --dataset=$DATASET_PATH/lfw/bmface_preprocess/bmface_LFW \
    --pairs=$DATASET_PATH/lfw/pairs.txt \
    --show=True