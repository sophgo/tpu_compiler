#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  export EVAL_FUNC=eval_imagenet_pytorch.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval imagenet with gluoncv dataloader"
  export EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

# gluoncv eval
$EVAL_FUNC \
    --model=mobilenet_v1.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean=103.94,116.78,123.68 \
    --input_scale=0.017 \
    --count=$1
