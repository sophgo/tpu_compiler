#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  export EVAL_FUNC=eval_imagenet_pytorch.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval imagenet with gluoncv dataloader"
  export EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch]"
  return 1
fi

# gluoncv eval
$EVAL_FUNC \
    --model=efficientnet-b0.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --loader_transforms=1 \
    --count=$1
