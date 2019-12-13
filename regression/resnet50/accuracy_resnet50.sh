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
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

# gluoncv eval
$EVAL_FUNC \
    --model=resnet50.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean_file=$MLIR_SRC_PATH/bindings/python/tools/mean_resize.npy \
    --count=$1

$EVAL_FUNC \
    --model=resnet50_quant_int8_per_layer.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean_file=$MLIR_SRC_PATH/bindings/python/tools/mean_resize.npy \
    --count=$1

$EVAL_FUNC \
    --model=resnet50_quant_int8_per_channel.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean_file=$MLIR_SRC_PATH/bindings/python/tools/mean_resize.npy \
    --count=$1

$EVAL_FUNC \
    --model=resnet50_quant_int8_multiplier.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean_file=$MLIR_SRC_PATH/bindings/python/tools/mean_resize.npy \
    --count=$1

$EVAL_FUNC \
    --model=resnet50_quant_bf16.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean_file=$MLIR_SRC_PATH/bindings/python/tools/mean_resize.npy \
    --count=$1
