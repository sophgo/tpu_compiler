#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  EVAL_FUNC=eval_imagenet_pytorch.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval imagenet with gluoncv dataloader"
  EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

count_num=$1

$EVAL_FUNC \
    --model=inception_v4.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$1

$EVAL_FUNC \
    --model=inception_v4_quant_int8_per_layer.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$1

$EVAL_FUNC \
    --model=inception_v4_quant_int8_per_channel.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$1

$EVAL_FUNC \
    --model=inception_v4_quant_int8_multiplier.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$1

$EVAL_FUNC \
    --model=inception_v4_quant_bf16.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$count_num

$EVAL_FUNC \
    --model=inception_v4_quant_int8_per_channel.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim 299 \
    --count=$1

echo $0 DONE
