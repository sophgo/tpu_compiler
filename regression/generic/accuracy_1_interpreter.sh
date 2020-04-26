#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


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

$EVAL_FUNC \
    --model=${NET}.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --input_scale $INPUT_SCALE \
    --count=$1

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_int8_per_tensor.mlir \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --count=$1
fi

#if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then
#  $EVAL_FUNC \
#      --model=${NET}_quant_int8_rshift_only.mlir \
#      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
#      --net_input_dims $NET_INPUT_DIMS \
#      --raw_scale $RAW_SCALE \
#      --mean $MEAN \
#      --input_scale $INPUT_SCALE \
#      --count=$1
#fi

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_int8_multiplier.mlir \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --count=$1
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_bf16.mlir \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --count=$1
fi

echo $0 DONE
