#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  export EVAL_FUNC=eval_caffe_classifier.py
else
  echo "invalid dataloader, choose [pytorch]"
  return 1
fi

# caffe eval
$EVAL_FUNC \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --pretrained_model=/data/models/caffe/efficientnet-b0.caffemodel \
    --model_def=/data/models/caffe/efficientnet-b0.prototxt \
    --loader_transforms=1 \
    --count=$1
