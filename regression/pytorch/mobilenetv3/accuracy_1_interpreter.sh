#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$3

# comes from `https://pypi.org/project/geffnet/` that 'Crop' column of `mobilenetv3_rw`
SCALE=0.875


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

#
# ImageNet ILSVRC2012
# Size: Train:138G+Val:6G
# https://drive.google.com/drive/folders/1dU3PiW6RRQkxfQL9qAR16EhSbUcsofQ8?usp=sharing
# comes from https://t.me/gdshareBot?start=1872
# and run `python preprocess_imagenet_validation_data.py /PATH_YOUR_DOWNLOADED_VAL_TAR/val imagenet_2012_validation_synset_labels.txt` to re-structure folder

# using transforms.Normalize rather
# plz refer https://github.com/rwightman/gen-efficientnet-pytorch/blob/48b290e6a37fbb5290e0df1567d73ed1f8d0f38e/data/transforms.py#L7
##$EVAL_FUNC \
##    --model=${NET}.mlir \
##    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
##    --mean=0.485,0.456,0.406 \
##    --input_scale=0.875 \
##    --loader_transforms 1 \
##    --count=$1
##    #--mean=103.94,116.78,123.68 \
##    #--input_scale=${SCALE} \

$EVAL_FUNC \
    --model=${NET}_quant_int8_multiplier.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean=0.485,0.456,0.406 \
    --loader_transforms 1 \
    --input_scale=${SCALE} \
    --count=$1
#
#$EVAL_FUNC \
#    --model=${NET}_quant_int8_per_layer.mlir \
#    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
#    --mean=103.94,116.78,123.68 \
#    --input_scale=${SCALE} \
#    --count=$1
#
#$EVAL_FUNC \
#    --model=${NET}_quant_int8_per_channel.mlir \
#    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
#    --mean=103.94,116.78,123.68 \
#    --input_scale=${SCALE} \
#    --count=$1
#

#$EVAL_FUNC \
#    --model=${NET}_quant_bf16.mlir \
#    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
#    --mean=103.94,116.78,123.68 \
#    --input_scale=${SCALE} \
#    --count=$1

echo $0 DONE
