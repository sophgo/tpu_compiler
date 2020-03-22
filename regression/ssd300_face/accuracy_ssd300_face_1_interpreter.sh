#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


RUN_FDDB=0
export EVAL_FUNC=eval_ssd300_face.py

if [ $RUN_FDDB -eq 1 ]; then
  $EVAL_FUNC \
      --model=ssd300_face_opt.mlir \
      --fddb \
      --image_path=$DATASET_PATH/fddb/images/ \
      --annotation=$DATASET_PATH/fddb/annotations/ \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_per_layer.mlir \
      --fddb \
      --image_path=$DATASET_PATH/fddb/images/ \
      --annotation=$DATASET_PATH/fddb/annotations/ \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_per_channel.mlir \
      --fddb \
      --image_path=$DATASET_PATH/fddb/images/ \
      --annotation=$DATASET_PATH/fddb/annotations/ \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_multiplier.mlir \
      --fddb \
      --image_path=$DATASET_PATH/fddb/images/ \
      --annotation=$DATASET_PATH/fddb/annotations/ \
      --count=$1
else
  $EVAL_FUNC \
      --model=ssd300_face_opt.mlir \
      --wider \
      --matlib \
      --image_path=$DATASET_PATH/widerface/WIDER_val/images/ \
      --annotation=$DATASET_PATH/widerface/wider_face_split/wider_face_val_bbx_gt.txt \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_per_layer.mlir \
      --wider \
      --matlib \
      --image_path=$DATASET_PATH/widerface/WIDER_val/images/ \
      --annotation=$DATASET_PATH/widerface/wider_face_split/wider_face_val_bbx_gt.txt \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_per_channel.mlir \
      --wider \
      --matlib \
      --image_path=$DATASET_PATH/widerface/WIDER_val/images/ \
      --annotation=$DATASET_PATH/widerface/wider_face_split/wider_face_val_bbx_gt.txt \
      --count=$1

  $EVAL_FUNC \
      --model=ssd300_face_quant_int8_multiplier.mlir \
      --wider \
      --matlib \
      --image_path=$DATASET_PATH/widerface/WIDER_val/images/ \
      --annotation=$DATASET_PATH/widerface/wider_face_split/wider_face_val_bbx_gt.txt \
      --count=$1
fi
