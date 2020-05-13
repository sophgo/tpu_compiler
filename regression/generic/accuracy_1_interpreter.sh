#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  EVAL_FUNC=eval_classifier.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval imagenet with gluoncv dataloader"
  EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

if [ $DO_ACCURACY_FP32_INTERPRETER -eq 1 ]; then
  $EVAL_FUNC \
      --mlir_file=${NET}.mlir \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --std $STD \
      --input_scale $INPUT_SCALE \
      --model_type mlir \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --count=$1
fi

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  $EVAL_FUNC \
    --mlir_file=${NET}_quant_int8_per_tensor.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then
  $EVAL_FUNC \
    --mlir_file=${NET}_quant_int8_rshift_only.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  if [ "$EVAL_MODEL_TYPE" = "face" ]; then
    _EVAL_FUNC=eval_retinaface_on_widerface.py

    #rm ${NET}_interpreter_result_int8 -rf
    $_EVAL_FUNC \
        --model ${NET}_quant_int8_multiplier.mlir \
        --net_input_dims $NET_INPUT_DIMS \
        --obj_threshold $OBJ_THRESHOLD \
        --nms_threshold $NMS_THRESHOLD \
        --images=$DATASET \
        --annotation=$ANNOTATION \
        --result=./${NET}_interpreter_result_int8 \
        --int8
  elif [ "$EVAL_MODEL_TYPE" = "lfw" ]; then
    _EVAL_FUNC=eval_arcface.py
    $_EVAL_FUNC \
      --model=${NET}_quant_int8_multiplier.mlir \
      --dataset=$DATASET_PATH/lfw/lfw \
      --pairs=$DATASET_PATH/lfw/pairs.txt \
      --show=True
  elif [ "$EVAL_MODEL_TYPE" = "coco" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_INT8
    $_EVAL_FUNC \
      --model=${NET}_quant_int8_multiplier.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --coco_image_path=$DATASET_PATH/coco/val2017/ \
      --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --coco_result_jason_file=./${NET}_coco_results_int8_multiplier.json \
      --count=$1
  else
    $EVAL_FUNC \
      --mlir_file=${NET}_quant_int8_multiplier.mlir \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --std $STD \
      --input_scale $INPUT_SCALE \
      --model_type mlir \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --count=$1
  fi
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  $EVAL_FUNC \
    --mlir_file=${NET}_quant_bf16.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

echo $0 DONE
