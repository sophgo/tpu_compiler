#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Eval with caffe"

if [ "$EVAL_MODEL_TYPE" = "imagenet" ]; then
  EVAL_FUNC=eval_classifier.py

  $EVAL_FUNC \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --dataset $DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --count=$1

elif [ "$EVAL_MODEL_TYPE" = "widerface" ]; then
  EVAL_FUNC=eval_caffe_retinaface_widerface.py

  #rm ${NET}_caffe_result_fp32 -rf
  $EVAL_FUNC \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --net_input_dims $NET_INPUT_DIMS \
      --obj_threshold $OBJ_THRESHOLD \
      --nms_threshold $NMS_THRESHOLD \
      --images=$DATASET \
      --annotation=$ANNOTATION \
      --result=./${NET}_caffe_result_fp32

elif [ "$EVAL_MODEL_TYPE" = "lfw" ]; then
  EVAL_FUNC=caffe_eval_arcface.py

  $EVAL_FUNC \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --dataset=$DATASET_PATH/lfw/lfw \
      --pairs=$DATASET_PATH/lfw/pairs.txt \
      --show=True

elif [ "$EVAL_MODEL_TYPE" = "coco" ]; then
  EVAL_FUNC=$EVAL_SCRIPT_CAFFE

  $EVAL_FUNC \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --net_input_dims $NET_INPUT_DIMS \
      --coco_image_path=$DATASET_PATH/coco/val2017/ \
      --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --coco_result_jason_file=./${NET}_coco_results_caffe.json \
      --count=$1

else
  echo "Unknown EVAL_MODEL_TYPE $EVAL_MODEL_TYPE"
  exit 1
fi

echo $0 DONE
