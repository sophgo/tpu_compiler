#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_detector_yolo.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt \
    --pretrained_model $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

echo $0 DONE
