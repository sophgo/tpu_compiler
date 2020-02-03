#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_detector_ssd.py

if [[ $2 -eq 1 ]]; then
$EVAL_FUNC \
    --model_def $MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt \
    --pretrained_model $MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel  \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_caffe.json \
    --count=$1  \
    --pre_result_json=./coco_results_caffe.json
else
$EVAL_FUNC \
    --model_def $MODEL_PATH/caffe/ssd300/deploy.prototxt \
    --pretrained_model $MODEL_PATH/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel  \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_caffe.json \
    --count=$1 
fi

echo $0 DONE



