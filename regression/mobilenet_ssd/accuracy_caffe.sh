#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

count=0
if [ x$1 != x ]; then
    count=$1
fi
EVAL_FUNC=eval_detection_voc.py

python $EVAL_FUNC \
    --model_def $MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt \
    --pretrained_model $MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
    --net_input_dims 300,300 \
    --voc_path=$DATASET_PATH \
    --count=$count

echo $0 DONE



