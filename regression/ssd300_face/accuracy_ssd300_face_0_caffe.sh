#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_detector_ssd300_face.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/face_detection/ssd300_face/caffe/ssd300_face-deploy.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/ssd300_face/caffe/res10_300x300_ssd_iter_140000.caffemodel \
    --fddb \
    --image_path=$DATASET_PATH/fddb/images/ \
    --annotation=$DATASET_PATH/fddb/annotations/ \
    --count=$1

$EVAL_FUNC \
    --model_def $MODEL_PATH/face_detection/ssd300_face/caffe/ssd300_face-deploy.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/ssd300_face/caffe/res10_300x300_ssd_iter_140000.caffemodel \
    --wider \
    --matlib \
    --image_path=$DATASET_PATH/widerface/WIDER_val/images/ \
    --annotation=$DATASET_PATH/widerface/wider_face_split/wider_face_val_bbx_gt.txt \
    --count=$1

echo $0 DONE



