#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=$PYTHON_TOOLS_PATH/model/retinaface/eval_caffe_retinaface_widerface.py

python $EVAL_FUNC \
    --model_def $MODEL_PATH/face_detection/retinaface_mobilenet/caffe/200219/mnet_25.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/retinaface_mobilenet/caffe/200219/mnet_25.caffemodel \
    --net_input_dims 600,600 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --images=$DATASET_PATH/widerface/WIDER_val/images \
    --annotation=$DATASET_PATH/widerface/wider_face_split \
    --result=./caffe_result_fp32

echo $0 DONE
