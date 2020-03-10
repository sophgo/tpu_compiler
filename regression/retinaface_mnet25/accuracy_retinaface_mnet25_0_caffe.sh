#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=$TPU_PYTHON_PATH/model/retinaface/eval_caffe_retinaface_widerface.py

rm mnet25_caffe_result_fp32 -rf
python $EVAL_FUNC \
    --model_def $MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel \
    --net_input_dims 320,320 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --images=$DATASET_PATH/widerface/WIDER_val/images \
    --annotation=$DATASET_PATH/widerface/wider_face_split \
    --result=./mnet25_caffe_result_fp32

echo $0 DONE

