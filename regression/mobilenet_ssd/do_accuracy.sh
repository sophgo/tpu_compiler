#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

count=0
if [ x$1 != x ]; then
    count=$1
fi
EVAL_FUNC=${MLIR_SRC_PATH}/python/cvi_toolkit/eval/eval_detector_voc.py

# caffe eval
echo "eval with caffe"
python3 $EVAL_FUNC \
    --model_def $MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt \
    --pretrained_model $MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
    --net_input_dims 300,300 \
    --mean 127.5,127.5,127.5 \
    --input_scale 0.007843 \
    --dataset=$DATASET_PATH/VOCdevkit \
    --count=$count

echo  "eval with mlir"
python3 $EVAL_FUNC \
    --mlir=mobilenet_ssd_quant_int8_multiplier.mlir \
    --net_input_dims 300,300 \
    --mean 127.5,127.5,127.5 \
    --input_scale 0.007843 \
    --dataset=$DATASET_PATH/VOCdevkit \
    --count=$count

echo $0 DONE



