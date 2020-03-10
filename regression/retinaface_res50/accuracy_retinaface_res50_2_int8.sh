#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=$TPU_PYTHON_PATH/eval_retinaface_on_widerface.py

rm res50_interpreter_result_int8 -rf
python $EVAL_FUNC \
    --model retinaface_res50_quant_int8.mlir \
    --net_input_dims 600,600 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --images=$DATASET_PATH/widerface/WIDER_val/images \
    --annotation=$DATASET_PATH/widerface/wider_face_split \
    --result=./res50_interpreter_result_int8 \
    --int8

echo $0 DONE

