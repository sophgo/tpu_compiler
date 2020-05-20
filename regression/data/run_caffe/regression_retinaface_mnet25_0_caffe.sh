#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

export PATH=$TPU_PYTHON_PATH/model/retinaface:$PATH
export PYTHONPATH=$TPU_PYTHON_PATH/model/retinaface:$PYTHONPATH

run_caffe_retinaface.py \
    --model_def $MODEL_DEF \
    --pretrained_model $MODEL_DAT \
    --input_file $REGRESSION_PATH/data/parade.jpg \
    --net_input_dims $NET_INPUT_DIMS \
    --batch_size $BATCH_SIZE \
    --dump_blobs $CAFFE_BLOBS_NPZ

cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32.npz face_rpn_cls_prob_reshape_stride8,face_rpn_bbox_pred_stride8,face_rpn_landmark_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_bbox_pred_stride16,face_rpn_landmark_pred_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_bbox_pred_stride32,face_rpn_landmark_pred_stride32

# VERDICT
echo $0 PASSED
