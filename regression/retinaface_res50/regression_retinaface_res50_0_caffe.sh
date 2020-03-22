#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


export PATH=$TPU_PYTHON_PATH/model/retinaface:$PATH
export PYTHONPATH=$TPU_PYTHON_PATH/model/retinaface:$PYTHONPATH

run_caffe_retinaface.py \
    --model_def $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel \
    --input_file $REGRESSION_PATH/retinaface_res50/data/parade.jpg \
    --dump_blobs retinaface_res50_caffe_blobs.npz

cvi_npz_tool.py extract retinaface_res50_caffe_blobs.npz retinaface_res50_in_fp32.npz data
cvi_npz_tool.py extract retinaface_res50_caffe_blobs.npz retinaface_res50_out_fp32_caffe.npz face_rpn_cls_prob_reshape_stride8,face_rpn_bbox_pred_stride8,face_rpn_landmark_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_bbox_pred_stride16,face_rpn_landmark_pred_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_bbox_pred_stride32,face_rpn_landmark_pred_stride32

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
cvi_npz_tool.py compare retinaface_res50_in_fp32.npz $REGRESSION_PATH/retinaface_res50/data/retinaface_res50_in_fp32.npz
cp $REGRESSION_PATH/retinaface_res50/data/retinaface_res50_in_fp32.npz retinaface_res50_in_fp32.npz

# VERDICT
echo $0 PASSED
