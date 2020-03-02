#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

export PATH=$TPU_PYTHON_PATH/model/retinaface:$PATH
export PYTHONPATH=$TPU_PYTHON_PATH/model/retinaface:$PYTHONPATH

run_caffe_retinaface.py \
    --model_def $MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel \
    --input_file $REGRESSION_PATH/retinaface_res50/data/parade.jpg \
    --net_input_dims 320,320 \
    --dump_blobs retinaface_mnet25_caffe_blobs.npz

npz_extract.py retinaface_mnet25_caffe_blobs.npz retinaface_mnet25_in_fp32.npz data
npz_extract.py retinaface_mnet25_caffe_blobs.npz retinaface_mnet25_out_fp32_caffe.npz face_rpn_cls_prob_reshape_stride8,face_rpn_bbox_pred_stride8,face_rpn_landmark_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_bbox_pred_stride16,face_rpn_landmark_pred_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_bbox_pred_stride32,face_rpn_landmark_pred_stride32

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py retinaface_mnet25_in_fp32.npz $REGRESSION_PATH/retinaface_mnet25/data/retinaface_mnet25_in_fp32.npz
cp $REGRESSION_PATH/retinaface_mnet25/data/retinaface_mnet25_in_fp32.npz retinaface_mnet25_in_fp32.npz


# VERDICT
echo $0 PASSED
