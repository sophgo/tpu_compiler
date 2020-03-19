#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_detector_ssd300_face.py \
    --model_def $MODEL_PATH/face_detection/ssd300_face/caffe/ssd300_face-deploy.prototxt \
    --pretrained_model $MODEL_PATH/face_detection/ssd300_face/caffe/res10_300x300_ssd_iter_140000.caffemodel \
    --net_input_dims 300,300 \
    --dump_blobs ssd300_face_blobs.npz \
    --dump_weights ssd300_face_weights.npz \
    --input_file $REGRESSION_PATH/ssd300_face/data/girl.jpg \
    --draw_image girl_out.jpg

# extract input and output
cvi_npz_tool.py extract ssd300_face_blobs.npz ssd300_face_in_fp32.npz data
cvi_npz_tool.py extract ssd300_face_blobs.npz ssd300_face_out_fp32_ref.npz detection_out

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
cvi_npz_tool.py compare ssd300_face_in_fp32.npz $REGRESSION_PATH/ssd300_face/data/ssd300_face_in_fp32.npz
cp $REGRESSION_PATH/ssd300_face/data/ssd300_face_in_fp32.npz ssd300_face_in_fp32.npz

# VERDICT
echo $0 PASSED
