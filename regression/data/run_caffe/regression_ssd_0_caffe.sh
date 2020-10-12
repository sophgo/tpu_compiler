#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# run caffe model
run_caffe_detector_ssd.py \
    --model_def $MODEL_DEF  \
    --pretrained_model $MODEL_DAT \
    --net_input_dims $NET_INPUT_DIMS  \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --input_scale $INPUT_SCALE \
    --dump_blobs ${NET}_blobs.npz \
    --batch_size $BATCH_SIZE \
    --input_file $REGRESSION_PATH/data/dog.jpg \
    --label_file $LABEL_MAP  \
    --draw_image dog_out.jpg

# extract input and output
cvi_npz_tool.py extract ${NET}_blobs.npz ${NET}_in_fp32.npz data
cvi_npz_tool.py extract ${NET}_blobs.npz ${NET}_out_fp32_ref.npz detection_out

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
# cvi_npz_tool.py compare ssd300_in_fp32.npz $REGRESSION_PATH/ssd300/data/ssd300_in_fp32.npz
# cp $REGRESSION_PATH/ssd300/data/ssd300_in_fp32.npz ssd300_in_fp32.npz

# VERDICT
echo $0 PASSED
