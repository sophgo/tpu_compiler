#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

run_caffe_detector_frcn.py \
    --model_def $MODEL_DEF \
    --pretrained_model $MODEL_DAT \
    --input_file $REGRESSION_PATH/data/dog.jpg \
    --net_input_dims $NET_INPUT_DIMS \
    --batch_size $BATCH_SIZE \
    --draw_image result.jpg \
    --dump_blobs $CAFFE_BLOBS_NPZ

cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz input
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32.npz output

# VERDICT
echo $0 PASSED