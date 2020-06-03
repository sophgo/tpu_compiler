#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --batch_size $BATCH_SIZE \
      --label_file $REGRESSION_PATH/data/synset_words.txt \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights ${NET}_weights.npz \
      $REGRESSION_PATH/data/cat.jpg \
      caffe_out.npy
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz $INPUT
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_raw_fp32.npz raw_data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_prob.npz $OUTPUTS_FP32

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
# cvi_npz_tool.py compare ${NET}_in_fp32.npz $REGRESSION_PATH/${NET}/data/${NET}_in_fp32.npz
# cp $REGRESSION_PATH/${NET}/data/${NET}_in_fp32.npz ${NET}_in_fp32.npz

# VERDICT
echo $0 PASSED
