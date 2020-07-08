#!/bin/bash
set -e
set -x


DIR="$( cd "$(dirname "$0")" ; pwd -P )"


CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_segmentation.py \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --net_input_dims $NET_INPUT_DIMS \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights ${NET}_weights.npz \
      --batch_size $BATCH_SIZE \
      --input_file $REGRESSION_PATH/data/city.png \
      --colours $REGRESSION_PATH/data/city_lut.png \
      --draw_image city_out.jpg
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_ref.npz Deconvolution23_deconv

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
# cvi_npz_tool.py compare ${NET}_in_fp32.npz $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz
# cp $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz ${NET}_in_fp32.npz

# VERDICT
echo $0 PASSED
