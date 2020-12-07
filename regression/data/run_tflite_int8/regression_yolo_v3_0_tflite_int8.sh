#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


TFLITE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_tflite_detector_yolo.py \
      --model_def $MODEL_DEF \
      --net_input_dims $NET_INPUT_DIMS \
      --obj_threshold 0.3 \
      --nms_threshold 0.5 \
      --gen_input_npz ${NET}_in_fp32.npz \
      --batch_size $BATCH_SIZE \
      --input_file $REGRESSION_PATH/data/dog.jpg \
      --label_file $REGRESSION_PATH/data/coco-labels-2014_2017.txt \
      --draw_image dog_out.jpg
fi

# extract input and output
# cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz input
#cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_ref.npz layer82-conv_Y,layer94-conv_Y,layer106-conv_Y

# VERDICT
echo $0 PASSED
