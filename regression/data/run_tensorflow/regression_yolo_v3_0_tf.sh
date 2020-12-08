#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


TF_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_tf_detector_yolo.py \
      --model_def $MODEL_DEF \
      --net_input_dims $NET_INPUT_DIMS \
      --obj_threshold 0.3 \
      --nms_threshold 0.5 \
      --batch_size $BATCH_SIZE \
      --dump_blobs $TF_BLOBS_NPZ \
      --input_file $REGRESSION_PATH/data/dog.jpg \
      --label_file $REGRESSION_PATH/data/coco-labels-2014_2017.txt \
      --draw_image dog_out.jpg

      cvi_npz_tool.py tranpose $TF_BLOBS_NPZ nhwc nchw
      cvi_npz_tool.py extract $TF_BLOBS_NPZ ${NET}_in_fp32.npz input
fi

# extract input and output

# VERDICT
echo $0 PASSED
