#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_onnx_detector_yolo.py \
      --model_def $MODEL_DEF \
      --net_input_dims $NET_INPUT_DIMS \
      --obj_threshold 0.3 \
      --nms_threshold 0.5 \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --batch_size $BATCH_SIZE \
      --input_file $REGRESSION_PATH/data/dog.jpg \
      --label_file $REGRESSION_PATH/data/coco-labels-2014_2017.txt \
      --draw_image dog_out.jpg \
      --yolov4-tiny 1
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz input
#cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_ref.npz layer82-conv_Y,layer94-conv_Y,layer106-conv_Y

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
# cvi_npz_tool.py compare ${NET}_in_fp32.npz $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz
# cp $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz ${NET}_in_fp32.npz

# VERDICT
echo $0 PASSED
