#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_detector_yolo.py \
    --model_def $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt \
    --pretrained_model $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel \
    --net_input_dims 416,416 \
    --obj_threshold 0.3 \
    --nms_threshold 0.5 \
    --dump_blobs yolo_v3_blobs.npz \
    --dump_weights yolo_v3_weights.npz \
    --input_file $REGRESSION_PATH/yolo_v3/data/dog.jpg \
    --label_file $REGRESSION_PATH/yolo_v3/data/coco-labels-2014_2017.txt \
    --draw_image dog_out.jpg

# extract input and output
npz_extract.py yolo_v3_blobs.npz yolo_v3_in_fp32.npz input
npz_extract.py yolo_v3_blobs.npz yolo_v3_out_fp32_ref.npz layer82-conv,layer94-conv,layer106-conv

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py yolo_v3_in_fp32.npz $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz
cp $REGRESSION_PATH/yolo_v3/data/yolo_v3_in_fp32.npz yolo_v3_in_fp32.npz

# VERDICT
echo $0 PASSED
