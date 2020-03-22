#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# assuming run after run regression_XXX.sh
EVAL_FUNC=eval_alpha_pose.py


# $EVAL_FUNC \
#     --yolov3_model=yolo_v3_416_opt.mlir \
#     --pose_model=alphapose_opt.mlir \
#     --net_input_dims 416,416 \
#     --pose_net_input_dims 256,192 \
#     --obj_threshold 0.6 \
#     --nms_threshold 0.5 \
#     --dataset=$DATASET_PATH/coco/val2017 \
#     --annotations=$DATASET_PATH/coco/annotations/person_keypoints_val2017.json \
#     --result_json=result_alpha_pose.json \
#     --input_file=/home/hongjun/timg.jpeg \
#     --draw_image=./pose_result.jpg \
#     --count=$1


$EVAL_FUNC \
    --yolov3_model=yolo_v3_416_opt.mlir \
    --pose_model=alphapose_opt.mlir \
    --net_input_dims 416,416 \
    --pose_net_input_dims 256,192 \
    --obj_threshold 0.2 \
    --nms_threshold 0.5 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/person_keypoints_val2017.json \
    --result_json=test.json \
    --draw_image=./pose_result.jpg \
    --count=$1

echo $0 DONE
