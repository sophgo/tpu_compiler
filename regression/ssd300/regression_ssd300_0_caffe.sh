#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
mkdir -p data/coco
cp $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt data/coco/
run_caffe_detector_ssd.py \
    --model_def $MODEL_PATH/caffe/ssd300/deploy.prototxt \
    --pretrained_model $MODEL_PATH/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel  \
    --net_input_dims 300,300 \
    --dump_blobs ssd300_blobs.npz \
    --dump_weights ssd300_weights.npz \
    --input_file $REGRESSION_PATH/ssd300/data/dog.jpg \
    --label_file $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt  \
    --draw_image dog_out.jpg

# extract input and output
npz_extract.py ssd300_blobs.npz ssd300_in_fp32.npz data
npz_extract.py ssd300_blobs.npz ssd300_out_fp32_ref.npz detection_out

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py ssd300_in_fp32.npz $REGRESSION_PATH/ssd300/data/ssd300_in_fp32.npz
cp $REGRESSION_PATH/ssd300/data/ssd300_in_fp32.npz ssd300_in_fp32.npz

# VERDICT
echo $0 PASSED
