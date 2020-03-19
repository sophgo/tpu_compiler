#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_INFERENCE_RESULT=0
COMPARE_ALL=0
CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt \
    --caffemodel $MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300.mlir \
    -o dummy.mlir

# test mlir interpreter
mlir-tpu-interpreter ssd300.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_fp32.npz \
    --dump-all-tensor=ssd300_tensor_all_fp32.npz

cvi_npz_tool.py compare ssd300_out_fp32.npz ssd300_out_fp32_ref.npz -v
cvi_npz_tool.py compare \
    ssd300_tensor_all_fp32.npz \
    ssd300_blobs.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vv
fi

# opt
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    --canonicalize \
    ssd300.mlir \
    -o ssd300_opt.mlir

# test opt
mlir-tpu-interpreter ssd300_opt.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_fp32.npz \
    --dump-all-tensor=ssd300_tensor_all_fp32.npz

cvi_npz_tool.py compare ssd300_out_fp32.npz ssd300_out_fp32_ref.npz -v

if [ $COMPARE_ALL -eq 1 ]; then
cvi_npz_tool.py compare \
    ssd300_tensor_all_fp32.npz \
    ssd300_blobs.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.9999,0.9999,0.9999 -vv
fi

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
run_mlir_detector_ssd.py \
      --model ssd300_opt.mlir \
      --net_input_dims 300,300 \
      --dump_blobs ssd300_blobs.npz \
      --obj_threshold 0.5 \
      --dump_weights ssd300_weights.npz \
      --input_file $REGRESSION_PATH/ssd300/data/dog.jpg \
      --label_file $MODEL_PATH/object_detection/ssd/caffe/ssd300/labelmap_coco.prototxt  \
      --draw_image ssd300_fp32_mlir_opt2_result.jpg
fi

# VERDICT
echo $0 PASSED
