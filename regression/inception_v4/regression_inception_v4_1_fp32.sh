#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt \
    --caffemodel $MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel \
    -o inception_v4.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v4_op_info.csv \
    inception_v4.mlir \
    -o inception_v4_id.mlir

## test mlir interpreter
mlir-tpu-interpreter inception_v4.mlir \
    --tensor-in inception_v4_in_fp32.npz \
    --tensor-out inception_v4_out_fp32.npz \
    --dump-all-tensor=inception_v4_tensor_all_fp32.npz
npz_compare.py inception_v4_out_fp32.npz inception_v4_out_fp32_prob.npz -v
# set tolerance to 0.91 now, need to check this with fp32 later
npz_compare.py \
    inception_v4_tensor_all_fp32.npz \
    inception_v4_blobs.npz \
    --op_info inception_v4_op_info.csv \
    --tolerance=0.99,0.99,0.91 -vvv

# opt1, convert bn to scale
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    inception_v4_id.mlir \
    -o inception_v4_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter inception_v4_opt.mlir \
    --tensor-in inception_v4_in_fp32.npz \
    --tensor-out inception_v4_opt_out_fp32.npz
npz_compare.py inception_v4_opt_out_fp32.npz inception_v4_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
