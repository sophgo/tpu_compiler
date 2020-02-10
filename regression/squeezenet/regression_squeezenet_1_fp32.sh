#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.prototxt \
    --caffemodel $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel \
    -o squeezenet_v1.1.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename squeezenet_v1.1_op_info.csv \
    squeezenet_v1.1.mlir \
    -o squeezenet_v1.1_id.mlir

# test mlir interpreter
mlir-tpu-interpreter squeezenet_v1.1.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_out_fp32.npz \
    --dump-all-tensor squeezenet_v1.1_tensor_all_fp32.npz
npz_compare.py squeezenet_v1.1_out_fp32.npz squeezenet_v1.1_out_fp32_prob.npz -v
npz_compare.py \
    squeezenet_v1.1_tensor_all_fp32.npz \
    squeezenet_v1.1_blobs.npz \
    --op_info squeezenet_v1.1_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    squeezenet_v1.1_id.mlir \
    -o squeezenet_v1.1_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter squeezenet_v1.1_opt.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_opt_out_fp32.npz
npz_compare.py squeezenet_v1.1_opt_out_fp32.npz squeezenet_v1.1_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
