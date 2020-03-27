#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt \
    --caffemodel $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel \
    -o squeezenet.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename squeezenet_op_info.csv \
    squeezenet.mlir \
    -o squeezenet_id.mlir

# test mlir interpreter
mlir-tpu-interpreter squeezenet.mlir \
    --tensor-in squeezenet_in_fp32.npz \
    --tensor-out squeezenet_out_fp32.npz \
    --dump-all-tensor squeezenet_tensor_all_fp32.npz
cvi_npz_tool.py compare squeezenet_out_fp32.npz squeezenet_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    squeezenet_tensor_all_fp32.npz \
    squeezenet_blobs.npz \
    --op_info squeezenet_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    squeezenet_id.mlir \
    -o squeezenet_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter squeezenet_opt.mlir \
    --tensor-in squeezenet_in_fp32.npz \
    --tensor-out squeezenet_opt_out_fp32.npz
cvi_npz_tool.py compare squeezenet_opt_out_fp32.npz squeezenet_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
