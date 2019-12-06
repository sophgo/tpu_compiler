#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    --fuse-relu \
    mobilenet_v2.mlir \
    -o mobilenet_v2_opt.mlir

################################
# prepare bf16 input
################################
bin_fp32_to_bf16.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    in_bf16.bin \
    0.017
# check
diff in_bf16.bin $DATA_PATH/test_cat_in_mobilenet_v2_bf16.bin

################################
# quantization
################################
mlir-opt \
    --quant-bf16 \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_quant_bf16.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    mobilenet_v2_quant_bf16.mlir \
    -o mobilenet_v2_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all.bin \
    18811152 0 18811152 1
bin_extract.py out_all.bin out_fc7_bf16.bin bf16 0x00049800 1000
diff out_fc7_bf16.bin $DATA_PATH/test_cat_out_mobilenet_v2_fc7_bf16.bin
# bin_bf16_to_fp32.py out_fc7_bf16.bin out_fc7_fp32.bin
# bin_dump.py out_fc7_fp32.bin float32 1 1 1 1000 5

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    mobilenet_v2_quant_bf16.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_bf16.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map_bf16.csv out_all.npz
npz_compare.py out_all.npz tensor_all_bf16.npz show 5

# VERDICT
echo $0 PASSED
