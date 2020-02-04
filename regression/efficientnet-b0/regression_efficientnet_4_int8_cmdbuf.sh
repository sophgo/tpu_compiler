#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/efficientnet-b0.prototxt \
    --caffemodel $MODEL_PATH/caffe/efficientnet-b0.caffemodel \
    -o efficientnet-b0.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
   --convert-bn-to-scale \
   --fold-scale \
   --merge-scale-into-conv \
   efficientnet-b0.mlir \
   -o efficientnet-b0_opt.mlir

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/efficientnet-b0/data/efficientnet-b0_threshold_table \
    efficientnet-b0_opt.mlir \
    -o efficientnet-b0_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    efficientnet-b0_cali.mlir \
    -o efficientnet-b0_quant_int8_per_channel.mlir

# get sigmoid table 
mlir-opt \
    --gen-sigmoid-table \
    efficientnet-b0_quant_int8_per_channel.mlir \
    -o efficientnet-b0_quant_int8_per_channel_table.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    efficientnet-b0_quant_int8_per_channel_table.mlir \
    -o  efficientnet-b0_quant_int8_per_channel_cmdbuf.mlir 
    
mlir-translate \
    efficientnet-b0_quant_int8_per_channel_cmdbuf.mlir \
     --mlir-to-cmdbuf \
     -o cmdbuf.bin

# # create int8 input
npz_to_bin.py $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz data efficientnet_in_fp32.bin
bin_fp32_to_int8.py \
    efficientnet_in_fp32.bin \
    efficientnet_in_int8.bin \
    1.0 \
    2.64064478874

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    efficientnet_in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    0x167DEF0 0 0x167DEF0 1 # size, offset, shift, batch

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter efficientnet-b0_quant_int8_per_channel_cmdbuf.mlir \
    --tensor-in $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_int8.npz \
    --dump-all-tensor=efficientnet_tensor_all_int8.npz 
# VERDICT
echo $0 PASSED
