#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

# quantization
mlir-opt \
    --quant-bf16 \
    efficientnet-b0_opt.mlir \
    -o efficientnet-b0_quant_bf16.mlir

# gen sigmoid table 
mlir-opt \
    --gen-sigmoid-table \
    -debug -debug-only=gen-sigmoid-table \
    efficientnet-b0_quant_bf16.mlir \
    -o efficientnet-b0_quant_bf16_table.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    --assign-layer-id \
    efficientnet-b0_quant_bf16_table.mlir \
    -o efficientnet-b0_quant_bf16_cmdbuf.mlir


mlir-translate \
    efficientnet-b0_quant_bf16_cmdbuf.mlir \
    --mlir-to-cmdbuf \
    -debug -debug-only=mlir-to-cmdbuf,lut_kernel,bmnet_bm1880v2_bmkernel_convbf16 \
    -o cmdbuf_bf16.bin


# create bf16 input
npz_to_bin.py $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz data efficientnet_in_fp32.bin
bin_fp32_to_bf16.py \
    efficientnet_in_fp32.bin \
    efficientnet_in_bf16.bin \
    1.0 

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    efficientnet_in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all_bf16.bin \
    0x2CFBC80 0 0x2CFBC80 1 # size, offset, shift, batch

# convert bin to npz
bin_to_npz.py \
    out_all_bf16.bin \
    neuron_map_bf16.csv \
    out_all_bf16.npz

# convert npz from bf16 to fp32
npz_bf16_to_fp32.py out_all_bf16.npz out_all_fp32.npz


# compare with golden fp32
npz_compare.py \
    out_all_fp32.npz \
    efficientnet_tensor_all_bf16.npz -vv

# need to check torlerance later
npz_compare.py \
    ./efficientnet_tensor_all_bf16.npz\
    ./efficientnet_tensor_all_fp32.npz \
    --tolerance=0.99,0.99,0.88 -vvv
    
# VERDICT
echo $0 PASSED
