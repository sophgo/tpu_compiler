#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    bmface-v3_quant_int8_multiplier.mlir \
    -o  bmface-v3_quant_int8_tg.mlir

# # create int8 input
npz_to_bin.py $REGRESSION_PATH/bmface_v3/data/bmface_v3_in_fp32_scale.npz data bmface_in_fp32.bin
bin_fp32_to_int8.py \
    bmface_in_fp32.bin \
    bmface_in_int8.bin \
    1.0 \
    0.996580362

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    bmface-v3_quant_int8_tg.mlir \
    -o bmface-v3_quant_int8_cmdbuf.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    bmface-v3_quant_int8_cmdbuf.mlir \
    -o cmdbuf.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --neuron_map neuron_map.csv \
    --output=bmface-v3_int8.cvimodel

# run cmdbuf

$RUNTIME_PATH/bin/test_cvinet \
    bmface_in_int8.bin  \
    bmface-v3_int8.cvimodel \
    bmface-v3_cmdbuf_out_all_int8.bin


# compare all tensors
bin_to_npz.py \
    bmface-v3_cmdbuf_out_all_int8.bin \
    neuron_map.csv \
    bmface-v3_cmdbuf_out_all_int8.npz

npz_compare.py \
    bmface-v3_cmdbuf_out_all_int8.npz \
    bmface-v3_tensor_all_int8_multiplier.npz\
    --op_info bmface-v3_op_info.csv \
    --tolerance 0.9,0.9,0.6 -v


# VERDICT
echo $0 PASSED

