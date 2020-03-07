#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py \
    bmface_v3_tensor_all_int8_multiplier.npz \
    data \
    bmface_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
#npz_to_bin.py bmface_v3_in_fp32.npz data bmface_in_fp32.bin
#bin_fp32_to_int8.py \
#    bmface_in_fp32.bin \
#    bmface_in_int8.bin \
#    1.0 \
#    0.996580362

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    bmface_v3_quant_int8_multiplier.mlir \
    -o  bmface_v3_quant_int8_tg.mlir
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    bmface_v3_quant_int8_tg.mlir \
    -o bmface_v3_quant_int8_cmdbuf.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    bmface_v3_quant_int8_cmdbuf.mlir \
    -o cmdbuf.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir bmface_v3_quant_int8_cmdbuf.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output bmface_v3_int8.cvimodel

# run cmdbuf
model_runner \
    --input bmface_in_int8.bin  \
    --model bmface_v3_int8.cvimodel \
    --output bmface_v3_cmdbuf_out_all_int8.bin

# compare all tensors
bin_to_npz.py \
    bmface_v3_cmdbuf_out_all_int8.bin \
    neuron_map.csv \
    bmface_v3_cmdbuf_out_all_int8.npz

npz_compare.py \
    bmface_v3_cmdbuf_out_all_int8.npz \
    bmface_v3_tensor_all_int8_multiplier.npz \
    --op_info bmface_v3_op_info.csv \
    --tolerance 0.9,0.9,0.6 -v


# VERDICT
echo $0 PASSED

