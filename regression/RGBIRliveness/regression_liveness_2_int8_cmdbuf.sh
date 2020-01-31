#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py $REGRESSION_PATH/RGBIRliveness/data/liveness_in_fp32.npz arr_0 liveness_in_fp32.bin
bin_fp32_to_int8.py \
    liveness_in_fp32.bin \
    liveness_in_int8.bin \
    1.0 \
    1.00000489

################################
# quantization 3: multiplier int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    liveness_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    liveness_in_int8.bin \
    weight_int8_multiplier.bin \
    cmdbuf_int8_multiplier.bin \
    liveness_cmdbuf_out_all_int8_multiplier.bin \
    306448 0 306448 1

# run int8 interpreter
mlir-tpu-interpreter \
    liveness_quant_int8_multiplier.mlir \
    --tensor-in $REGRESSION_PATH/RGBIRliveness/data/liveness_in_fp32.npz \
    --tensor-out dummy.bin \
    --dump-all-tensor=liveness_tensor_all_int8_multiplier.npz

# compare all tensors
bin_to_npz.py \
    liveness_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    liveness_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    liveness_cmdbuf_out_all_int8_multiplier.npz \
    liveness_tensor_all_int8_multiplier.npz

# VERDICT
echo $0 PASSED
