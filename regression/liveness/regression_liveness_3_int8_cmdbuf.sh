#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


################################
# prepare int8 input
################################
cvi_npz_tool.py to_bin \
    liveness_tensor_all_int8_multiplier.npz \
    data \
    liveness_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
#cvi_npz_tool.py to_bin liveness_in_fp32.npz arr_0 liveness_in_fp32.bin
#bin_fp32_to_int8.py \
#    liveness_in_fp32.bin \
#    liveness_in_int8.bin \
#    1.0 \
#    1.00000489

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    liveness_quant_int8_multiplier.mlir \
    -o  liveness_quant_int8_tg.mlir

################################
# quantization 3: multiplier int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    liveness_quant_int8_tg.mlir \
    -o liveness_quant_int8_cmdbuf.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    liveness_quant_int8_cmdbuf.mlir \
    -o cmdbuf.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir liveness_quant_int8_cmdbuf.mlir \
    --output liveness_int8_multiplier.cvimodel

# run cvimodel
model_runner \
    --dump-all-tensors \
    --input liveness_in_fp32.npz  \
    --model liveness_int8_multiplier.cvimodel \
    --output liveness_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    liveness_cmdbuf_out_all_int8_multiplier.npz \
    liveness_tensor_all_int8_multiplier.npz \
    --op_info liveness_op_info.csv \
    --tolerance 0.9,0.9,0.6 -v

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  NET=liveness
  cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  cp ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_cmdbuf_out_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  # cp ${NET}_tensor_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  # cp ${NET}_neuron_map_int8_multiplier.csv $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
