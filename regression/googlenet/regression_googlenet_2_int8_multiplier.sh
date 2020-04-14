#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/googlenet/data/googlenet_calibration_table \
    googlenet_opt.mlir \
    -o googlenet_cali.mlir

mlir-opt \
    --tpu-quant \
    --gen-lrn-table \
    --print-tpu-op-info \
    --tpu-op-info-filename googlenet_op_info_int8_multiplier.csv \
    googlenet_cali.mlir \
    -o googlenet_quant_int8_multiplier.mlir

mlir-tpu-interpreter googlenet_quant_int8_multiplier.mlir \
    --tensor-in googlenet_in_fp32.npz \
    --tensor-out googlenet_out_int8_multiplier.npz \
    --dump-all-tensor=googlenet_tensor_all_int8_multiplier.npz

cvi_npz_tool.py compare \
      googlenet_tensor_all_int8_multiplier.npz \
      googlenet_blobs.npz \
      --op_info googlenet_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.96,0.96,0.71 -vv

################################
# Lower for quantization: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    googlenet_quant_int8_multiplier.mlir \
    -o googlenet_quant_int8_multiplier_tg.mlir

mlir-opt \
    --tg-fuse-leakyrelu --conv-ic-alignment \
    googlenet_quant_int8_multiplier_tg.mlir \
    -o googlenet_quant_int8_multiplier_tg_opt.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=googlenet_weight_map_int8_multiplier.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=googlenet_neuron_map_int8_multiplier.csv \
    --convert-cpu-op \
    googlenet_quant_int8_multiplier_tg_opt.mlir \
    -o googlenet_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    googlenet_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir googlenet_quant_int8_multiplier_addr.mlir \
    --output=googlenet_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    googlenet_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    googlenet_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
model_runner \
    --dump-all-tensors \
    --input googlenet_in_fp32.npz \
    --model googlenet_int8_multiplier.cvimodel \
    --output googlenet_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    googlenet_cmdbuf_out_all_int8_multiplier.npz \
    googlenet_tensor_all_int8_multiplier.npz \
    --op_info googlenet_op_info_int8_multiplier.csv


# VERDICT
echo $0 PASSED
