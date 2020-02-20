#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py retinaface_res50_in_fp32.npz data retinaface_res50_in_fp32.bin

# Depend on retinaface_res50_threshold_table
bin_fp32_to_int8.py \
    retinaface_res50_in_fp32.bin \
    retinaface_res50_in_int8.bin \
    1.0 \
    255.003890991

################################
# quantization : multiplier int8
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
    retinaface_res50-int8.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -debug \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=retinaface_res50_int8_multiplier.cvimodel

# run cmdbuf
$RUNTIME_PATH/bin/test_cvinet \
    retinaface_res50_in_int8.bin \
    retinaface_res50_int8_multiplier.cvimodel \
    retinaface_res50_cmdbuf_out_all_int8_multiplier.bin

# compare all tensors
bin_to_npz.py \
    retinaface_res50_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    retinaface_res50_cmdbuf_out_all_int8_multiplier.npz

npz_compare.py \
    retinaface_res50_cmdbuf_out_all_int8_multiplier.npz \
    retinaface_res50_tensor_all_int8.npz \
    --op_info retinaface_res50_op_info_int8.csv


# VERDICT
echo $0 PASSED
