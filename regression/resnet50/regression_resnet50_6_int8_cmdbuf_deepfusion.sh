#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# assuming resnet50_quant_int8_multiplier.mlir already exists
# assuming resnet50_in_int8.bin already exists

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple \
    --deep-fusion-simple-stats=resnet50_deepfusion_stats.csv \
    resnet50_quant_int8_multiplier_addr.mlir \
    -o resnet50_opt_deepfusion.mlir

################################
# backend
################################

mlir-opt \
    --deep-fusion-tg2tl-la \
    resnet50_quant_int8_multiplier_addr.mlir \
    -o resnet50_quant_int8_multiplier_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    resnet50_quant_int8_multiplier_tl_la.mlir \
    -o resnet50_quant_int8_multiplier_tl_lw.mlir

# generate cmdbuf
mlir-translate \
    resnet50_quant_int8_multiplier_tl_la.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_la.bin

mlir-translate \
    resnet50_quant_int8_multiplier_tl_lw.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_lw.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_la.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_tl_la.mlir \
    --output=resnet50_int8_la.cvimodel

build_cvimodel.py \
    --cmdbuf cmdbuf_lw.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_tl_lw.mlir \
    --output=resnet50_int8_lw.cvimodel

# profiling cmdbuf
cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp $INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

################################
# run cmdbuf with cmodel
################################
#$RUNTIME_PATH/bin/test_bmnet \
#    resnet50_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_la.bin \
#    out_all_la.bin \
#    16460784 0 16460784 1
#bin_extract.py \
#    out_all_la.bin \
#    out_fc1000_la.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    out_fc1000_la.bin \
#    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
#    int8 1 1 1 1000 5

#$RUNTIME_PATH/bin/test_bmnet \
#    resnet50_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_lw.bin \
#    out_all_lw.bin \
#    16460784 0 16460784 1
#bin_extract.py \
#    out_all_lw.bin \
#    out_fc1000_lw.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    out_fc1000_lw.bin \
#    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
#    int8 1 1 1 1000 5

model_runner \
    --dump-all-tensors \
    --input resnet50_in_int8.bin \
    --model resnet50_int8_la.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_la.bin

model_runner \
    --dump-all-tensors \
    --input resnet50_in_int8.bin \
    --model resnet50_int8_lw.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_lw.bin

if [ $COMPARE_ALL -eq 1 ]; then
  bin_to_npz.py \
      resnet50_cmdbuf_out_all_int8_la.bin \
      neuron_map.csv \
      resnet50_cmdbuf_out_all_int8_la.npz
  npz_compare.py \
      resnet50_cmdbuf_out_all_int8_la.npz \
      resnet50_tensor_all_int8_multiplier.npz \
      --op_info resnet50_op_info_int8_multiplier.csv

  bin_to_npz.py \
      resnet50_cmdbuf_out_all_int8_lw.bin \
      neuron_map.csv \
      resnet50_cmdbuf_out_all_int8_lw.npz
  npz_compare.py \
      resnet50_cmdbuf_out_all_int8_lw.npz \
      resnet50_tensor_all_int8_multiplier.npz \
      --op_info resnet50_op_info_int8_multiplier.csv
fi

# VERDICT
echo $0 PASSED
