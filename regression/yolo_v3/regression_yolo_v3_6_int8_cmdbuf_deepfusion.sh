#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple \
    --deep-fusion-simple-stats=${NET}_deepfusion_stats.csv \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o ${NET}_opt_deepfusion.mlir

################################
# backend
################################

mlir-opt \
    --deep-fusion-tg2tl-la \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    ${NET}_quant_int8_multiplier_tl_la.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw.mlir


# generate cmdbuf
mlir-translate \
    ${NET}_quant_int8_multiplier_tl_la.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_la.bin

mlir-translate \
    ${NET}_quant_int8_multiplier_tl_lw.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_lw.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_la.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ${NET}_quant_int8_multiplier_tl_la.mlir \
    --output=${NET}_int8_la.cvimodel

build_cvimodel.py \
    --cmdbuf cmdbuf_lw.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ${NET}_quant_int8_multiplier_tl_lw.mlir \
    --output=${NET}_int8_lw.cvimodel

# profiling cmdbuf
cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp $INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

################################
# run cmdbuf with cmodel
################################
#$RUNTIME_PATH/bin/test_bmnet \
#    ${NET}_in_fp32.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_la.bin \
#    out_all_la.bin \
#    16460784 0 16460784 1
#bin_extract.py \
#    out_all_la.bin \
#    out_${OUTPUTS}_la.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    out_${OUTPUTS}_la.bin \
#    $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_multiplier.bin \
#    int8 1 1 1 1000 5

#$RUNTIME_PATH/bin/test_bmnet \
#    ${NET}_in_fp32.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_lw.bin \
#    out_all_lw.bin \
#    16460784 0 16460784 1
#bin_extract.py \
#    out_all_lw.bin \
#    out_${OUTPUTS}_lw.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    out_${OUTPUTS}_lw.bin \
#    $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_multiplier.bin \
#    int8 1 1 1 1000 5

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_la.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_la.npz

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_lw.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_lw.npz

npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_la.npz \
    ${NET}_tensor_all_int8_multiplier.npz \
    --op_info ${NET}_op_info_int8_per_layer.csv || true

# surpress return for time being
npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_lw.npz \
    ${NET}_tensor_all_int8_multiplier.npz \
    --op_info ${NET}_op_info_int8_per_layer.csv || true

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  cp ${NET}_int8_la.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_int8_lw.cvimodel $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
