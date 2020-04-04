#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

# assuming ${NET}_quant_int8_multiplier.mlir already exists
# assuming ${NET}_in_fp32.bin already exists

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

# cat for logging
echo "cat ${NET}_quant_int8_multiplier_tl_lw.mlir"
cat ${NET}_quant_int8_multiplier_tl_lw.mlir

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
#build_cvimodel.py \
#    --cmdbuf cmdbuf_la.bin \
#    --weight weight_int8_multiplier.bin \
#    --mlir ${NET}_quant_int8_multiplier_tl_la.mlir \
#    --output=${NET}_int8_la.cvimodel

build_cvimodel.py \
    --cmdbuf cmdbuf_lw.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ${NET}_quant_int8_multiplier_tl_lw.mlir \
    --output=${NET}_int8_lw.cvimodel

# profiling cmdbuf
# cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp $INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

#model_runner \
#    --dump-all-tensors \
#    --input ${NET}_in_fp32.npz \
#    --model ${NET}_int8_la.cvimodel \
#    --batch-num $BATCH_SIZE \
#    --output ${NET}_cmdbuf_out_all_int8_la.npz

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_lw.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_lw.npz

if [ $COMPARE_ALL -eq 1 ]; then
  #cvi_npz_tool.py compare \
  #    ${NET}_cmdbuf_out_all_int8_la.npz \
  #    ${NET}_tensor_all_int8_multiplier.npz \
  #    --op_info ${NET}_op_info_int8_multiplier.csv

  # surpress return for time being
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_lw.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv
fi

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  # mv ${NET}_int8_la.cvimodel $CVIMODEL_REL_PATH
  mv ${NET}_int8_lw.cvimodel $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
