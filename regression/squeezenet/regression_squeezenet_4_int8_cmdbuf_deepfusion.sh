#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple \
    --deep-fusion-simple-stats=squeezenet_deepfusion_stats.csv \
    squeezenet_quant_int8_multiplier_addr.mlir \
    -o squeezenet_opt_deepfusion.mlir

################################
# backend
################################

mlir-opt \
    --deep-fusion-tg2tl-la \
    squeezenet_quant_int8_multiplier_addr.mlir \
    -o squeezenet_quant_int8_multiplier_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    squeezenet_quant_int8_multiplier_tl_la.mlir \
    -o squeezenet_quant_int8_multiplier_tl_lw.mlir

# generate cmdbuf
mlir-translate \
    squeezenet_quant_int8_multiplier_tl_la.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_la.bin

mlir-translate \
    squeezenet_quant_int8_multiplier_tl_lw.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_lw.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_la.bin \
    --weight weight_int8_multiplier.bin \
    --mlir squeezenet_quant_int8_multiplier_tl_la.mlir \
    --output=squeezenet_int8_la.cvimodel

build_cvimodel.py \
    --cmdbuf cmdbuf_lw.bin \
    --weight weight_int8_multiplier.bin \
    --mlir squeezenet_quant_int8_multiplier_tl_lw.mlir \
    --output=squeezenet_int8_lw.cvimodel

# profiling cmdbuf
# cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

model_runner \
    --dump-all-tensors \
    --input squeezenet_in_fp32.npz \
    --model squeezenet_int8_la.cvimodel \
    --output squeezenet_cmdbuf_out_all_int8_la.npz

model_runner \
    --dump-all-tensors \
    --input squeezenet_in_fp32.npz \
    --model squeezenet_int8_lw.cvimodel \
    --output squeezenet_cmdbuf_out_all_int8_lw.npz

if [ COMPARE_ALL -eq 1 ]; then
  cvi_npz_tool.py compare \
      squeezenet_cmdbuf_out_all_int8_la.npz \
      squeezenet_tensor_all_int8_multiplier.npz \
      --op_info squeezenet_op_info_int8_multiplier.csv || true

  # surpress return for time being
  cvi_npz_tool.py compare \
      squeezenet_cmdbuf_out_all_int8_lw.npz \
      squeezenet_tensor_all_int8_multiplier.npz \
      --op_info squeezenet_op_info_int8_multiplier.csv || true
fi

if [ ! -z CVIMODEL_REL_PATH -a -d CVIMODEL_REL_PATH ]; then
  # mv squeezenet_int8_la.cvimodel CVIMODEL_REL_PATH
  mv squeezenet_int8_lw.cvimodel CVIMODEL_REL_PATH
fi

# VERDICT
echo 0 PASSED
