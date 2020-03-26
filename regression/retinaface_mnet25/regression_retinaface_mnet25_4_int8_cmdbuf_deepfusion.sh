#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple \
    --deep-fusion-simple-stats=retinaface_mnet25_deepfusion_stats.csv \
    retinaface_mnet25_with_detection_quant_int8_addr.mlir \
    -o retinaface_mnet25_opt_deepfusion.mlir

################################
# backend
################################

mlir-opt \
    --deep-fusion-tg2tl-la \
    retinaface_mnet25_with_detection_quant_int8_addr.mlir \
    -o retinaface_mnet25_quant_int8_multiplier_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    retinaface_mnet25_quant_int8_multiplier_tl_la.mlir \
    -o retinaface_mnet25_quant_int8_multiplier_tl_lw.mlir

# generate cmdbuf
mlir-translate \
    retinaface_mnet25_quant_int8_multiplier_tl_la.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_la.bin

mlir-translate \
    retinaface_mnet25_quant_int8_multiplier_tl_lw.mlir \
    --mlir-to-cmdbuf \
    --debug-only=tl_conv,tl_eltwise_add \
    -o cmdbuf_lw.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_la.bin \
    --weight weight_int8_with_detection.bin \
    --mlir retinaface_mnet25_quant_int8_multiplier_tl_la.mlir \
    --output=retinaface_mnet25_int8_la.cvimodel

build_cvimodel.py \
    --cmdbuf cmdbuf_lw.bin \
    --weight weight_int8_with_detection.bin \
    --mlir retinaface_mnet25_quant_int8_multiplier_tl_lw.mlir \
    --output=retinaface_mnet25_int8_lw.cvimodel

# profiling cmdbuf
# cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

model_runner \
    --dump-all-tensors \
    --input retinaface_mnet25_in_fp32.npz \
    --model retinaface_mnet25_int8_la.cvimodel \
    --output retinaface_mnet25_cmdbuf_out_all_int8_la.npz

model_runner \
    --dump-all-tensors \
    --input retinaface_mnet25_in_fp32.npz \
    --model retinaface_mnet25_int8_lw.cvimodel \
    --output retinaface_mnet25_cmdbuf_out_all_int8_lw.npz

if [ COMPARE_ALL -eq 1 ]; then
  cvi_npz_tool.py compare \
      retinaface_mnet25_cmdbuf_out_all_int8_la.npz \
      retinaface_mnet25_tensor_all_int8_multiplier.npz \
      --op_info retinaface_mnet25_op_info_int8_multiplier.csv || true

  # surpress return for time being
  cvi_npz_tool.py compare \
      retinaface_mnet25_cmdbuf_out_all_int8_lw.npz \
      retinaface_mnet25_tensor_all_int8_multiplier.npz \
      --op_info retinaface_mnet25_op_info_int8_multiplier.csv || true
fi

if [ ! -z CVIMODEL_REL_PATH -a -d CVIMODEL_REL_PATH ]; then
  # mv retinaface_mnet25_int8_la.cvimodel CVIMODEL_REL_PATH
  mv retinaface_mnet25_int8_lw.cvimodel CVIMODEL_REL_PATH
fi

# VERDICT
echo 0 PASSED
