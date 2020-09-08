#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

COMPARE_ALL=1

COMPRESS_WEIGHT=1

# assuming ${NET}_quant_int8_multiplier.mlir already exists
# assuming ${NET}_in_fp32.bin already exists

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple-stats=${NET}_deepfusion_stats.csv \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o ${NET}_opt_deepfusion.mlir

################################
# backend
################################

mlir-opt \
    --deep-fusion-group-slice \
    --deep-fusion-tg2tl-la \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    ${NET}_quant_int8_multiplier_tl_la.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw.mlir

if [ $COMPRESS_WEIGHT -eq 1 ]; then
  # Compress weight
  mlir-opt \
      --compress-weight \
      --tpu-compressed-weight-map-filename=${NET}_quant_int8_tl_lw_compressed_weight_stats.csv \
      ${NET}_quant_int8_multiplier_tl_lw.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw_z.mlir

  # assign weight address & neuron address
  mlir-opt \
      --assign-weight-address \
      --tpu-weight-address-align=16 \
      --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier_z.csv \
      --tpu-weight-bin-filename=weight_int8_multiplier.bin \
      --tpu-generate-compressed-weight \
      ${NET}_quant_int8_multiplier_tl_lw_z.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw.mlir
fi

mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=neuron_map_xxx.csv \
    ${NET}_quant_int8_multiplier_tl_lw.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_1.mlir

mlir-opt \
    --divide-ops-to-func \
    ${NET}_quant_int8_multiplier_tl_lw_1.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --weight-file weight_int8_multiplier.bin \
    ${NET}_quant_int8_multiplier_tl_lw_func.mlir \
    -o ${NET}_int8_lw.cvimodel

# profiling cmdbuf
# cvi_profiling --cmdbuf cmdbuf_lw.bin
# libreoffice analysis.csv
# cp $INSTALL_PATH/bin/performance.html .
# google-chrome performance.html

#model_runner \
#    --input ${NET}_in_fp32.npz \
#    --model ${NET}_int8_la.cvimodel \
#    --batch-num $BATCH_SIZE \
#    --output ${NET}_cmdbuf_out_all_int8_la.npz

# open INFO log with VLOG
# GLOG_minloglevel=0 GLOG_logtostderr=1 GLOG_v=3 \
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

# VERDICT
echo $0 PASSED
