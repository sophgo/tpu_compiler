#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

COMPARE_ALL=1

OP_LOWERING=0
COMPRESS_WEIGHT=1

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

# cat for logging
echo "cat ${NET}_quant_int8_multiplier_tl_lw.mlir"
cat ${NET}_quant_int8_multiplier_tl_lw.mlir

if [ $OP_LOWERING -eq 1 ]; then
  # function argument lower to MemRefType
  mlir-opt \
      --convert-func-to-memref \
      ${NET}_quant_int8_multiplier_tl_lw.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw_memref.mlir

  # op lower to MemRefType
  mlir-opt \
      --convert-tg-op-to-memref \
      ${NET}_quant_int8_multiplier_tl_lw_memref.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw_op_memref.mlir

  # memory space w/ global memory reuse
  mlir-opt \
      --enable-reuse-global-memory=true \
      --assign-neuron-address-memref \
      --tpu-neuron-address-align-memref=16 \
      --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
      ${NET}_quant_int8_multiplier_tl_lw_op_memref.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw_op_memref_addr.mlir

  # tg op back to TensorType
  mlir-opt \
      --convert-tg-op-to-tensor \
      ${NET}_quant_int8_multiplier_tl_lw_op_memref_addr.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw_op_tensor_addr.mlir

  # function argument back to TensorType
  mlir-opt \
      --convert-func-to-tensor \
      ${NET}_quant_int8_multiplier_tl_lw_op_tensor_addr.mlir \
      -o ${NET}_quant_int8_multiplier_tl_lw.mlir
fi

# generate cmdbuf
# --debug-only=tl_conv,tl_eltwise_add \
mlir-translate \
    ${NET}_quant_int8_multiplier_tl_la.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf_la.bin

mlir-translate \
    ${NET}_quant_int8_multiplier_tl_lw.mlir \
    --mlir-to-cmdbuf \
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

# if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
#   mv ${NET}_int8_lw.cvimodel $CVIMODEL_REL_PATH
# fi

# VERDICT
echo $0 PASSED
