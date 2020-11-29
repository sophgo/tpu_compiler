#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

mlir-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8.csv \
    -o ${NET}_quant_int8.mlir

$DIR/../mlir_to_cvimodel.sh \
   -i ${NET}_quant_int8.mlir \
   -o ${NET}_int8.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8.npz

mlir-tpu-interpreter ${NET}_quant_int8.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8.npz \
    --dump-all-tensor=${NET}_tensor_all_int8.npz

if [ ! -z ${TOLERANCE_INT8_MULTIPLER} ]; then
cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_int8.csv \
    --dequant \
    --stats_int8_tensor \
    --except ${EXCEPTS} \
    --tolerance=${TOLERANCE_INT8_MULTIPLER} -vv
fi

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8.npz \
    ${NET}_tensor_all_int8.npz \
    --op_info ${NET}_op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  if [ $BATCH_SIZE -eq 1 ]; then
    cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
    mv ${NET}_int8.cvimodel $CVIMODEL_REL_PATH/${NET}.cvimodel
    cp ${NET}_cmdbuf_out_all_int8.npz $CVIMODEL_REL_PATH/${NET}_out_all.npz
  else
    cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_in_fp32.npz
    mv ${NET}_int8.cvimodel $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}.cvimodel
    cp ${NET}_cmdbuf_out_all_int8.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_out_all.npz
  fi
fi

echo $0 PASSED
