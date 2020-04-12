#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ $MODEL_TYPE = "caffe" ]; then
  $REGRESSION_PATH/convert_model_caffe.sh \
      ${MODEL_DEF} \
      ${MODEL_DAT} \
      ${BATCH_SIZE} \
      ${CALI_TABLE} \
      ${NET}.cvimodel
elif [ $MODEL_TYPE = "onnx" ]; then
  $REGRESSION_PATH/convert_model_onnx.sh \
      ${MODEL_DEF} \
      ${NET} \
      ${BATCH_SIZE} \
      ${CALI_TABLE} \
      ${NET}.cvimodel
else
  echo "Invalid MODEL_TYPE=$MODEL_TYPE"
  return 1
fi

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_out_all.npz

cvi_npz_tool.py compare \
    ${NET}_out_all.npz \
    ${NET}_tensor_all_int8_multiplier.npz \
    --op_info op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  mv ${NET}.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_out_all.npz $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
