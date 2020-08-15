#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

$REGRESSION_PATH/convert_model.sh \
    -i ${MODEL_DEF} \
    -d ${MODEL_DAT} \
    -t ${MODEL_TYPE} \
    -b ${BATCH_SIZE} \
    -q ${CALI_TABLE} \
    -l ${USE_LAYERGROUP} \
    -o ${NET}.cvimodel

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
  if [ $BATCH_SIZE -eq 1 ]; then
    cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
    mv ${NET}.cvimodel $CVIMODEL_REL_PATH
    cp ${NET}_out_all.npz $CVIMODEL_REL_PATH
  else
    cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_in_fp32.npz
    mv ${NET}.cvimodel $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}.cvimodel
    cp ${NET}_out_all.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_out_all.npz
  fi
fi

# VERDICT
echo $0 PASSED
