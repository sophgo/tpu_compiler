#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ $MODEL_TYPE = "caffe" ]; then
  if [ $DO_PREPROCESS -eq 1 ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $REGRESSION_PATH/convert_model_caffe_lg_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          ${BATCH_SIZE} \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE_PREPROCESS} \
          ${NET}.cvimodel
    else
      $REGRESSION_PATH/convert_model_caffe_df_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          ${BATCH_SIZE} \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE_PREPROCESS} \
          ${NET}.cvimodel
    fi
  else
    if [ $USE_LAYERGROUP = "1" ]; then
      $REGRESSION_PATH/convert_model_caffe_lg.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          ${BATCH_SIZE} \
          ${CALI_TABLE} \
          ${NET}.cvimodel
    else
      $REGRESSION_PATH/convert_model_caffe_df.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          ${BATCH_SIZE} \
          ${CALI_TABLE} \
          ${NET}.cvimodel
    fi
  fi
elif [ $MODEL_TYPE = "onnx" ]; then
  if [ $USE_LAYERGROUP = "1" ]; then
    $REGRESSION_PATH/convert_model_onnx_lg.sh \
        ${MODEL_DEF} \
        ${NET} \
        ${BATCH_SIZE} \
        ${CALI_TABLE} \
        ${NET}.cvimodel
  else
    $REGRESSION_PATH/convert_model_onnx_df.sh \
        ${MODEL_DEF} \
        ${NET} \
        ${BATCH_SIZE} \
        ${CALI_TABLE} \
        ${NET}.cvimodel
  fi
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

if [ $USE_LAYERGROUP = "1" ]; then
  # echo "compare only output"
  cvi_npz_tool.py compare \
      ${NET}_out_all.npz \
      ${NET}_out_int8_multiplier.npz \
      --op_info op_info_int8.csv \
      --excepts $EXCEPTS
else
  cvi_npz_tool.py compare \
      ${NET}_out_all.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info op_info_int8.csv \
      --excepts $EXCEPTS
fi

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
