#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ $DO_NOT_BF16_UNDER_182x -eq 1 ]; then
  exit 0
fi

PIXEL_FORMAT='BGR_PACKED'
if [ $BGRAY -eq 1 ]; then
    PIXEL_FORMAT='GRAYSCALE'
fi

# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --pixel_format $PIXEL_FORMAT \
    --aligned 0 \
    --batch_size $BATCH_SIZE \
    --input_name input \
    --output_npz ${NET}_only_resize_in_fp32.npz

tpuc-opt \
    --add-tpu-preprocess \
    --pixel_format $PIXEL_FORMAT \
    ${NET}_quant_bf16.mlir \
    -o ${NET}_quant_bf16_fused_preprocess.mlir

# test fused preprocess bf16 interpreter
tpuc-interpreter ${NET}_quant_bf16_fused_preprocess.mlir \
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_bf16_fused_preprocess.npz \
    --dump-all-tensor=${NET}_tensor_all_bf16_fused_preprocess.npz \
    --use-tpu-quant-op

cvi_npz_tool.py compare \
    ${NET}_tensor_all_bf16_fused_preprocess.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --dequant \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=$TOLERANCE_BF16 \
    -vv

# cvimodel
$DIR/../mlir_to_cvimodel.sh \
    -i ${NET}_quant_bf16_fused_preprocess.mlir \
    -o ${NET}_bf16_fused_preprocess.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_only_resize_in_fp32.npz \
    --model ${NET}_bf16_fused_preprocess.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_bf16_fused_preprocess.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_bf16_fused_preprocess.npz \
    ${NET}_tensor_all_bf16_fused_preprocess.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance ${TOLERANCE_BF16_CMDBUF}

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression_bf16
  if [ $BATCH_SIZE -eq 4 ]; then
    DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression_bf16_bs4
  fi
  mkdir -p $DST_DIR

  mv ${NET}_only_resize_in_fp32.npz \
      $DST_DIR/${NET}_only_resize_in_fp32.npz
  mv ${NET}_bf16_fused_preprocess.cvimodel $DST_DIR/${NET}_bf16.cvimodel
  mv ${NET}_cmdbuf_out_all_bf16_fused_preprocess.npz \
      $DST_DIR/${NET}_bf16_out_all.npz
fi

# VERDICT
echo $0 PASSED
