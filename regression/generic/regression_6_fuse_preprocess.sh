#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --pixel_format BGR_PACKED \
    --aligned 0 \
    --batch_size $BATCH_SIZE \
    --input_name input \
    --output_npz ${NET}_only_resize_in_fp32.npz

tpuc-opt \
    --add-tpu-preprocess \
    --pixel_format BGR_PACKED \
    --input_aligned=false \
    ${NET}_quant_int8.mlir \
    -o ${NET}_quant_int8_multiplier_fused_preprocess.mlir

# test fused preprocess int8 interpreter
tpuc-interpreter ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_int8_multiplier_fused_preprocess.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --use-tpu-quant-op

cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_int8.csv \
    --dequant \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=$TOLERANCE_INT8_MULTIPLER \
    -vv \
    --stats_int8_tensor

# cvimodel
$DIR/../mlir_to_cvimodel.sh \
    -i ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
    -o ${NET}_fused_preprocess.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_only_resize_in_fp32.npz \
    --model ${NET}_fused_preprocess.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --op_info ${NET}_op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  if [ $BATCH_SIZE -eq 1 ]; then
    cp ${NET}_only_resize_in_fp32.npz \
        $CVIMODEL_REL_PATH/${NET}_fused_preprocess_in_fp32.npz
    mv ${NET}_fused_preprocess.cvimodel $CVIMODEL_REL_PATH
    cp ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
        $CVIMODEL_REL_PATH/${NET}_fused_preprocess_out_all.npz

  else
    cp ${NET}_only_resize_in_fp32.npz \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess_in_fp32.npz
    mv ${NET}_fused_preprocess.cvimodel \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess.cvimodel
    cp ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess_out_all.npz
  fi
fi

# VERDICT
echo $0 PASSED
