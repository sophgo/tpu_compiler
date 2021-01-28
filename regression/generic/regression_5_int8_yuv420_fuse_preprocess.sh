#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"
if [ $BATCH_SIZE != 1 ]; then
    echo "only batch 1 is supported for yuv420 testing"
    exit 0
fi

# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --pixel_format YUV420_PLANAR \
    --aligned 1 \
    --batch_size $BATCH_SIZE \
    --input_name input \
    --output_npz ${NET}_only_resize_in_fp32.npz

tpuc-opt \
    --add-tpu-preprocess \
    --pixel_format YUV420_PLANAR \
    --input_aligned=true \
    ${NET}_quant_int8.mlir \
    -o ${NET}_quant_int8_yuv420.mlir

# test fused preprocess interpreter
tpuc-interpreter ${NET}_quant_int8_yuv420.mlir \
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_int8_yuv420.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_yuv420.npz \
    --use-tpu-quant-op

#--excepts="$EXCEPTS,input,data" \
cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8_yuv420.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_int8.csv \
    --dequant \
    --tolerance=$TOLERANCE_INT8_MULTIPLER \
    -vv \
    --stats_int8_tensor

# cvimodel
$DIR/../mlir_to_cvimodel.sh \
    -i ${NET}_quant_int8_yuv420.mlir \
    -o ${NET}_int8_yuv420.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_only_resize_in_fp32.npz \
    --model ${NET}_int8_yuv420.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_yuv420.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_yuv420.npz \
    ${NET}_tensor_all_int8_yuv420.npz \
    --op_info ${NET}_op_info_int8.csv \

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression_fused_preprocess
  if [ $BATCH_SIZE -eq 4 ]; then
    DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression_bs4_fused_preprocess
  fi
  mkdir -p $DST_DIR

  mv ${NET}_only_resize_in_fp32.npz $DST_DIR/${NET}_only_resize_yuv420_in_fp32.npz
  mv ${NET}_int8_yuv420.cvimodel $DST_DIR/
  mv ${NET}_cmdbuf_out_all_int8_yuv420.npz \
     $DST_DIR/${NET}_int8_yuv420_out_all.npz
fi

# VERDICT
echo $0 PASSED
