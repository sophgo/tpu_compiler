#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo "$0 net=$NET"

PIXEL_FORMAT='BGR_PACKED'
if [ $BGRAY -eq 1 ]; then
    PIXEL_FORMAT='GRAYSCALE'
fi

postfix=''
if [ $BATCH_SIZE -eq 4 ]; then
  postfix='_bs4'
fi

model_transform.py \
  --model_type ${MODEL_TYPE} \
  --model_name ${NET} \
  --model_def ${MODEL_DEF} \
  --model_data ${MODEL_DAT} \
  --image ${IMAGE_PATH} \
  --image_resize_dims ${IMAGE_RESIZE_DIMS} \
  --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
  --net_input_dims ${NET_INPUT_DIMS} \
  --raw_scale ${RAW_SCALE} \
  --mean ${MEAN} \
  --std ${STD} \
  --input_scale ${INPUT_SCALE} \
  --model_channel_order ${MODEL_CHANNEL_ORDER} \
  --gray ${BGRAY} \
  --batch_size $BATCH_SIZE \
  --tolerance ${TOLERANCE_FP32} \
  --excepts ${EXCEPTS} \
  --mlir ${NET}_fp32.mlir

if [ $DO_QUANT_INT8 -eq 1 ]; then
  model_deploy.py \
    --model_name ${NET} \
    --mlir ${NET}_fp32.mlir \
    --calibration_table ${CALI_TABLE} \
    --mix_precision_table ${MIX_PRECISION_TABLE} \
    --chip ${SET_CHIP_NAME} \
    --image ${IMAGE_PATH} \
    --tolerance ${TOLERANCE_INT8_MULTIPLER} \
    --excepts ${EXCEPTS} \
    --fuse_preprocess \
    --pixel_format $PIXEL_FORMAT \
    --aligned_frame false \
    --correctness 0.99,0.99,0.99 \
    --cvimodel ${NET}_int8.cvimodel

  DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression${postfix}_fused_preprocess
  mkdir -p $DST_DIR

  mv ${NET}_in_fp32_resize_only.npz $DST_DIR/${NET}_only_resize_in_fp32.npz
  mv ${NET}_int8.cvimodel $DST_DIR/${NET}_int8.cvimodel
  mv ${NET}_all_tensors_sim.npz $DST_DIR/${NET}_fused_preprocess_out_all.npz
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  model_deploy.py \
    --model_name ${NET} \
    --mlir ${NET}_fp32.mlir \
    --all_bf16 \
    --image ${IMAGE_PATH} \
    --chip ${SET_CHIP_NAME} \
    --tolerance ${TOLERANCE_BF16} \
    --excepts ${EXCEPTS} \
    --fuse_preprocess \
    --pixel_format $PIXEL_FORMAT \
    --aligned_frame false \
    --correctness ${TOLERANCE_BF16_CMDBUF} \
    --cvimodel ${NET}_bf16.cvimodel

  DST_DIR=$CVIMODEL_REL_PATH/cvimodel_regression_bf16${postfix}
  mkdir -p $DST_DIR
  mv ${NET}_in_fp32_resize_only.npz $DST_DIR/${NET}_only_resize_in_fp32.npz
  mv ${NET}_bf16.cvimodel $DST_DIR/
  mv ${NET}_all_tensors_sim.npz $DST_DIR/${NET}_bf16_out_all.npz
fi