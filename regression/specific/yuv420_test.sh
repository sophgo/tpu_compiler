#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=mobilenet_v2
source $DIR/../generic/generic_models.sh

echo "test preprocess yuv420_csc"

# caffe test
CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --batch_size $BATCH_SIZE \
      --label_file $LABEL_FILE \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      $IMAGE_PATH \
      caffe_out.npy
fi


# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --net_input_dims ${IMAGE_RESIZE_DIMS} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --raw_scale 255 \
    --mean 0,0,0 \
    --std 1,1,1 \
    --input_scale 1 \
    --pixel_format YUV420 \
    --batch_size $BATCH_SIZE \
    --npz_name ${NET}_in_fp32.npz \
    --crop_method=${PREPROCESS_CROPMETHOD} \
    --input_name input


# for uint8 dtype, for model runner(cmodel)
cvi_preprocess.py  \
    --image_file $IMAGE_PATH \
    --net_input_dims ${IMAGE_RESIZE_DIMS} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --raw_scale 255 \
    --mean 0,0,0 \
    --std 1,1,1 \
    --input_scale 1 \
    --pixel_format YUV420 \
    --astype uint8 \
    --batch_size $BATCH_SIZE \
    --crop_method=${PREPROCESS_CROPMETHOD} \
    --npz_name ${NET}_in_uint8.npz \
    --input_name input

input_shape=`cvi_npz_tool.py get_shape ${NET}_in_fp32.npz input`


cvi_model_convert.py \
    --model_path $MODEL_DEF \
    --model_dat=$MODEL_DAT \
    --model_name ${NET} \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --net_input_dims ${NET_INPUT_DIMS} \
    --raw_scale ${RAW_SCALE} \
    --mean ${MEAN} \
    --std ${STD} \
    --input_scale ${INPUT_SCALE} \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --batch_size $BATCH_SIZE \
    --pixel_format YUV420 \
    --convert_preprocess 1 \
    --mlir_file_path ${NET}_fused_preprocess.mlir

tpuc-opt \
    --convert-bn-to-scale \
    --convert-clip-to-relu6 \
    --canonicalize \
    --eltwise-early-stride \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_fuesd_preprocess.csv \
    ${NET}_fused_preprocess.mlir \
    -o ${NET}_opt_fused_preprocess.mlir

# test frontend optimizations
tpuc-interpreter ${NET}_opt_fused_preprocess.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_fuesd_preprocess.csv  \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=0.98,0.98,0.82 -vv

tpuc-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE}\
    ${NET}_opt_fused_preprocess.mlir \
    -o ${NET}_cali_fused_preprocess.mlir

tpuc-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
    ${NET}_cali_fused_preprocess.mlir \
    -o ${NET}_quant_int8_multiplier_fused_preprocess.mlir

# test fused preprocess int8 interpreter
tpuc-interpreter ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8_multiplier_fused_preprocess.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --use-tpu-quant-op

cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
    --dequant \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=0.95,0.94,0.68 \
    -vv \
    --stats_int8_tensor

# cvimodel
$DIR/../mlir_to_cvimodel.sh \
    -i ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
    -o ${NET}_fused_preprocess.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_uint8.npz \
    --model ${NET}_fused_preprocess.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv

# VERDICT
echo $0 PASSED
