#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --net_input_dims ${IMAGE_RESIZE_DIMS} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --data_format nhwc \
    --batch_size $BATCH_SIZE \
    --input_name input \
    --output_npz ${NET}_only_resize_in_fp32.npz \

input_shape=`cvi_npz_tool.py get_shape ${NET}_only_resize_in_fp32.npz input`

if [ x$PREPROCESS_CROPMETHOD = x"aspect_ratio" ]; then
    export IMAGE_RESIZE_DIMS=${NET_INPUT_DIMS}
fi

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
    --pixel_format BGR_PACKAGE \
    --batch_size $BATCH_SIZE \
    --convert_preprocess 1 \
    --mlir_file_path ${NET}_fused_preprocess.mlir

tpuc-opt \
    --convert-bn-to-scale \
    --convert-clip-to-relu6 \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_fuesd_preprocess.csv \
    ${NET}_fused_preprocess.mlir \
    -o ${NET}_opt_fused_preprocess.mlir

# test frontend optimizations
tpuc-interpreter ${NET}_opt_fused_preprocess.mlir \
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_fuesd_preprocess.csv  \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=0.999,0.999,0.998 -vv

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
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_int8_multiplier_fused_preprocess.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --use-tpu-quant-op

cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
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
    --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv

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
