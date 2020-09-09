#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

# make image data only resize, for interpreter, use fp32
cvi_preprocess.py \
    --image_file $IMAGE_PATH \
    --net_input_dims ${IMAGE_RESIZE_DIMS} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --raw_scale 255 \
    --mean 0,0,0 \
    --std 1,1,1 \
    --input_scale 1 \
    --data_format nhwc \
    --batch_size $BATCH_SIZE \
    --npz_name ${NET}_only_resize_in_fp32.npz \
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
    --data_format nhwc \
    --astype uint8 \
    --batch_size $BATCH_SIZE \
    --crop_method=${PREPROCESS_CROPMETHOD} \
    --npz_name ${NET}_only_resize_in_uint8.npz \
    --input_name input

input_shape=`cvi_npz_tool.py get_shape ${NET}_only_resize_in_fp32.npz input`

if [ $PREPROCESS_CROPMETHOD -eq "aspect_ratio" ]; then
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
    --batch_size $BATCH_SIZE \
    --convert_preprocess 1 \
    --mlir_file_path ${NET}_fused_preprocess.mlir

mlir-opt \
    --fuse-relu \
    ${MLIR_OPT_FE_PRE} \
    --canonicalize \
    ${MLIR_OPT_FE_POST} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_fuesd_preprocess.csv \
    ${NET}_fused_preprocess.mlir \
    -o ${NET}_opt_fused_preprocess.mlir

# test frontend optimizations
mlir-tpu-interpreter ${NET}_opt_fused_preprocess.mlir \
    --tensor-in ${NET}_only_resize_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info_fuesd_preprocess.csv  \
    --excepts="$EXCEPTS,input,data" \
    --tolerance=0.999,0.999,0.998 -vv

mlir-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE}\
    ${NET}_opt_fused_preprocess.mlir \
    -o ${NET}_cali_fused_preprocess.mlir

mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --convert-quant-op \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
    ${NET}_cali_fused_preprocess.mlir \
    -o ${NET}_quant_int8_multiplier_fused_preprocess.mlir

# test fused preprocess int8 interpreter
mlir-tpu-interpreter ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
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
mlir-opt \
    ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
    --use-tpu-quant-op \
    --tpu-lower --reorder-op | \
  mlir-opt \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment | \
  mlir-opt \
    --group-ops | \
  mlir-opt \
    --dce \
    --deep-fusion-tg2tl-la \
    --deep-fusion-tl-la2lw | \
  mlir-opt \
    --compress-weight | \
  mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_int8_lg_fused_preprocess.csv \
    --tpu-weight-bin-filename=weight_fused_process.bin | \
  mlir-opt \
    --tpu-generate-compressed-weight \
    --assign-neuron-address \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=neuron_map_lg_fused_preprocess.csv \
    --divide-ops-to-func | \
mlir-translate \
    --mlir-to-cvimodel \
    --weight-file weight_fused_process.bin \
    -o ${NET}_fused_preprocess.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_only_resize_in_uint8.npz \
    --model ${NET}_fused_preprocess.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
    ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
    --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  if [ $BATCH_SIZE -eq 1 ]; then
    cp ${NET}_only_resize_in_uint8.npz \
        $CVIMODEL_REL_PATH/${NET}_fused_preprocess_in_uint8.npz
    mv ${NET}_fused_preprocess.cvimodel $CVIMODEL_REL_PATH
    cp ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
        $CVIMODEL_REL_PATH/${NET}_fused_preprocess_out_all.npz

  else
    cp ${NET}_only_resize_in_uint8.npz \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess_in_uint8.npz
    mv ${NET}_fused_preprocess.cvimodel \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess.cvimodel
    cp ${NET}_cmdbuf_out_all_int8_multiplier_fused_preprocess.npz \
        $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_fused_preprocess_out_all.npz
  fi
fi

# VERDICT
echo $0 PASSED
