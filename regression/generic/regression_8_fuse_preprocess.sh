#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CHECK_NON_OPT_VERSION=0

if [ $DO_FUSE_PREPROCESS -eq 1 ] && [ $BATCH_SIZE -eq 1 ]; then
    # make image data only resize
    cvi_preprocess.py  \
      --image_file $REGRESSION_PATH/data/cat.jpg \
      --net_input_dims ${IMAGE_RESIZE_DIMS} \
      --image_resize_dims ${IMAGE_RESIZE_DIMS} \
      --raw_scale 255 \
      --mean 0,0,0 \
      --std 1,1,1 \
      --input_scale 1 \
      --data_format nhwc \
      --batch_size $BATCH_SIZE \
      --npz_name ${NET}_only_resize_in_fp32.npz \
      --input_name input

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
      --batch_size $BATCH_SIZE \
      --input_scale ${INPUT_SCALE} \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --convert_preprocess 1 \
      --mlir_file_path ${NET}_fused_preprocess.mlir

    mlir-opt \
      --fuse-relu \
      ${MLIR_OPT_FE_PRE} \
      --canonicalize \
      ${MLIR_OPT_FE_POST} \
      --assign-layer-id \
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
      --excepts="$EXCEPTS,input" \
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
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
      ${NET}_cali_fused_preprocess.mlir \
      -o ${NET}_quant_int8_multiplier_fused_preprocess.mlir

    # test fused preprocess int8 interpreter
    mlir-tpu-interpreter ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
        --tensor-in ${NET}_only_resize_in_fp32.npz \
        --tensor-out ${NET}_out_int8_multiplier_fused_preprocess.npz \
        --dump-all-tensor=${NET}_tensor_all_int8_multiplier_fused_preprocess.npz

    cvi_npz_tool.py compare \
        ${NET}_tensor_all_int8_multiplier_fused_preprocess.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_multiplier_fused_preprocess.csv \
        --dequant \
        --excepts="$EXCEPTS,input" \
        --tolerance=$TOLERANCE_INT8_MULTIPLER_FUSE_PREPROCESS \
        -vv \
        --stats_int8_tensor

    # cvimodel
    mlir-opt \
      ${NET}_quant_int8_multiplier_fused_preprocess.mlir \
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
        --tpu-neuron-address-align=16 \
        --tpu-neuron-map-filename=neuron_map_lg_fused_preprocess.csv \
        -o int8_layergroup_addr.mlir
    mlir-opt \
        --divide-ops-to-func \
        int8_layergroup_addr.mlir \
        -o int8_layergroup_func.mlir

    mlir-translate \
        --mlir-to-cvimodel \
        --weight-file weight_fused_process.bin \
        int8_layergroup_func.mlir \
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

fi

# VERDICT
echo $0 PASSED
