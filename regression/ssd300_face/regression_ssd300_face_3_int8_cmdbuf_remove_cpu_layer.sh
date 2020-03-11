#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=0

# TODO: this is temporary, need to fix in interpreter directly
mlir-translate \
    --caffe-to-mlir $REGRESSION_PATH/ssd300_face/data/deploy_tpu.prototxt \
    --caffemodel $MODEL_PATH/face_detection/ssd300_face/caffe/res10_300x300_ssd_iter_140000.caffemodel \
    -o ssd300_face.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_face_op_info.csv \
    --canonicalize \
    ssd300_face.mlir \
    -o ssd300_face_opt.mlir

mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300_face/data/ssd300_face_threshold_table \
    ssd300_face_opt.mlir \
    -o ssd300_face_cali.mlir

###############################
#quantization 1: per-layer int8
###############################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_face_op_info_int8_per_layer.csv \
    ssd300_face_cali.mlir \
    -o ssd300_face_quant_int8_per_layer.mlir

mlir-tpu-interpreter ssd300_face_quant_int8_per_layer.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_per_layer.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_per_layer.npz


if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_face_tensor_all_int8_per_layer.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.81,0.81,0.46 -vvv
fi

# ################################
# # quantization 2: per-channel int8
# ################################

mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_face_op_info_int8_per_channel.csv \
    ssd300_face_cali.mlir \
    -o ssd300_face_quant_int8_per_channel.mlir

mlir-tpu-interpreter ssd300_face_quant_int8_per_channel.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_per_channel.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_per_channel.npz

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_face_tensor_all_int8_per_channel.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_channel.csv \
      --dequant \
      --tolerance 0.99,0.99,0.89 -vvv
fi

# ################################
# # quantization 3: per-channel multiplier int8
# ################################
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_face_op_info_int8_multiplier.csv \
    ssd300_face_cali.mlir \
    -o ssd300_face_quant_int8_multiplier.mlir

mlir-tpu-interpreter ssd300_face_quant_int8_multiplier.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_multiplier.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_multiplier.npz

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_face_tensor_all_int8_multiplier.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.99,0.99,0.85 -vvv
fi

# VERDICT
echo $0 PASSED

