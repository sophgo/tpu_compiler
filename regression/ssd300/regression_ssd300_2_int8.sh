#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
CHECK_INFERENCE_RESULT=0
DO_CALIBRATION=0

if [ $DO_CALIBRATION -eq 1 ]; then
# calibration
python ../../../../mlir/externals/calibration_tool/run_calibration.py \
    ssd300 ssd300_opt2.mlir \
    $DATA_PATH/input.txt \
    --input_num=1000

cp ./result/ssd300_threshold_table $REGRESSION_PATH/ssd300/data/
else
# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300/data/ssd300_threshold_table \
    ssd300_opt2.mlir \
    -o ssd300_cali.mlir

###############################
#quantization 1: per-layer int8
###############################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --gen-sqrt-table \
    --gen-div-table \
    --tpu-op-info-filename ssd300_op_info_int8_per_layer.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_layer.mlir

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd.py \
      --model ssd300_quant_int8_per_layer.mlir \
      --net_input_dims 300,300 \
      --dump_blobs ssd300_blobs.npz \
      --obj_threshold 0.5 \
      --dump_weights ssd300_weights.npz \
      --input_file $REGRESSION_PATH/ssd300/data/dog.jpg \
      --label_file $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt  \
      --draw_image ssd300_quant_int8_per_layer_result.jpg
fi

mlir-tpu-interpreter ssd300_quant_int8_per_layer.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_per_layer.npz \
    --dump-all-tensor=ssd300_tensor_all_int8_per_layer.npz

npz_extract.py \
    ssd300_tensor_all_int8_per_layer.npz \
    ssd300_out_int8_per_layer.npz \
    detection_out

npz_compare.py \
      ssd300_out_int8_per_layer.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_layer.csv \
      --tolerance 0.997,0.997,0.935 -vvv

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_per_layer.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.994,0.993,0.898 -vvv

      #pool4 is the lowest
fi

# ################################
# # quantization 2: per-channel int8
# ################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --gen-sqrt-table \
    --gen-div-table \
    --tpu-op-info-filename ssd300_op_info_int8_per_channel.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_channel.mlir

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd.py \
      --model ssd300_quant_int8_per_channel.mlir \
      --net_input_dims 300,300 \
      --dump_blobs ssd300_blobs.npz \
      --dump_weights ssd300_weights.npz \
      --obj_threshold 0.5 \
      --input_file $REGRESSION_PATH/ssd300/data/dog.jpg \
      --label_file $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt  \
      --draw_image ssd300_quant_int8_per_channel.jpg
fi

mlir-tpu-interpreter ssd300_quant_int8_per_channel.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_per_channel.npz \
    --dump-all-tensor=ssd300_tensor_all_int8_per_channel.npz

npz_extract.py \
    ssd300_tensor_all_int8_per_channel.npz \
    ssd300_out_int8_per_channel.npz \
    detection_out

npz_compare.py \
      ssd300_out_int8_per_channel.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_channel.csv \
      --tolerance 0.998,0.998,0.942 -vvv


if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_per_channel.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_channel.csv \
      --dequant \
      --tolerance 0.988,0.986,0.848 -vvv

      #conv4_3_norm_mbox_loc is the lowest layer
fi

# ################################
# # quantization 3: per-channel multiplier int8
# ################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --gen-sqrt-table \
    --gen-div-table \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_multiplier.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_multiplier.mlir

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd.py \
      --model ssd300_quant_int8_multiplier.mlir \
      --net_input_dims 300,300 \
      --dump_blobs ssd300_blobs.npz \
      --obj_threshold 0.5 \
      --dump_weights ssd300_weights.npz \
      --input_file $REGRESSION_PATH/ssd300/data/dog.jpg \
      --label_file $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt  \
      --draw_image ssd300_quant_int8_multiplier.jpg
fi

mlir-tpu-interpreter ssd300_quant_int8_multiplier.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_multiplier.npz \
    --dump-all-tensor=ssd300_tensor_all_int8_multiplier.npz

npz_extract.py \
    ssd300_tensor_all_int8_multiplier.npz \
    ssd300_out_int8_multiplier.npz \
    detection_out
npz_compare.py \
      ssd300_out_int8_multiplier.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_multiplier.csv \
      --tolerance 0.993,0.992,0.882 -vvv


if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_multiplier.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.993,0.992,0.882 -vvv

      #detection_out is the lowest layer
fi

fi
# VERDICT
echo $0 PASSED
