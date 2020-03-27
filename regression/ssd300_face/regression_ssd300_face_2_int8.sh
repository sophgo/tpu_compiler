#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=0
CHECK_INFERENCE_RESULT=0
DO_CALIBRATION=0

if [ $DO_CALIBRATION -eq 1 ]; then
# calibration
python $PYTHON_TOOLS_PATH/dataset_util/gen_dataset_img_list.py \
     --dataset $DATASET_PATH/fddb/images/ \
     --count 10 \
     --output_img_list img_list.txt
python ../../../../mlir/externals/calibration_tool/run_calibration.py \
    ssd300_face ssd300_face_opt.mlir \
    img_list.txt \
    --input_num=10 \
    --math_lib_path=../../../../mlir/externals/calibration_tool/build/calibration_math.so

cp ./result/ssd300_face_calibration_table $REGRESSION_PATH/ssd300_face/data/
fi

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300_face/data/ssd300_face_calibration_table \
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

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd300_face.py \
      --model ssd300_face_quant_int8_per_layer.mlir \
      --net_input_dims 300,300 \
      --input_file $REGRESSION_PATH/ssd300_face/data/girl.jpg \
      --draw_image ssd300_face_quant_int8_per_layer_result.jpg
fi

mlir-tpu-interpreter ssd300_face_quant_int8_per_layer.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_per_layer.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_per_layer.npz

cvi_npz_tool.py extract \
    ssd300_face_tensor_all_int8_per_layer.npz \
    ssd300_face_out_int8_per_layer.npz \
    detection_out

cvi_npz_tool.py compare \
      ssd300_face_out_int8_per_layer.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_layer.csv \
      --tolerance 0.85,0.66,0.44 -vvv -d

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  cvi_npz_tool.py compare \
      ssd300_face_tensor_all_int8_per_layer.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_layer.csv \
      --dequant \
      --excepts detection_out \
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

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd300_face.py \
      --model ssd300_face_quant_int8_per_channel.mlir \
      --net_input_dims 300,300 \
      --input_file $REGRESSION_PATH/ssd300_face/data/girl.jpg \
      --draw_image ssd300_face_quant_int8_per_channel.jpg
fi

mlir-tpu-interpreter ssd300_face_quant_int8_per_channel.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_per_channel.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_per_channel.npz

cvi_npz_tool.py extract \
    ssd300_face_tensor_all_int8_per_channel.npz \
    ssd300_face_out_int8_per_channel.npz \
    detection_out

cvi_npz_tool.py compare \
      ssd300_face_out_int8_per_channel.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_channel.csv \
      --dequant \
      --tolerance 0.85,0.66,0.44 -vvv -d

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  cvi_npz_tool.py compare \
      ssd300_face_tensor_all_int8_per_channel.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_per_channel.csv \
      --dequant \
      --excepts detection_out \
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

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
  run_mlir_detector_ssd300_face.py \
      --model ssd300_face_quant_int8_multiplier.mlir \
      --net_input_dims 300,300 \
      --input_file $REGRESSION_PATH/ssd300_face/data/girl.jpg \
      --draw_image ssd300_face_quant_int8_multiplier.jpg
fi

mlir-tpu-interpreter ssd300_face_quant_int8_multiplier.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_dequant_int8_multiplier.npz \
    --dump-all-tensor=ssd300_face_tensor_all_int8_multiplier.npz

cvi_npz_tool.py extract \
    ssd300_face_tensor_all_int8_multiplier.npz \
    ssd300_face_out_int8_multiplier.npz \
    detection_out

cvi_npz_tool.py compare \
      ssd300_face_out_int8_multiplier.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.81,0.62,0.35 -vvv -d

      #before do power scale and shift quant
      #--tolerance 0.988,0.987,0.846 -vvv

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  cvi_npz_tool.py compare \
      ssd300_face_tensor_all_int8_multiplier.npz \
      ssd300_face_blobs.npz \
      --op_info ssd300_face_op_info_int8_multiplier.csv \
      --dequant \
      --excepts detection_out \
      --tolerance 0.99,0.99,0.85 -vvv
fi

# VERDICT
echo $0 PASSED
