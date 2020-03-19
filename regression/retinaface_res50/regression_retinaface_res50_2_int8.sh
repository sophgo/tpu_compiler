#!/bin/bash
set -e

COMPARE_ALL=1
DO_CALIBRATION=0

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  python $PYTHON_TOOLS_PATH/dataset_util/gen_dataset_img_list.py \
      --dataset $DATASET_PATH/widerface/WIDER_val \
      --count 100 \
      --output_img_list cali_list_widerface_100.txt

  python $PYTHON_TOOLS_PATH/model/retinaface/calibrate_retinaface.py \
      retinaface_res50 \
      retinaface_res50_opt.mlir \
      cali_list_widerface_100.txt \
      retinaface_res50_threshold_table \
      --input_num=100 \
      --histogram_bin_num=65536 \
      --out_path=. \
      --math_lib_path=$CALIBRATION_TOOL_PATH/calibration_math.so
else
  cp $REGRESSION_PATH/retinaface_res50/data/retinaface_res50_threshold_table .
fi

mlir-opt \
    --import-calibration-table \
    --calibration-table retinaface_res50_threshold_table \
    retinaface_res50_opt.mlir \
    -o retinaface_res50_cali.mlir

# Quantization
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info_int8.csv \
    retinaface_res50_cali.mlir \
    -o retinaface_res50_quant_int8.mlir

# Interpreter int8 result
mlir-tpu-interpreter retinaface_res50_quant_int8.mlir \
    --tensor-in retinaface_res50_in_fp32.npz \
    --tensor-out retinaface_res50_out_dequant_int8.npz \
    --dump-all-tensor=retinaface_res50_tensor_all_int8.npz

# compare output
cvi_npz_tool.py extract \
    retinaface_res50_tensor_all_int8.npz \
    retinaface_res50_out_int8.npz \
    face_rpn_bbox_pred_stride16,face_rpn_bbox_pred_stride32,face_rpn_bbox_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_cls_prob_reshape_stride8,face_rpn_landmark_pred_stride16,face_rpn_landmark_pred_stride32,face_rpn_landmark_pred_stride8
cvi_npz_tool.py compare \
      retinaface_res50_out_int8.npz \
      retinaface_res50_caffe_blobs.npz \
      --op_info retinaface_res50_op_info_int8.csv \
      --dequant \
      --tolerance 0.95,0.94,0.68 -vvv

if [ $COMPARE_ALL -eq 1 ]; then
  cvi_npz_tool.py compare \
      retinaface_res50_tensor_all_int8.npz \
      retinaface_res50_caffe_blobs.npz \
      --op_info retinaface_res50_op_info_int8.csv \
      --dequant \
      --tolerance 0.86,0.83,0.49 -vv
fi

# VERDICT
echo $0 PASSED
