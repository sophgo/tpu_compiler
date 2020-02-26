#!/bin/bash
set -e

COMPARE_ALL=1
DO_CALIBRATION=0

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  python $PYTHON_TOOLS_PATH/dataset_util/gen_dataset_img_list.py \
      --dataset $DATASET_PATH/widerface/WIDER_val \
      --count 300 \
      --output_img_list cali_list_widerface_100.txt

  python $PYTHON_TOOLS_PATH/model/retinaface/calibrate_retinaface.py \
      retinaface_mnet25 \
      retinaface_mnet25_opt.mlir \
      cali_list_widerface_100.txt \
      retinaface_mnet25_threshold_table \
      --net_input_dims=320,320 \
      --input_num=300 \
      --histogram_bin_num=65536 \
      --out_path=. \
      --math_lib_path=$CALIBRATION_TOOL_PATH/calibration_math.so
else
  cp $REGRESSION_PATH/retinaface_mnet25/data/retinaface_mnet25_threshold_table .
fi

mlir-opt \
    --import-calibration-table \
    --calibration-table retinaface_mnet25_threshold_table \
    retinaface_mnet25_opt.mlir \
    -o retinaface_mnet25_cali.mlir

# Quantization
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_mnet25_op_info_int8.csv \
    retinaface_mnet25_cali.mlir \
    -o retinaface_mnet25_int8.mlir

# Interpreter int8 result
mlir-tpu-interpreter retinaface_mnet25_int8.mlir \
    --tensor-in retinaface_mnet25_in_fp32.npz \
    --tensor-out retinaface_mnet25_out_dequant_int8.npz \
    --dump-all-tensor=retinaface_mnet25_tensor_all_int8.npz

# compare output
npz_extract.py \
    retinaface_mnet25_tensor_all_int8.npz \
    retinaface_mnet25_out_int8.npz \
    face_rpn_bbox_pred_stride16,face_rpn_bbox_pred_stride32,face_rpn_bbox_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_cls_prob_reshape_stride8,face_rpn_landmark_pred_stride16,face_rpn_landmark_pred_stride32,face_rpn_landmark_pred_stride8
npz_compare.py \
      retinaface_mnet25_out_int8.npz \
      retinaface_mnet25_caffe_blobs.npz \
      --op_info retinaface_mnet25_op_info_int8.csv \
      --dequant \
      --tolerance 0.97,0.97,0.78 -vvv

if [ $COMPARE_ALL -eq 1 ]; then
  npz_compare.py \
      retinaface_mnet25_tensor_all_int8.npz \
      retinaface_mnet25_caffe_blobs.npz \
      --op_info retinaface_mnet25_op_info_int8.csv \
      --dequant \
      --tolerance 0.94,0.92,0.65 -vvv
fi

# VERDICT
echo $0 PASSED
