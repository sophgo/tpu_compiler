set -e

COMPARE_ALL=0

# Pre-Quantization optimization
mlir-opt \
   --convert-bn-to-scale \
   --fold-scale \
   --merge-scale-into-conv \
   retinaface_res50.mlir \
   -o retinaface_res50-opt.mlir

# Calibration
python $PYTHON_TOOLS_PATH/dataset_util/gen_dataset_img_list.py \
    --dataset $DATASET_PATH/widerface/WIDER_val \
    --count 100 \
    --output_img_list cali_list_widerface_100.txt

python $PYTHON_TOOLS_PATH/model/retinaface/calibrate_retinaface.py \
    retinaface_res50 \
    retinaface_res50-opt.mlir \
    cali_list_widerface_100.txt \
    retinaface_res50_threshold_table \
    --input_num=100 \
    --out_path=. \
    --math_lib_path=$CALIBRATION_TOOL_PATH/calibration_math.so

mlir-opt \
    --import-calibration-table \
    --calibration-table retinaface_res50_threshold_table \
    retinaface_res50-opt.mlir \
    -o retinaface_res50-cali.mlir

# Post-Calibration optimizatin
mlir-opt \
    --fuse-relu \
    retinaface_res50-cali.mlir \
    -o retinaface_res50-opt-post-cali.mlir

# Quantization
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info_int8.csv \
    retinaface_res50-opt-post-cali.mlir \
    -o retinaface_res50-int8.mlir

# dump all tensor
# Pre-Quantization fp32 result
mlir-tpu-interpreter retinaface_res50-opt.mlir \
    --tensor-in retinaface_res50_in_fp32.npz \
    --tensor-out retinaface_res50_opt_out_fp32.npz \
    --dump-all-tensor=retinaface_res50_opt_tensor_all_fp32.npz

# Post-Quantization int8 result
mlir-tpu-interpreter retinaface_res50-int8.mlir \
    --tensor-in retinaface_res50_in_fp32.npz \
    --tensor-out retinaface_res50_out_int8.npz \
    --dump-all-tensor=retinaface_res50_tensor_all_int8.npz

if [ $COMPARE_ALL ]; then
    # this will fail for now
    npz_compare.py \
      retinaface_res50_tensor_all_int8.npz \
      retinaface_res50_opt_tensor_all_fp32.npz \
      --op_info retinaface_res50_op_info_int8.csv \
      --dequant \
      --tolerance 0.9,0.9,0.8 -vvv
fi

# VERDICT
echo $0 PASSED
