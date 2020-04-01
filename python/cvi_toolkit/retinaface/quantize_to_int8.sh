MLIR_OPT=/workspace/llvm-project/build/bin/mlir-opt

# Pre-Quantization optimization
$MLIR_OPT \
   --convert-bn-to-scale \
   --fold-scale \
   --merge-scale-into-conv \
   retinaface_res50.mlir \
   -o retinaface_res50-opt.mlir

# Calibration
echo "/workspace/llvm-project/llvm/projects/mlir/externals/python_tools/model/retinaface/probe.jpg" > calibration_input.txt
cp retinaface_res50-opt.mlir $CALIBRATION_TOOL_PATH
cp R50*.npz $CALIBRATION_TOOL_PATH
cp calibrate_retinaface.py $CALIBRATION_TOOL_PATH

pushd $CALIBRATION_TOOL_PATH
python calibrate_retinaface.py \
    retinaface_res50 \
    $CALIBRATION_TOOL_PATH/retinaface_res50-opt.mlir \
    /workspace/llvm-project/llvm/projects/mlir/externals/python_tools/model/retinaface/calibration_input.txt \
    --out_path=/workspace/llvm-project/llvm/projects/mlir/externals/python_tools/model/retinaface \
    --math_lib_path=$CALIBRATION_TOOL_PATH/calibration_math.so

rm retinaface_res50-opt.mlir
rm R50*.npz
rm calibrate_retinaface.py
popd

$MLIR_OPT \
    --import-calibration-table \
    --calibration-table retinaface_res50_threshold_table \
    retinaface_res50-opt.mlir \
    -o retinaface_res50-cali.mlir

# Post-Calibration optimizatin
$MLIR_OPT \
    --fuse-relu \
    retinaface_res50-cali.mlir \
    -o retinaface_res50-opt-post-cali.mlir

# Quantization
$MLIR_OPT \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    retinaface_res50-opt-post-cali.mlir \
    -o retinaface_res50-int8.mlir
