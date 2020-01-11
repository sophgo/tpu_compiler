#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/yolov3/416/yolov3_416.prototxt \
    --caffemodel $MODEL_PATH/caffe/yolov3/416/yolov3_416.caffemodel \
    -o yolo_v3_416.mlir

# test mlir interpreter
mlir-tpu-interpreter yolo_v3_416.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz
bin_compare.py out.bin $DATA_PATH/test_dot_out_yolo_v3_416_fp32.bin \
    float32 1 1 1 689520

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    yolo_v3_416.mlir \
    -o yolo_v3_416_opt.mlir

# test opt
mlir-tpu-interpreter yolo_v3_416_opt.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out_opt.bin \
    --dump-all-tensor=tensor_all_fp32.npz
bin_compare.py out_opt.bin out.bin float32 1 1 1 689520

# calibration
python ../llvm/projects/mlir/externals/calibration_tool/run_calibration.py \
    yolo_v3 yolo_v3_416_opt.mlir \
    $DATA_PATH/input_coco_100.txt \
    --input_num=100

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali.mlir

################################
# quantization 1: per-layer int8
################################
mlir-opt \
    --quant-int8 \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_per_layer.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_per_layer.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out_int8_per_layer.bin \
    --dump-all-tensor=tensor_all_int8_per_layer.npz

################################
# quantization 2: per-channel int8
################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_per_channel.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_per_channel.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out_int8_per_channel.bin \
    --dump-all-tensor=tensor_all_int8_per_channel.npz

################################
# quantization 3: per-channel multiplier int8
################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_multiplier.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_multiplier.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out_int8_multiplier.bin \
    --dump-all-tensor=tensor_all_int8_multiplier.npz

# VERDICT
echo $0 PASSED
