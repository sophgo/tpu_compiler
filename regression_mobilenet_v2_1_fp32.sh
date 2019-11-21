# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# translate from caffe model
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel /data/models/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# test mlir interpreter
./bin/mlir-tpu-interpreter mobilenet_v2.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out.bin
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    $DATA_DIR/test_cat_out_mobilenet_v2_fp32.bin out.bin float32 1 1 1 1000 5 5

# opt1, convert bn to scale
./bin/mlir-opt \
    --convert-bn-to-scale \
    mobilenet_v2.mlir \
    -o mobilenet_v2-opt1.mlir

# test opt1
./bin/mlir-tpu-interpreter mobilenet_v2-opt1.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-opt1.bin
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    out.bin out-opt1.bin float32 1 1 1 1000 5 5

# opt2, fold consecutive scales
./bin/mlir-opt \
    --fold-scale \
    mobilenet_v2-opt1.mlir \
    -o mobilenet_v2-opt2.mlir

# test opt2
./bin/mlir-tpu-interpreter mobilenet_v2-opt2.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-opt2.bin
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    out.bin out-opt2.bin float32 1 1 1 1000 5 5

# opt3, merge scale into conv
./bin/mlir-opt \
    --fuse-scale-into-conv \
    mobilenet_v2-opt2.mlir \
    -o mobilenet_v2-opt3.mlir

# test opt3
./bin/mlir-tpu-interpreter mobilenet_v2-opt3.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-opt3.bin
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    out.bin out-opt3.bin float32 1 1 1 1000 5 5

# opt4, fuse relu into conv
./bin/mlir-opt \
    --fuse-relu \
    mobilenet_v2-opt3.mlir \
    -o mobilenet_v2-opt4.mlir

# test opt4
./bin/mlir-tpu-interpreter mobilenet_v2-opt4.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-opt4.bin
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    out.bin out-opt4.bin float32 1 1 1 1000 5 5

# VERDICT
echo $0 PASSED