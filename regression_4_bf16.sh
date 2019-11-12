# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# translate from caffe
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir

# apply all possible pre-calibration optimizations
./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    --fuse-relu \
    resnet-50.mlir \
    -o resnet-50-opt.mlir

# fp32 inference
./bin/mlir-tpu-interpreter resnet-50-opt.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out.bin  \
    --dump-all-tensor=tensor_all.npz

# quantization
./bin/mlir-opt \
    --quant-bf16 \
    resnet-50-opt.mlir \
    -o resnet-50-quant-bf16.mlir

# bf16 inference
./bin/mlir-tpu-interpreter resnet-50-quant-bf16.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-bf16.bin \
    --dump-all-tensor=tensor_all-bf16.npz
python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
    out.bin out-quant-bf16.bin float32 1 1 1 1000 5

# VERDICT
echo $0 PASSED
