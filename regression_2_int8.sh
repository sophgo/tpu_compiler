# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# translate from caffe, apply all possible pre-calibration optimizations
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir
./bin/mlir-opt \
    --convert-bn-to-scale \
    resnet-50.mlir \
    -o resnet-50-opt1.mlir
./bin/mlir-opt \
    --fold-scale \
    resnet-50-opt1.mlir \
    -o resnet-50-opt2.mlir
./bin/mlir-opt \
    --fuse-scale-into-conv \
    resnet-50-opt2.mlir \
    -o resnet-50-opt3.mlir
cp ResNet-50-model.npz ResNet-50-model-opt3.npz

# import calibration table
./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_DIR/bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt3.mlir \
    -o resnet-50-cali.mlir

# quantization 1: per-layer int8
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8.mlir
cp ResNet-50-model.npz ResNet-50-model_quant_int8.npz

./bin/mlir-tpu-interpreter resnet-50-quant-int8.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc1000 out_fc1000.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc1000.bin out_fc1000_int8.bin 1 1 1 1000
diff out_fc1000_int8.bin $DATA_DIR/test_cat_out_fc1000-int8.bin

# quantization 2: per-channel int8
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-per-channel.mlir
cp ResNet-50-model.npz ResNet-50-model_quant_int8_per_channel.npz

./bin/mlir-tpu-interpreter resnet-50-quant-int8-per-channel.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-per-channel.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8-per-channel.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc1000 out_fc1000.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc1000.bin out_fc1000_int8.bin 1 1 1 1000
diff out_fc1000_int8.bin $DATA_DIR/test_cat_out_fc1000-int8-per-channel.bin

# quantization 3: per-channel int8 with multiplier
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir
cp ResNet-50-model.npz ResNet-50-model_quant_int8_multiplier.npz

./bin/mlir-tpu-interpreter resnet-50-quant-int8-multiplier.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8-multiplier.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc1000 out_fc1000.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc1000.bin out_fc1000_int8.bin 1 1 1 1000
diff out_fc1000_int8.bin $DATA_DIR/test_cat_out_fc1000-int8-multiplier.bin

# VERDICT
echo $0 PASSED
