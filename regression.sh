# !/bin/bash

./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir

cp ResNet-50-model.npz ResNet-50-model_bak.npz

./bin/mlir-opt \
    --convert-bn-to-scale \
    resnet-50.mlir \
    -o resnet-50-opt1.mlir

cp ResNet-50-model.npz ResNet-50-model-opt1.npz

./bin/mlir-opt \
    --fold-scale \
    resnet-50-opt1.mlir \
    -o resnet-50-opt2.mlir

cp ResNet-50-model.npz ResNet-50-model-opt2.npz

./bin/mlir-opt \
    --fuse-scale-into-conv \
    resnet-50-opt2.mlir \
    -o resnet-50-opt3.mlir

cp ResNet-50-model.npz ResNet-50-model-opt3.npz

./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt3.mlir \
    -o resnet-50-cali.mlir

./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8.mlir

cp ResNet-50-model.npz ResNet-50-model_quant_int8.npz
cp ResNet-50-model-opt3.npz ResNet-50-model.npz

./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-per-channel.mlir

cp ResNet-50-model.npz ResNet-50-model_quant_int8_per_channel.npz
cp ResNet-50-model-opt3.npz ResNet-50-model.npz

./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir

cp ResNet-50-model.npz ResNet-50-model_quant_int8_multiplier.npz
# cp ResNet-50-model-opt3.npz ResNet-50-model.npz

./bin/mlir-tpu-interpreter resnet-50-quant-int8-multiplier.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out-quant-int8-multiplier.bin \
--dump-all-tensor=tensor_all.npz
