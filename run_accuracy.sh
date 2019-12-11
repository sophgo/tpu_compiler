#~/bin/bash

# copy into build dir to run

export PYTHONPATH=./lib:$PYTHONPATH

cp ResNet-50-model_quant_int8_multiplier.npz ResNet-50-model.npz

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8-multiplier.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

cp ResNet-50-model_quant_int8_per_channel.npz ResNet-50-model.npz

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8-per-channel.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

cp ResNet-50-model_quant_int8.npz ResNet-50-model.npz

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

cp ResNet-50-model_bak.npz ResNet-50-model.npz

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

