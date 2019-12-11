#~/bin/bash

# copy into build dir to run
# assuming run after regression_1_fp32.sh and regression_2_int8.sh

export PYTHONPATH=./lib:$PYTHONPATH

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-bf16.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8-multiplier.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8-per-channel.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-quant-int8.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=resnet-50-opt3.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
--model=mobilenet_v2.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
--input_scale=0.017 \
--count=$1
