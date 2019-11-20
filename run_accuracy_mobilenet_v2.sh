#~/bin/bash

# copy into build dir to run
# assuming run after regression_1_fp32.sh and regression_2_int8.sh

export PYTHONPATH=./lib:$PYTHONPATH

# gluoncv eval

# pytorch eval
#python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
#--model=mobilenet_v2-quant-bf16.mlir \
#--dataset=/data/dataset/imagenet/img_val_extracted \
#--mean=103.94,116.78,123.68 \
#--input_scale=0.017 \
#--count=$1

python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
--model=mobilenet_v2-quant-int8-multiplier.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean=103.94,116.78,123.68 \
--input_scale=0.017 \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
--model=mobilenet_v2-quant-int8-per-channel.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean=103.94,116.78,123.68 \
--input_scale=0.017 \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
--model=mobilenet_v2-quant-int8.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean=103.94,116.78,123.68 \
--input_scale=0.017 \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
--model=mobilenet_v2-opt3.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean=103.94,116.78,123.68 \
--input_scale=0.017 \
--count=$1

python ../llvm/projects/mlir/bindings/python/tools/eval_imagenet_pytorch.py \
--model=mobilenet_v2.mlir \
--dataset=/data/dataset/imagenet/img_val_extracted \
--mean=103.94,116.78,123.68 \
--input_scale=0.017 \
--count=$1
