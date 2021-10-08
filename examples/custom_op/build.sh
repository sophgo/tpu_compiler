#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

export PYTHONPATH=$DIR/code/python:$PYTHONPATH

rm -rf build && mkdir -p build
pushd build

# build library
MLIR_INCLUDE=$MLIR_PATH/tpuc/include
CNPY_INCLUDE=$MLIR_PATH/cnpy/include

BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"

cmake -G Ninja \
      ${BUILD_FLAG} \
      -DMLIR_INCLUDE=$MLIR_INCLUDE \
      -DCNPY_INCLUDE=$CNPY_INCLUDE \
      -DCMAKE_INSTALL_PREFIX=lib \
      $DIR/code
cmake --build . --target install

# convert model

$DIR/model/mymodel.py \
    --model_path $DIR/model/fake.prototxt \
    --model_dat $DIR/model/fake.caffemodel \
    --mlir_file_path fake.mlir

# bf16
model_deploy.py \
    --model_name fake \
    --mlir fake.mlir \
    --all_bf16 \
    --chip cv183x \
    --image input.npz \
    --tolerance 0.9,0.9,0.7 \
    --custom_op_plugin lib/libCustomOpPlugin.so \
    --debug \
    --cvimodel fake_bf16.cvimodel

popd