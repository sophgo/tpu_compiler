#!/bin/bash

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

pushd $PYTHON_TOOLS_PATH/dataset_util/widerface
make clean
popd
pushd $PYTHON_TOOLS_PATH/model/retinaface
make clean
popd

rm -rf $CAFFE_PATH
rm -rf $MKLDNN_PATH
rm -rf $BMKERNEL_PATH
rm -rf $CMODEL_PATH
rm -rf $SUPPORT_PATH
rm -rf $CVIBUILDER_PATH
rm -rf $RUNTIME_PATH
rm -rf $MLIR_PATH

rm -rf $MLIR_SRC_PATH/third_party/caffe/build
rm -rf $MLIR_SRC_PATH/externals/bmkernel/build
rm -rf $MLIR_SRC_PATH/externals/cmodel/build
rm -rf $MLIR_SRC_PATH/externals/support/build
# rm -rf $MLIR_SRC_PATH/externals/cvibuilder/build
rm -rf $MLIR_SRC_PATH/externals/runtime/build

rm -rf $TPU_BASE/llvm-project/build
