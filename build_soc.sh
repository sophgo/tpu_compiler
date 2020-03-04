#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

# modify install path to install_soc
export CVIKERNEL_PATH=$TPU_BASE/install_soc_cvikernel
export SUPPORT_PATH=$TPU_BASE/install_soc_support
export RUNTIME_PATH=$TPU_BASE/install_soc_cviruntime

export TOOLCHAIN_FILE_PATH=$TPU_BASE/llvm-project/llvm/projects/mlir/externals/runtime/scripts/toolchain-aarch64-linux.cmake

# build host flatbuffers
if [ ! -e $MLIR_SRC_PATH/third_party/flatbuffers/build ]; then
  mkdir $MLIR_SRC_PATH/third_party/flatbuffers/build
fi
pushd $MLIR_SRC_PATH/third_party/flatbuffers/build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH ..
cmake --build . --target install
popd
pushd $MLIR_SRC_PATH/third_party/flatbuffers
cp -a python $FLATBUFFERS_PATH/
popd

# generate target-independent flatbuffer schema

pushd $BUILD_PATH/build_cvimodel/include
$INSTALL_PATH/flatbuffers/bin/flatc --cpp --gen-object-api \
    $MLIR_SRC_PATH/externals/cvibuilder/src/cvimodel.fbs
popd


if [ ! -e $CVIBUILDER_PATH ]; then
  mkdir $CVIBUILDER_PATH
  mkdir $CVIBUILDER_PATH/include
fi
pushd $CVIBUILDER_PATH/include
flatc --cpp --gen-object-api $MLIR_SRC_PATH/externals/cvibuilder/src/cvimodel.fbs
popd
pushd $MLIR_SRC_PATH/externals/cvibuilder
cp -a python $CVIBUILDER_PATH/
popd

# build target flat buffer
# not build test since it will trigger self test
export FLATBUFFERS_PATH=$TPU_BASE/install_soc_flatbuffers
if [ ! -e $MLIR_SRC_PATH/third_party/flatbuffers/build_soc ]; then
  mkdir $MLIR_SRC_PATH/third_party/flatbuffers/build_soc
fi
pushd $MLIR_SRC_PATH/third_party/flatbuffers/build_soc
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH -DFLATBUFFERS_BUILD_TESTS=OFF ..
cmake --build . --target install
popd

# build cvikernel
if [ ! -e $MLIR_SRC_PATH/externals/cvikernel/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/cvikernel/build_soc
fi
pushd $MLIR_SRC_PATH/externals/cvikernel/build_soc
cmake -DCHIP=BM1880v2 -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH -DCMAKE_INSTALL_PREFIX=$CVIKERNEL_PATH ..
cmake --build . --target install
popd

# build support
if [ ! -e $MLIR_SRC_PATH/externals/support/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/support/build_soc
fi
pushd $MLIR_SRC_PATH/externals/support/build_soc
cmake DCHIP=BM1880v2 -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH -DCMAKE_INSTALL_PREFIX=$SUPPORT_PATH ..
cmake --build . --target install
popd

# build cmodel
#if [ ! -e $MLIR_SRC_PATH/externals/cmodel/build ]; then
#  mkdir $MLIR_SRC_PATH/externals/cmodel/build
#fi
#pushd $MLIR_SRC_PATH/externals/cmodel/build
#cmake -DCHIP=BM1880v2 -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
#    -DSUPPORT_PATH=$SUPPORT_PATH -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH ..
#cmake --build . --target install
#popd

# build runtime
if [ ! -e $MLIR_SRC_PATH/externals/runtime/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/runtime/build_soc
fi
pushd $MLIR_SRC_PATH/externals/runtime/build_soc
cmake -DCHIP=BM1880v2 -DRUNTIME=SOC \
	-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
	-DSUPPORT_PATH=$SUPPORT_PATH \
	-DCVIKERNEL_PATH=$CVIKERNEL_PATH \
        -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH -DCVIBUILDER_PATH=$CVIBUILDER_PATH \
	-DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH ..
cmake --build . --target install
popd
