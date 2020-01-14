#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

# modify install path to install_soc
export BMKERNEL_PATH=$TPU_BASE/install_soc_bmkernel
export SUPPORT_PATH=$TPU_BASE/install_soc_support
export RUNTIME_PATH=$TPU_BASE/install_soc_runtime

export TOOLCHAIN_FILE_PATH=$TPU_BASE/llvm-project/llvm/projects/mlir/externals/runtime/scripts/toolchain-aarch64-linux.cmake

# build bmkernel
if [ ! -e $MLIR_SRC_PATH/externals/bmkernel/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/bmkernel/build_soc
fi
pushd $MLIR_SRC_PATH/externals/bmkernel/build_soc
cmake -DCHIP=BM1880v2 -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH -DCMAKE_INSTALL_PREFIX=$BMKERNEL_PATH ..
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
#cmake -DCHIP=BM1880v2 -DBMKERNEL_PATH=$BMKERNEL_PATH \
#    -DSUPPORT_PATH=$SUPPORT_PATH -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH ..
#cmake --build . --target install
#popd

# build bmbuilder
#if [ ! -e $MLIR_SRC_PATH/externals/bmbuilder/build_soc ]; then
#  mkdir $MLIR_SRC_PATH/externals/bmbuilder/build_soc
#fi
#pushd $MLIR_SRC_PATH/externals/bmbuilder/build_soc
#cmake -DBMKERNEL_PATH=$BMKERNEL_PATH \
#    -DCMAKE_INSTALL_PREFIX=$BMBUILDER_PATH ..
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
	-DBMKERNEL_PATH=$BMKERNEL_PATH \
	-DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH ..
cmake --build . --target install
popd
