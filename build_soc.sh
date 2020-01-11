#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh


# build bmkernel
if [ ! -e $MLIR_SRC_PATH/externals/bmkernel/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/bmkernel/build_soc
fi
pushd $MLIR_SRC_PATH/externals/bmkernel/build_soc
cmake -DCHIP=BM1880v2 -DCMAKE_TOOLCHAIN_FILE=/mnt2/mlir_0106/llvm-project/llvm/projects/mlir/toolchain-aarch64-linux.cmake -DCMAKE_INSTALL_PREFIX=$BMKERNEL_PATH ..
cmake --build . --target install
popd

# build support
if [ ! -e $MLIR_SRC_PATH/externals/support/build_soc ]; then
  mkdir $MLIR_SRC_PATH/externals/support/build_soc
fi
pushd $MLIR_SRC_PATH/externals/support/build_soc
cmake DCHIP=BM1880v2 -DCMAKE_TOOLCHAIN_FILE=/mnt2/mlir_0106/llvm-project/llvm/projects/mlir/toolchain-aarch64-linux.cmake -DCMAKE_INSTALL_PREFIX=$SUPPORT_PATH ..
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
-DCMAKE_TOOLCHAIN_FILE=/mnt2/mlir_0106/llvm-project/llvm/projects/mlir/toolchain-aarch64-linux.cmake \
    -DSUPPORT_PATH=$SUPPORT_PATH \
    -DBMKERNEL_PATH=$BMKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH ..
cmake --build . --target install
popd
