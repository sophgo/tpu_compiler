#!/bin/bash
set -e
#
# Download toolchian from
# https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/
#
# Assuming toolchain setup in $TPU_BASE/tools
#   $ cd $TPU_BASE/tools
#
# Unzip gcc & runtime
#   $ tar xf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
#   $ tar xf sysroot-glibc-linaro-2.23-2017.05-aarch64-linux-gnu.tar.xz
#
# Create link
#   $ ln -s gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu gcc_aarch64-linux-gnu
#   $ ln -s sysroot-glibc-linaro-2.23-2017.05-aarch64-linux-gnu sysroot_aarch64-linux-gnu
#

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

export ARM_TOOLCHAIN_GCC_PATH=$TPU_BASE/tools/gcc_aarch64-linux-gnu
export ARM_TOOLCHAIN_SYSROOT_PATH=$TPU_BASE/tools/sysroot_aarch64-linux-gnu

export PATH=$ARM_TOOLCHAIN_GCC_PATH/bin:$PATH
export AARCH64_SYSROOT_PATH=$ARM_TOOLCHAIN_SYSROOT_PATH
export RAMDISK_PATH=$TPU_BASE/ramdisk/prebuild
export TOOLCHAIN_FILE_PATH=$MLIR_SRC_PATH/externals/cviruntime/scripts/toolchain-aarch64-linux.cmake

# install path
export FLATBUFFERS_SOC_PATH=$INSTALL_SOC_PATH/flatbuffers
export CVIKERNEL_SOC_PATH=$INSTALL_SOC_PATH/cvikernel
export CVIRUNTIME_SOC_PATH=$INSTALL_SOC_PATH

# mkdir
if [ ! -e $INSTALL_SOC_PATH ]; then
  mkdir -p $INSTALL_SOC_PATH
fi
if [ ! -e $BUILD_SOC_PATH ]; then
  mkdir -p $BUILD_SOC_PATH
fi
if [ ! -e $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi
if [ ! -e $BUILD_PATH ]; then
  mkdir -p $BUILD_PATH
fi

# build host flatbuffers
if [ ! -e $BUILD_PATH/build_flatbuffers ]; then
  mkdir -p $BUILD_PATH/build_flatbuffers
fi
pushd $BUILD_PATH/build_flatbuffers
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd

# build target flat buffer
if [ ! -e $BUILD_SOC_PATH/build_flatbuffers ]; then
  mkdir -p $BUILD_SOC_PATH/build_flatbuffers
fi
pushd $BUILD_SOC_PATH/build_flatbuffers
# CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_SOC_PATH \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd

# generate target-independent flatbuffer schema
if [ ! -e $BUILD_SOC_PATH/build_cvimodel ]; then
  mkdir -p $BUILD_SOC_PATH/build_cvimodel
fi
pushd $BUILD_SOC_PATH/build_cvimodel
cmake -G Ninja -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    $MLIR_SRC_PATH/externals/cvibuilder
cmake --build . --target install
popd

# build cvikernel
if [ ! -e $BUILD_SOC_PATH/build_cvikernel ]; then
  mkdir -p $BUILD_SOC_PATH/build_cvikernel
fi
pushd $BUILD_SOC_PATH/build_cvikernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCMAKE_INSTALL_PREFIX=$CVIKERNEL_SOC_PATH \
    $MLIR_SRC_PATH/externals/cvikernel
cmake --build . --target install
popd

# build cnpy
if [ ! -e $BUILD_SOC_PATH/build_cnpy ]; then
  mkdir -p $BUILD_SOC_PATH/build_cnpy
fi
pushd $BUILD_SOC_PATH/build_cnpy
cmake -G Ninja $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_SOC_PATH \
    -DRAMDISK_PATH=${RAMDISK_PATH} \
    $MLIR_SRC_PATH/third_party/cnpy
cmake --build . --target install
popd

# build runtime
if [ ! -e $BUILD_SOC_PATH/build_cviruntime ]; then
  mkdir $BUILD_SOC_PATH/build_cviruntime
fi
pushd $BUILD_SOC_PATH/build_cviruntime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=SOC $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCVIKERNEL_PATH=$CVIKERNEL_SOC_PATH \
    -DCNPY_PATH=$INSTALL_SOC_PATH/lib \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_SOC_PATH \
    -DCVIBUILDER_PATH=$BUILD_SOC_PATH/build_cvimodel \
    -DCMAKE_INSTALL_PREFIX=$CVIRUNTIME_SOC_PATH \
    -DRAMDISK_PATH=${RAMDISK_PATH} \
    $MLIR_SRC_PATH/externals/cviruntime
cmake --build . --target install
popd
