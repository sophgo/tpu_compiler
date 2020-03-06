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
export TOOLCHAIN_FILE_PATH=$MLIR_SRC_PATH/externals/runtime/scripts/toolchain-aarch64-linux.cmake

# install path
export BMKERNEL_SOC_PATH=$INSTALL_SOC_PATH
export RUNTIME_SOC_PATH=$INSTALL_SOC_PATH
export FLATBUFFERS_SOC_PATH=$INSTALL_SOC_PATH/flatbuffers

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
if [ ! -f $INSTALL_PATH/flatbuffers/bin/flatc ]; then
if [ ! -e $BUILD_PATH/build_flatbuffers ]; then
  mkdir -p $BUILD_PATH/build_flatbuffers
fi
pushd $BUILD_PATH/build_flatbuffers
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd
fi

# generate target-independent flatbuffer schema
<<<<<<< HEAD

pushd $BUILD_PATH/build_cvimodel/include
$INSTALL_PATH/flatbuffers/bin/flatc --cpp --gen-object-api \
    $MLIR_SRC_PATH/externals/cvibuilder/src/cvimodel.fbs
popd


if [ ! -e $CVIBUILDER_PATH ]; then
  mkdir $CVIBUILDER_PATH
  mkdir $CVIBUILDER_PATH/include
if [ ! -f $INSTALL_PATH/include/cvibuilder/cvimodel_generated.h ]; then
if [ ! -e $BUILD_PATH/build_cvimodel ]; then
  mkdir -p $BUILD_PATH/build_cvimodel
fi
pushd $BUILD_PATH/build_cvimodel
cmake -G Ninja -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    $MLIR_SRC_PATH/externals/cvibuilder
cmake --build . --target install
popd
fi

# build target flat buffer
if [ ! -e $BUILD_PATH/build_flatbuffers_soc ]; then
  mkdir -p $BUILD_PATH/build_flatbuffers_soc
fi
pushd $BUILD_PATH/build_flatbuffers_soc
# CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_SOC_PATH \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd

<<<<<<< HEAD
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

=======
# build bmkernel
if [ ! -e $BUILD_SOC_PATH/build_bmkernel ]; then
  mkdir -p $BUILD_SOC_PATH/build_bmkernel
fi
pushd $BUILD_SOC_PATH/build_bmkernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCMAKE_INSTALL_PREFIX=$BMKERNEL_SOC_PATH \
    $MLIR_SRC_PATH/externals/bmkernel
cmake --build . --target install
popd

# build runtime
if [ ! -e $BUILD_SOC_PATH/build_runtime ]; then
  mkdir $BUILD_SOC_PATH/build_runtime
fi
pushd $BUILD_SOC_PATH/build_runtime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=SOC $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DBMKERNEL_PATH=$CVIKERNEL_SOC_PATH \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_SOC_PATH \
    -DCVIBUILDER_PATH=$BUILD_PATH/build_cvimodel \
    -DCMAKE_INSTALL_PREFIX=$RUNTIME_SOC_PATH \
    $MLIR_SRC_PATH/externals/runtime
cmake --build . --target install
popd

