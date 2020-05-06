#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ -z $BUILD_OPENCV ]; then
  export BUILD_OPENCV=0
else
  export BUILD_OPENCV=$BUILD_OPENCV
fi
if [ -z $BUILD_SAMPLES ]; then
  export BUILD_SAMPLES=0
else
  export BUILD_SAMPLES=$BUILD_SAMPLES
fi

#
# mkdir
#
if [ ! -e $INSTALL_SOC_PATH_32BIT ]; then
  mkdir -p $INSTALL_SOC_PATH_32BIT
fi
if [ ! -e $BUILD_SOC_PATH_32BIT ]; then
  mkdir -p $BUILD_SOC_PATH_32BIT
fi
if [ ! -e $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi
if [ ! -e $BUILD_PATH ]; then
  mkdir -p $BUILD_PATH
fi

#
# Setup Toolchain
#
if [[ -z "$ARM_TOOLCHAIN_GCC_PATH" ]]; then
  if [[ -z "$HOST_TOOL_PATH" ]]; then
    ARM_TOOLCHAIN_GCC_PATH=$TPU_BASE/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf
  else
    ARM_TOOLCHAIN_GCC_PATH=$HOST_TOOL_PATH/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf
  fi
fi
if [ ! -e "$ARM_TOOLCHAIN_GCC_PATH" ]; then
  echo "ARM_TOOLCHAIN_GCC_PATH not exist"
  return 1
fi
export ARM_TOOLCHAIN_GCC_PATH=$ARM_TOOLCHAIN_GCC_PATH

if [[ -z "$AARCH32_SYSROOT_PATH" ]]; then
  if [[ -z "$RAMDISK_PATH" ]]; then
    AARCH32_SYSROOT_PATH=$TPU_BASE/ramdisk/sysroot/sysroot-glibc-linaro-2.23-2017.05-arm-linux-gnueabihf
  else
    AARCH32_SYSROOT_PATH=$RAMDISK_PATH/sysroot/sysroot-glibc-linaro-2.23-2017.05-arm-linux-gnueabihf
  fi
fi
if [ ! -e "$AARCH32_SYSROOT_PATH" ]; then
  echo "AARCH32_SYSROOT_PATH not exist"
  return 1
fi



#ARM_TOOLCHAIN_GCC_PATH=$TPU_BASE/host-tools/gcc/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf
#AARCH32_SYSROOT_PATH=$TPU_BASE/ramdisk/sysroot/sysroot-glibc-linaro-2.23-2017.05-arm-linux-gnueabihf

export AARCH_SYSROOT_PATH=$AARCH32_SYSROOT_PATH
export TOOLCHAIN_FILE_PATH=$MLIR_SRC_PATH/externals/cviruntime/scripts/toolchain-linux-gnueabihf.cmake
export PATH=$ARM_TOOLCHAIN_GCC_PATH/bin:$PATH

#
# install path
#
export FLATBUFFERS_SOC_PATH=$INSTALL_SOC_PATH_32BIT/flatbuffers
export CVIKERNEL_SOC_PATH=$INSTALL_SOC_PATH_32BIT
export CVIRUNTIME_SOC_PATH=$INSTALL_SOC_PATH_32BIT

#
# build
#
BUILD_TYPE="RELEASE"
if [ "$BUILD_TYPE" == "RELEASE" ]; then
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3"
else
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"
  export BUILD_PATH=${BUILD_PATH}_debug
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
if [ ! -e $BUILD_SOC_PATH_32BIT/build_flatbuffers ]; then
  mkdir -p $BUILD_SOC_PATH_32BIT/build_flatbuffers
fi
pushd $BUILD_SOC_PATH_32BIT/build_flatbuffers
# CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
cmake -G Ninja $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_SOC_PATH \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd

# generate target-independent flatbuffer schema
if [ ! -e $BUILD_SOC_PATH_32BIT/build_cvimodel ]; then
  mkdir -p $BUILD_SOC_PATH_32BIT/build_cvimodel
fi
pushd $BUILD_SOC_PATH_32BIT/build_cvimodel
cmake -G Ninja -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    $MLIR_SRC_PATH/externals/cvibuilder
cmake --build . --target install
popd

# build cvikernel
if [ ! -e $BUILD_SOC_PATH_32BIT/build_cvikernel ]; then
  mkdir -p $BUILD_SOC_PATH_32BIT/build_cvikernel
fi
pushd $BUILD_SOC_PATH_32BIT/build_cvikernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCMAKE_INSTALL_PREFIX=$CVIKERNEL_SOC_PATH \
    $MLIR_SRC_PATH/externals/cvikernel
cmake --build . --target install -- -v
popd

# build cnpy
if [ ! -e $BUILD_SOC_PATH_32BIT/build_cnpy ]; then
  mkdir -p $BUILD_SOC_PATH_32BIT/build_cnpy
fi
pushd $BUILD_SOC_PATH_32BIT/build_cnpy
cmake -G Ninja $BUILD_FLAG \
    -DCMAKE_SYSROOT=$AARCH_SYSROOT_PATH \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_SOC_PATH_32BIT \
    $MLIR_SRC_PATH/third_party/cnpy
cmake --build . --target install
popd

# build runtime
if [ ! -e $BUILD_SOC_PATH_32BIT/build_cviruntime ]; then
  mkdir $BUILD_SOC_PATH_32BIT/build_cviruntime
fi

pushd $BUILD_SOC_PATH_32BIT/build_cviruntime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=SOC $BUILD_FLAG \
    -DCMAKE_SYSROOT=$AARCH_SYSROOT_PATH \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
    -DCVIKERNEL_PATH=$CVIKERNEL_SOC_PATH \
    -DCNPY_PATH=$INSTALL_SOC_PATH_32BIT/lib \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_SOC_PATH \
    -DCVIBUILDER_PATH=$BUILD_SOC_PATH_32BIT/build_cvimodel \
    -DCMAKE_INSTALL_PREFIX=$CVIRUNTIME_SOC_PATH \
    -DENABLE_TEST=OFF \
    $MLIR_SRC_PATH/externals/cviruntime
cmake --build . --target install -- -v
popd

export OPENCV_SOC_PATH=$INSTALL_SOC_PATH_32BIT/opencv
if [ $BUILD_OPENCV -eq 1 ]; then
  # build opencv
  # clone opencv to $TPU_BASE/opencv
  # checkout tag 3.2.0
  if [ ! -e $BUILD_SOC_PATH_32BIT/build_opencv ]; then
    mkdir $BUILD_SOC_PATH_32BIT/build_opencv
  fi
  pushd $BUILD_SOC_PATH_32BIT/build_opencv
  cmake -G Ninja $BUILD_FLAG \
      -DWITH_CUDA=OFF -DWITH_DC1394=OFF -DWITH_GPHOTO2=OFF \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_SYSROOT=$AARCH_SYSROOT_PATH \
      -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
      -DBUILD_opencv_videoio=OFF -DBUILD_opencv_highgui=OFF \
      -DBUILD_opencv_superres=OFF -DBUILD_opencv_videostab=OFF \
      -DBUILD_opencv_stitching=OFF -DBUILD_opencv_objdetect=OFF \
      -DBUILD_opencv_calib3d=OFF -DBUILD_opencv_ml=OFF \
      -DBUILD_opencv_video=OFF -DBUILD_opencv_flann=OFF \
      -DBUILD_opencv_photo=OFF \
      -DCMAKE_INSTALL_PREFIX=$OPENCV_SOC_PATH \
      $TPU_BASE/opencv
  cmake --build . --target install
  popd
fi

export SAMPLES_SOC_PATH=$INSTALL_SOC_PATH_32BIT/samples
if [ $BUILD_SAMPLES -eq 1 ]; then
  if [ ! -e $BUILD_SOC_PATH_32BIT/build_samples ]; then
    mkdir $BUILD_SOC_PATH_32BIT/build_samples
  fi
  pushd $BUILD_SOC_PATH_32BIT/build_samples
  cmake -G Ninja $BUILD_FLAG \
      -DCMAKE_SYSROOT=$AARCH_SYSROOT_PATH \
      -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH \
      -DTPU_SDK_PATH=$INSTALL_SOC_PATH_32BIT \
      -DOPENCV_PATH=$OPENCV_SOC_PATH \
      -DCMAKE_INSTALL_PREFIX=$SAMPLES_SOC_PATH \
      $MLIR_SRC_PATH/externals/cviruntime/samples
  cmake --build . --target install -- -v
  popd
fi

# Copy some files for release build
mkdir -p $INSTALL_SOC_PATH_32BIT/cmake
cp $TOOLCHAIN_FILE_PATH $INSTALL_SOC_PATH_32BIT/cmake
cp $AARCH32_SYSROOT_PATH/lib/libglog.so.0.4.0 $INSTALL_SOC_PATH_32BIT/lib
pushd $INSTALL_SOC_PATH_32BIT/lib
ln -nsf libglog.so.0.4.0 libglog.so.0.0
ln -nsf libglog.so.0.4.0 libglog.so.0
ln -nsf libglog.so.0.4.0 libglog.so
popd
