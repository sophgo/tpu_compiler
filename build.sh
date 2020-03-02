#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

BUILD_TYPE="RELEASE"
if [ "$BUILD_TYPE" == "RELEASE" ]; then
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3"
else
  BUILD_FLAG=""
fi

# mkdir
if [ ! -e $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi

if [ ! -e $TPU_PYTHON_PATH ]; then
  mkdir -p $TPU_PYTHON_PATH
fi

if [ ! -e $BUILD_PATH ]; then
  mkdir -p $BUILD_PATH
fi

# download and unzip mkldnn
if [ ! -e $MKLDNN_PATH ]; then
  wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
  tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
  mkdir -p $MKLDNN_PATH
  mv mkldnn_lnx_1.0.2_cpu_gomp/* $MKLDNN_PATH/
  rm mkldnn_lnx_1.0.2_cpu_gomp.tgz
  rm -rf mkldnn_lnx_1.0.2_cpu_gomp
fi

# build caffe
if [ ! -e $BUILD_PATH/build_caffe ]; then
  mkdir -p $BUILD_PATH/build_caffe
fi
pushd $BUILD_PATH/build_caffe
cmake -G Ninja -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH \
    $MLIR_SRC_PATH/third_party/caffe
cmake --build . --target install
popd

# build flatbuffers
if [ ! -e $BUILD_PATH/build_flatbuffers ]; then
  mkdir -p $BUILD_PATH/build_flatbuffers
fi
pushd $BUILD_PATH/build_flatbuffers
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH \
    $MLIR_SRC_PATH/third_party/flatbuffers
cmake --build . --target install
popd
cp -a $MLIR_SRC_PATH/third_party/flatbuffers/python $FLATBUFFERS_PATH/

# build bmkernel
if [ ! -e $BUILD_PATH/build_bmkernel ]; then
  mkdir -p $BUILD_PATH/build_bmkernel
fi
pushd $BUILD_PATH/build_bmkernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$BMKERNEL_PATH \
    $MLIR_SRC_PATH/externals/bmkernel
cmake --build . --target install
popd

# build mlir-tpu
pushd $BUILD_PATH
cmake -G Ninja -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON \
    $BUILD_FLAG \
    -DMKLDNN_PATH=$MKLDNN_PATH \
    -DCAFFE_PATH=$CAFFE_PATH \
    -DBMKERNEL_PATH=$BMKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$MLIR_PATH \
    $TPU_BASE/llvm-project/llvm
cmake --build . --target check-mlir
# cmake --build . --target pymlir
# cmake --build . --target pybind
cmake --build . --target install
popd
cp $MLIR_SRC_PATH/bindings/python/tools/*.py $TPU_PYTHON_PATH/
# python utils
cp -a $MLIR_SRC_PATH/python/utils/* $TPU_PYTHON_PATH/
pushd $TPU_PYTHON_PATH/model/retinaface; make; popd
# calibration tool
if [ ! -e $BUILD_PATH/build_calibration ]; then
  mkdir -p $BUILD_PATH/build_calibration
fi
pushd $BUILD_PATH/build_calibration
cmake $MLIR_SRC_PATH/python/calibration && make
cp calibration_math.so $INSTALL_PATH/lib
popd
cp $MLIR_SRC_PATH/python/calibration/*.py $TPU_PYTHON_PATH/

# cvibuilder
if [ ! -e $BUILD_PATH/build_cvimodel ]; then
  mkdir -p $BUILD_PATH/build_cvimodel/include
fi
pushd $BUILD_PATH/build_cvimodel/include
$INSTALL_PATH/flatbuffers/bin/flatc --cpp --gen-object-api \
    $MLIR_SRC_PATH/externals/cvibuilder/src/cvimodel.fbs
popd
cp -a $MLIR_SRC_PATH/externals/cvibuilder/python/* $TPU_PYTHON_PATH/

# build cmodel
if [ ! -e $BUILD_PATH/build_cmodel ]; then
  mkdir -p $BUILD_PATH/build_cmodel
fi
pushd $BUILD_PATH/build_cmodel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DBMKERNEL_PATH=$BMKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH \
    $MLIR_SRC_PATH/externals/cmodel
cmake --build . --target install
popd

# build runtime
if [ ! -e $BUILD_PATH/build_runtime ]; then
  mkdir $BUILD_PATH/build_runtime
fi
pushd $BUILD_PATH/build_runtime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL $BUILD_FLAG \
    -DBMKERNEL_PATH=$BMKERNEL_PATH \
    -DCMODEL_PATH=$CMODEL_PATH \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCVIBUILDER_PATH=$BUILD_PATH/build_cvimodel \
    -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH \
    $MLIR_SRC_PATH/externals/runtime
cmake --build . --target install
popd

# SoC build
# TODO
