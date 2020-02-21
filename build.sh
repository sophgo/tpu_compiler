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

# download and unzip mkldnn
if [ ! -e $MKLDNN_PATH ]; then
  wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
  tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
  mv mkldnn_lnx_1.0.2_cpu_gomp $MKLDNN_PATH
  rm mkldnn_lnx_1.0.2_cpu_gomp.tgz
fi

# build caffe
CAFFE_USE_INTEL_BRANCH=0  # use master by default
if [ ! -e $MLIR_SRC_PATH/third_party/caffe/build ]; then
  mkdir $MLIR_SRC_PATH/third_party/caffe/build
fi
# based on master branch (tpu_master)
if [ $CAFFE_USE_INTEL_BRANCH -eq 0 ]; then
  pushd $MLIR_SRC_PATH/third_party/caffe/build
  cmake -G Ninja -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
      -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
  cmake --build . --target install
  popd
fi
# based on intel branch (tpu_intel)
# will download mkl/mlsl automatically
# copy external/mkl/* into $CAFFE_PATH as well
if [ $CAFFE_USE_INTEL_BRANCH -eq 1 ]; then
  CAFFE_MKLDNN_PATH=$MLIR_SRC_PATH/third_party/caffe/external/mkldnn
  if [ ! -e $CAFFE_MKLDNN_PATH/install ]; then
    source /etc/lsb-release
    if [ $DISTRIB_RELEASE = "18.04" ]; then
      echo "Ubuntu 18.04"
      pushd $CAFFE_MKLDNN_PATH
      ln -s install_ubuntu1804 install
      popd
    elif [ $DISTRIB_RELEASE = "16.04" ]; then
      echo "Ubuntu 16.04"
      pushd $CAFFE_MKLDNN_PATH
      ln -s install_ubuntu1604 install
      popd
    else
      echo "Not Ubuntu 18.04 or 16.04"
      echo "Please build caffe manually according to third_party/README.md"
      return 1
    fi
  fi
  pushd $MLIR_SRC_PATH/third_party/caffe/build
  MKLDNNROOT=$CAFFE_MKLDNN_PATH/install cmake \
      -DUSE_OPENCV=OFF -DDISABLE_MKLDNN_DOWNLOAD=1 \
      -DUSE_OPENMP=OFF -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF -DUSE_MLSL=OFF  \
      -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
  cmake --build . --target install
  cp ../external/mkl $CAFFE_PATH -a
  popd
fi

# build flatbuffers
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

# generate flatbuffer schema
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

# build bmkernel
if [ ! -e $MLIR_SRC_PATH/externals/bmkernel/build ]; then
  mkdir $MLIR_SRC_PATH/externals/bmkernel/build
fi
pushd $MLIR_SRC_PATH/externals/bmkernel/build
cmake -G Ninja -DCHIP=BM1880v2 -DCMAKE_INSTALL_PREFIX=$BMKERNEL_PATH $BUILD_FLAG ..
cmake --build . --target install
popd

# build support
if [ ! -e $MLIR_SRC_PATH/externals/support/build ]; then
  mkdir $MLIR_SRC_PATH/externals/support/build
fi
pushd $MLIR_SRC_PATH/externals/support/build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$SUPPORT_PATH $BUILD_FLAG ..
cmake --build . --target install
popd

# build cmodel
if [ ! -e $MLIR_SRC_PATH/externals/cmodel/build ]; then
  mkdir $MLIR_SRC_PATH/externals/cmodel/build
fi
pushd $MLIR_SRC_PATH/externals/cmodel/build
cmake -G Ninja -DCHIP=BM1880v2 -DBMKERNEL_PATH=$BMKERNEL_PATH $BUILD_FLAG \
    -DSUPPORT_PATH=$SUPPORT_PATH -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH ..
cmake --build . --target install
popd

# build runtime
if [ ! -e $MLIR_SRC_PATH/externals/runtime/build ]; then
  mkdir $MLIR_SRC_PATH/externals/runtime/build
fi
pushd $MLIR_SRC_PATH/externals/runtime/build
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL \
    -DSUPPORT_PATH=$SUPPORT_PATH \
    -DBMKERNEL_PATH=$BMKERNEL_PATH -DCMODEL_PATH=$CMODEL_PATH \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH -DCVIBUILDER_PATH=$CVIBUILDER_PATH $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH ..
cmake --build . --target install
popd

# build calibration tool
if [ ! -e $MLIR_SRC_PATH/externals/calibration_tool/build ]; then
  mkdir $MLIR_SRC_PATH/externals/calibration_tool/build
fi
if [ ! -e $CALIBRATION_TOOL_PATH ]; then
  mkdir $CALIBRATION_TOOL_PATH
fi
pushd $MLIR_SRC_PATH/externals/calibration_tool/build
cmake ..  && make
cp calibration_math.so $CALIBRATION_TOOL_PATH
cp ../*.py $CALIBRATION_TOOL_PATH
popd

# build python tool
pushd $PYTHON_TOOLS_PATH/model/retinaface
make
popd

# build mlir-tpu
if [ ! -e $TPU_BASE/llvm-project/build ]; then
  mkdir $TPU_BASE/llvm-project/build
fi
pushd $TPU_BASE/llvm-project/build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH=$CAFFE_PATH \
    -DMKLDNN_PATH=$MKLDNN_PATH -DBMKERNEL_PATH=$BMKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$MLIR_PATH -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON \
    $BUILD_FLAG
cmake --build . --target check-mlir
cmake --build . --target pymlir
cmake --build . --target pybind
popd

cd $TPU_BASE/llvm-project/build

