#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

# build caffe
if [ ! -e $MLIR_SRC_PATH/third_party/caffe/build ]; then
  mkdir $MLIR_SRC_PATH/third_party/caffe/build
fi
pushd $MLIR_SRC_PATH/third_party/caffe/build
cmake -G Ninja -DUSE_OPENCV=OFF -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
cmake --build . --target install
popd

# download and unzip mkldnn
if [ ! -e $MKLDNN_PATH ]; then
  wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
  tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
  mv mkldnn_lnx_1.0.2_cpu_gomp $MKLDNN_PATH
  rm mkldnn_lnx_1.0.2_cpu_gomp.tgz
fi


# build bmkernel
if [ ! -e $MLIR_SRC_PATH/externals/bmkernel/build ]; then
  mkdir $MLIR_SRC_PATH/externals/bmkernel/build
fi
pushd $MLIR_SRC_PATH/externals/bmkernel/build
cmake -G Ninja -DCHIP=BM1880v2 -DCMAKE_INSTALL_PREFIX=$BMKERNEL_PATH ..
cmake --build . --target install
popd

# build support
if [ ! -e $MLIR_SRC_PATH/externals/support/build ]; then
  mkdir $MLIR_SRC_PATH/externals/support/build
fi
pushd $MLIR_SRC_PATH/externals/support/build
cmake -G Ninja -DCHIP=BM1880v2 -DCMAKE_INSTALL_PREFIX=$SUPPORT_PATH ..
cmake --build . --target install
popd

# build cmodel
if [ ! -e $MLIR_SRC_PATH/externals/cmodel/build ]; then
  mkdir $MLIR_SRC_PATH/externals/cmodel/build
fi
pushd $MLIR_SRC_PATH/externals/cmodel/build
cmake -G Ninja -DCHIP=BM1880v2 -DBMKERNEL_PATH=$BMKERNEL_PATH -DSUPPORT_PATH=$SUPPORT_PATH -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH ..
cmake --build . --target install
popd

# build bmbuilder
if [ ! -e $MLIR_SRC_PATH/externals/bmbuilder/build ]; then
  mkdir $MLIR_SRC_PATH/externals/bmbuilder/build
fi
pushd $MLIR_SRC_PATH/externals/bmbuilder/build
cmake -G Ninja -DCHIP=BM1880v2 -DBMKERNEL_PATH=$BMKERNEL_PATH -DCMAKE_INSTALL_PREFIX=$BMBUILDER_PATH ..
cmake --build . --target install
popd

# build runtime
if [ ! -e $MLIR_SRC_PATH/externals/runtime/build ]; then
  mkdir $MLIR_SRC_PATH/externals/runtime/build
fi
pushd $MLIR_SRC_PATH/externals/runtime/build
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL -DSUPPORT_PATH=$SUPPORT_PATH -DBMBUILDER_PATH=$BMBUILDER_PATH -DBMKERNEL_PATH=$BMKERNEL_PATH -DCMODEL_PATH=$CMODEL_PATH -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH ..
cmake --build . --target install
popd

# build mlir-tpu
if [ ! -e $TPU_BASE/llvm-project/build ]; then
  mkdir $TPU_BASE/llvm-project/build
fi
pushd $TPU_BASE/llvm-project/build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH=$CAFFE_PATH -DMKLDNN_PATH=$MKLDNN_PATH -DBMKERNEL_PATH=$BMKERNEL_PATH -DCMAKE_INSTALL_PREFIX=$MLIR_PATH -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON
cmake --build . --target check-mlir
cmake --build . --target pymlir
popd

cd $TPU_BASE/llvm-project/build

