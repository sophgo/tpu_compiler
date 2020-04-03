#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# build caffe
if [ ! -e $BUILD_PATH/build_caffe ]; then
  mkdir -p $BUILD_PATH/build_caffe
  pushd $BUILD_PATH/build_caffe
  cmake -G Ninja -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
      -DBLAS=open -DUSE_OPENMP=TRUE \
      -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH \
      -DCMAKE_CXX_FLAGS=-std=gnu++11 \
      -Dpython_version=$PYTHON_VERSION \
      $MLIR_SRC_PATH/third_party/caffe
  popd
fi
pushd $BUILD_PATH/build_caffe
cmake --build . --target install
popd

# build flatbuffers
if [ ! -e $BUILD_PATH/build_flatbuffers ]; then
  mkdir -p $BUILD_PATH/build_flatbuffers
  pushd $BUILD_PATH/build_flatbuffers
  cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$FLATBUFFERS_PATH \
      $MLIR_SRC_PATH/third_party/flatbuffers
  popd
fi
pushd $BUILD_PATH/build_flatbuffers
cmake --build . --target install
popd
cp -a $MLIR_SRC_PATH/third_party/flatbuffers/python $FLATBUFFERS_PATH/

# build cvikernel
if [ ! -e $BUILD_PATH/build_cvikernel ]; then
  mkdir -p $BUILD_PATH/build_cvikernel
  pushd $BUILD_PATH/build_cvikernel
  cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
      -DCMAKE_INSTALL_PREFIX=$CVIKERNEL_PATH \
      $MLIR_SRC_PATH/externals/cvikernel
  popd
fi
pushd $BUILD_PATH/build_cvikernel
cmake --build . --target install
popd

# build mlir-tpu
pushd $BUILD_PATH
cmake --build . --target check-mlir
cmake --build . --target pymlir
cmake --build . --target pybind
cmake --build . --target install
cp lib/pymlir*.so $TPU_PYTHON_PATH
cp lib/pybind*.so $TPU_PYTHON_PATH
popd

CVI_PY_TOOLKIT=$MLIR_SRC_PATH/python/cvi_toolkit
# python package
cp -ar $CVI_PY_TOOLKIT/ $TPU_PYTHON_PATH/
cp -ar $CVI_PY_TOOLKIT/*.py $TPU_PYTHON_PATH/


# python script
cp $CVI_PY_TOOLKIT/binary_helper/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/calibration/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/eval/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/inference/caffe/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/inference/mlir/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/inference/onnx/*.py $TPU_PYTHON_PATH/

cp -ar  $CVI_PY_TOOLKIT/retinaface/ $TPU_PYTHON_PATH/
pushd $TPU_PYTHON_PATH/retinaface; make; popd
cp -ar $TPU_PYTHON_PATH/retinaface/* $TPU_PYTHON_PATH/


# calibration tool
if [ ! -e $BUILD_PATH/build_calibration ]; then
  mkdir -p $BUILD_PATH/build_calibration
fi
pushd $BUILD_PATH/build_calibration
cmake $CVI_PY_TOOLKIT/calibration && make
cp calibration_math.so $INSTALL_PATH/lib
popd

# cvibuilder
if [ ! -e $BUILD_PATH/build_cvimodel ]; then
  mkdir -p $BUILD_PATH/build_cvimodel
  pushd $BUILD_PATH/build_cvimodel
  cmake -G Ninja -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
      $MLIR_SRC_PATH/externals/cvibuilder
  popd
fi
pushd $BUILD_PATH/build_cvimodel
cmake --build . --target install
popd

# build cmodel
if [ ! -e $BUILD_PATH/build_cmodel ]; then
  mkdir -p $BUILD_PATH/build_cmodel
  pushd $BUILD_PATH/build_cmodel
  cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
      -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
      -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH \
      $MLIR_SRC_PATH/externals/cmodel
  popd
fi
pushd $BUILD_PATH/build_cmodel
cmake --build . --target install
popd

# build cviruntime
if [ ! -e $BUILD_PATH/build_cviruntime ]; then
  mkdir $BUILD_PATH/build_cviruntime
  pushd $BUILD_PATH/build_cviruntime
  cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL $BUILD_FLAG \
      -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
      -DCMODEL_PATH=$CMODEL_PATH \
      -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
      -DCVIBUILDER_PATH=$BUILD_PATH/build_cvimodel \
      -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH \
      $MLIR_SRC_PATH/externals/cviruntime
  popd
fi
pushd $BUILD_PATH/build_cviruntime
cmake --build . --target install
popd

# build systemc (for profiling)
# building has some issue, has to build in place for now
# copy the source dir to build dir
if [ ! -e $BUILD_PATH/build_systemc ]; then
  mkdir $BUILD_PATH/build_systemc
  pushd $BUILD_PATH/build_systemc
  cp $MLIR_SRC_PATH/third_party/systemc-2.3.3/* . -a
  autoreconf -ivf
  ./configure CXXFLAGS=-std=c++11
  popd
fi
pushd $BUILD_PATH/build_systemc
make -j`nproc`
make install
mkdir -p $SYSTEMC_PATH
cp -a include $SYSTEMC_PATH/
cp -a lib-linux64 $SYSTEMC_PATH/
popd

# build profiling
if [ ! -e $BUILD_PATH/build_profiling ]; then
  mkdir $BUILD_PATH/build_profiling
  pushd $BUILD_PATH/build_profiling
  cmake -G Ninja  \
      -DSYSTEMC_PATH=$SYSTEMC_PATH \
      -DCMAKE_INSTALL_PREFIX=$PROFILING_PATH \
      $BUILD_PROFILING_FLAG \
      $MLIR_SRC_PATH/externals/profiling
  popd
fi
pushd $BUILD_PATH/build_profiling
cmake --build . --target install
popd
cp $MLIR_SRC_PATH/externals/profiling/tool/performance.html $PROFILING_PATH/bin/

