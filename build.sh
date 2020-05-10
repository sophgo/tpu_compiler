#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

BUILD_TYPE="RELEASE"
if [ "$BUILD_TYPE" == "RELEASE" ]; then
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3"
else
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"
  export BUILD_PATH=${BUILD_PATH}_debug
fi

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi

# mkdir
if [ ! -e $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
else
  rm -rf $INSTALL_PATH/mkldnn
  find $INSTALL_PATH -name *.h -exec rm {} \;
fi

if [ ! -e $TPU_PYTHON_PATH ]; then
  mkdir -p $TPU_PYTHON_PATH
fi

if [ ! -e $BUILD_PATH ]; then
  mkdir -p $BUILD_PATH
fi

# download and unzip mkldnn
if [ ! -e $MKLDNN_PATH ]; then
  if [ ! -f mkldnn_lnx_1.0.2_cpu_gomp.tgz ]; then
    wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
  fi
  tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
  mkdir -p $MKLDNN_PATH
  mv mkldnn_lnx_1.0.2_cpu_gomp/* $MKLDNN_PATH/
  rm -rf mkldnn_lnx_1.0.2_cpu_gomp
fi

# build caffe
if [ ! -e $BUILD_PATH/build_caffe ]; then
  mkdir -p $BUILD_PATH/build_caffe
fi
pushd $BUILD_PATH/build_caffe
cmake -G Ninja -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DBLAS=open -DUSE_OPENMP=TRUE \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH \
    -DCMAKE_CXX_FLAGS=-std=gnu++11 \
    -Dpython_version=$PYTHON_VERSION \
    $MLIR_SRC_PATH/third_party/caffe
cmake --build . --target install
popd

# build opencv
if [ ! -e $BUILD_PATH/build_opencv ]; then
  mkdir -p $BUILD_PATH/build_opencv
fi
pushd $BUILD_PATH/build_opencv
cmake -G Ninja \
    -DWITH_CUDA=OFF -DWITH_IPP=OFF -DWITH_LAPACK=OFF \
    -DWITH_DC1394=OFF -DWITH_GPHOTO2=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_videoio=OFF \
    -DBUILD_opencv_superres=OFF -DBUILD_opencv_videostab=OFF \
    -DBUILD_opencv_stitching=OFF -DBUILD_opencv_objdetect=OFF \
    -DBUILD_opencv_calib3d=OFF -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_video=OFF -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_photo=OFF \
    -DCMAKE_INSTALL_PREFIX=$OPENCV_PATH \
    $MLIR_SRC_PATH/third_party/opencv
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

# build cvikernel
if [ ! -e $BUILD_PATH/build_cvikernel ]; then
  mkdir -p $BUILD_PATH/build_cvikernel
fi
pushd $BUILD_PATH/build_cvikernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$CVIKERNEL_PATH \
    $MLIR_SRC_PATH/externals/cvikernel
cmake --build . --target install
popd

# cvibuilder
if [ ! -e $BUILD_PATH/build_cvimodel ]; then
  mkdir -p $BUILD_PATH/build_cvimodel
fi
pushd $BUILD_PATH/build_cvimodel
cmake -G Ninja -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    $MLIR_SRC_PATH/externals/cvibuilder
cmake --build . --target install
popd

# build mlir-tpu
pushd $BUILD_PATH
cmake -G Ninja -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON \
    -DOCAMLFIND=NO \
    $BUILD_FLAG \
    -DMKLDNN_PATH=$MKLDNN_PATH \
    -DCAFFE_PATH=$CAFFE_PATH \
    -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$MLIR_PATH \
    -DPYBIND11_PYTHON_VERSION=$PYTHON_VERSION \
    $TPU_BASE/llvm-project/llvm
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

# TFLite flatbuffer Schema
${FLATBUFFERS_PATH}/bin/flatc \
    -o $TPU_PYTHON_PATH --python \
    $MLIR_SRC_PATH/python/tflite_schema/schema.fbs

# calibration tool
if [ ! -e $BUILD_PATH/build_calibration ]; then
  mkdir -p $BUILD_PATH/build_calibration
fi
pushd $BUILD_PATH/build_calibration
cmake $CVI_PY_TOOLKIT/calibration && make
cp calibration_math.so $INSTALL_PATH/lib
popd

# build cmodel
if [ ! -e $BUILD_PATH/build_cmodel ]; then
  mkdir -p $BUILD_PATH/build_cmodel
fi
pushd $BUILD_PATH/build_cmodel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
    -DCMAKE_INSTALL_PREFIX=$CMODEL_PATH \
    $MLIR_SRC_PATH/externals/cmodel
cmake --build . --target install
popd

# build cviruntime
if [ ! -e $BUILD_PATH/build_cviruntime ]; then
  mkdir $BUILD_PATH/build_cviruntime
fi
pushd $BUILD_PATH/build_cviruntime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL $BUILD_FLAG \
    -DCVIKERNEL_PATH=$CVIKERNEL_PATH \
    -DCMODEL_PATH=$CMODEL_PATH \
    -DFLATBUFFERS_PATH=$FLATBUFFERS_PATH \
    -DCVIBUILDER_PATH=$BUILD_PATH/build_cvimodel \
    -DCMAKE_INSTALL_PREFIX=$RUNTIME_PATH \
    -DENABLE_TEST=ON \
    $MLIR_SRC_PATH/externals/cviruntime
cmake --build . --target install
ctest --progress || true
popd

# build systemc (for profiling)
# building has some issue, has to build in place for now
# copy the source dir to build dir
if [ ! -e $BUILD_PATH/build_systemc ]; then
  mkdir $BUILD_PATH/build_systemc
fi
pushd $BUILD_PATH/build_systemc
cp $MLIR_SRC_PATH/third_party/systemc-2.3.3/* . -a
autoreconf -ivf
./configure CXXFLAGS=-std=c++11
make -j`nproc`
make install
mkdir -p $SYSTEMC_PATH
cp -a include $SYSTEMC_PATH/
cp -a lib-linux64 $SYSTEMC_PATH/
popd

# build profiling
if [ ! -e $BUILD_PATH/build_profiling ]; then
  mkdir $BUILD_PATH/build_profiling
fi
pushd $BUILD_PATH/build_profiling
cmake -G Ninja  \
    -DSYSTEMC_PATH=$SYSTEMC_PATH \
    -DCMAKE_INSTALL_PREFIX=$PROFILING_PATH \
    $BUILD_PROFILING_FLAG \
    $MLIR_SRC_PATH/externals/profiling
cmake --build . --target install
popd
cp $MLIR_SRC_PATH/externals/profiling/tool/performance.html $PROFILING_PATH/bin/

# Clean up some files for release build
if [ "$1" = "RELEASE" ]; then
  if [ -z $INSTALL_PATH ]; then
    echo "INSTALL_PATH not specified"
    exit 1
  fi
  rm -f $INSTALL_PATH/bin/llvm-*
  rm -f $INSTALL_PATH/bin/llc
  rm -f $INSTALL_PATH/bin/lli
  rm -f $INSTALL_PATH/bin/opt
  rm -f $INSTALL_PATH/bin/sancov
  rm -f $INSTALL_PATH/bin/dsymutil
  rm -f $INSTALL_PATH/bin/bugpoint
  rm -f $INSTALL_PATH/bin/verify-uselistorder
  rm -f $INSTALL_PATH/bin/sanstats
  rm -f $INSTALL_PATH/bin/yaml2obj
  rm -f $INSTALL_PATH/lib/*.a
  rm -f $INSTALL_PATH/lib/libLTO.so*
  rm -f $INSTALL_PATH/lib/libmlir_runner_utils.so*
  rm -f $INSTALL_PATH/lib/libRemarks.so*
  rm -rf $INSTALL_PATH/lib/cmake/
  # install regression
  mkdir -p $INSTALL_PATH/regression
  cp -a $MLIR_SRC_PATH/regression/generic $INSTALL_PATH/regression/
  # cp -a $MLIR_SRC_PATH/regression/parallel $INSTALL_PATH/regression/
  cp -a $MLIR_SRC_PATH/regression/data $INSTALL_PATH/regression/
  cp -a $MLIR_SRC_PATH/regression/convert_model_caffe.sh $INSTALL_PATH/regression/
  cp -a $MLIR_SRC_PATH/regression/convert_model_onnx.sh $INSTALL_PATH/regression/
  cp -a $MLIR_SRC_PATH/regression/generate_all_cvimodels.sh $INSTALL_PATH/regression/
  cp -a $MLIR_SRC_PATH/regression/run_regression.sh $INSTALL_PATH/regression/

  # install env script
  cp $MLIR_SRC_PATH/cvitek_envs.sh $INSTALL_PATH/

  # generate models for release and samples
  pushd $BUILD_PATH
  $MLIR_SRC_PATH/regression/generate_all_cvimodels.sh
  mkdir -p cvimodel_samples
  cp cvimodel_release/mobilenet_v2.cvimodel cvimodel_samples/
  cp cvimodel_release/yolo_v3_416_with_detection.cvimodel cvimodel_samples/
  cp cvimodel_release/alphapose.cvimodel cvimodel_samples/
  cp cvimodel_release/retinaface_mnet25_with_detection.cvimodel cvimodel_samples/
  cp cvimodel_release/retinaface_res50_with_detection.cvimodel cvimodel_samples/
  cp cvimodel_release/arcface_res50.cvimodel cvimodel_samples/

  if [ "$BUILD_CVITEK_ZOO" = "1" ]; then
    $MLIR_SRC_PATH/regression/cvitek_zoo/cvitek_zoo_generate_cvimodels.sh
  fi
  popd
fi

# build python package
pushd $MLIR_SRC_PATH
if [ $PYTHON_VERSION == "2" ]; then
  echo "Not support build python2 package"
elif [ $PYTHON_VERSION == "3" ]; then
  pip3 install wheel
  python3 setup/python3/setup.py bdist_wheel --dist-dir=$INSTALL_PATH/python3_package/ --plat-name="linux_x86_64"
  python3 setup/python3/setup.py clean
fi
popd

# SoC build
if [ "$1" = "SOC" ]; then
source build_soc.sh
fi

