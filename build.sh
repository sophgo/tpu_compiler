#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}
BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
TPU_PYTHON_PATH=$INSTALL_PATH/tpuc/python

BUILD_TYPE="RELEASE"
if [ "$BUILD_TYPE" == "RELEASE" ]; then
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3"
else
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"
  BUILD_PATH=${BUILD_PATH}_debug
fi

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
echo "TPU_PYTHON_PATH: $TPU_PYTHON_PATH"

# prepare install/build dir
mkdir -p $BUILD_PATH
mkdir -p $INSTALL_PATH
mkdir -p $TPU_PYTHON_PATH

# download and unzip mkldnn
if [ ! -e $INSTALL_PATH/mkldnn ]; then
  mkdir -p $BUILD_PATH/mkldnn
  pushd $BUILD_PATH/mkldnn
  if [ ! -f mkldnn_lnx_1.0.2_cpu_gomp.tgz ]; then
    wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
  fi
  tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
  mkdir -p $INSTALL_PATH/mkldnn
  mv mkldnn_lnx_1.0.2_cpu_gomp/* $INSTALL_PATH/mkldnn
  rm -rf mkldnn_lnx_1.0.2_cpu_gomp
  popd
fi

# build pybind11
mkdir -p $BUILD_PATH/pybind11
pushd $BUILD_PATH/pybind11
cmake -G Ninja \
    -DPYBIND11_TEST=OFF \
    -DPYBIND11_PYTHON_VERSION=3 \
    $PROJECT_ROOT/third_party/pybind11 \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/pybind11
cmake --build . --target install
popd

# build llvm
mkdir -p $BUILD_PATH/llvm
pushd $BUILD_PATH/llvm
cmake -G Ninja \
  $BUILD_FLAG \
  $PROJECT_ROOT/third_party/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_BINDINGS_PYTHON_ENABLED=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -Dpybind11_DIR=$INSTALL_PATH/pybind11/share/cmake/pybind11
ninja
popd
mkdir -p $INSTALL_PATH/tpuc/python
mkdir -p $INSTALL_PATH/tpuc/lib
cp -rf $BUILD_PATH/llvm/python/* $INSTALL_PATH/tpuc/python/
cp -rf $BUILD_PATH/llvm/lib/libMLIRPublicAPI* $INSTALL_PATH/tpuc/lib

# build caffe
mkdir -p $BUILD_PATH/caffe
pushd $BUILD_PATH/caffe
cmake -G Ninja \
    $PROJECT_ROOT/third_party/caffe \
    -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DBLAS=open -DUSE_OPENMP=TRUE \
    -DCMAKE_CXX_FLAGS=-std=gnu++11 \
    -Dpython_version="3" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/caffe
cmake --build . --target install
popd

# build flatbuffers
mkdir -p $BUILD_PATH/flatbuffers
pushd $BUILD_PATH/flatbuffers
cmake -G Ninja \
    $PROJECT_ROOT/third_party/flatbuffers \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/flatbuffers
cmake --build . --target install
popd
cp -a $PROJECT_ROOT/third_party/flatbuffers/python \
      $INSTALL_PATH/flatbuffers/

# build cvikernel
mkdir -p $BUILD_PATH/cvikernel
pushd $BUILD_PATH/cvikernel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    $PROJECT_ROOT/externals/cvikernel
cmake --build . --target install
popd

# cvibuilder
mkdir -p $BUILD_PATH/cvimodel
pushd $BUILD_PATH/cvimodel
cmake -G Ninja -DFLATBUFFERS_PATH=$INSTALL_PATH/flatbuffers \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    $PROJECT_ROOT/externals/cvibuilder
cmake --build . --target install
popd

# cnpy
mkdir -p $BUILD_PATH/cnpy
pushd $BUILD_PATH/cnpy
cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/cnpy \
    $PROJECT_ROOT/third_party/cnpy
cmake --build . --target install
popd
cp $INSTALL_PATH/cnpy/lib/* $INSTALL_PATH/tpuc/lib/

mkdir -p $BUILD_PATH/tpuc
pushd $BUILD_PATH/tpuc
cmake -G Ninja \
    $BUILD_FLAG \
    -DMKLDNN_PATH=$INSTALL_PATH/mkldnn \
    -DCAFFE_PATH=$INSTALL_PATH/caffe \
    -DCVIKERNEL_PATH=$INSTALL_PATH/cvikernel \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    -Dpybind11_DIR=$INSTALL_PATH/pybind11/share/cmake/pybind11 \
    -DMLIR_DIR=$BUILD_PATH/llvm/lib/cmake/mlir \
    -DLLVM_DIR=$BUILD_PATH/llvm/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    $PROJECT_ROOT
cmake --build . --target install
popd

# build opencv
mkdir -p $BUILD_PATH/opencv
pushd $BUILD_PATH/opencv
cmake -G Ninja \
    $PROJECT_ROOT/third_party/opencv \
    -DWITH_CUDA=OFF -DWITH_IPP=OFF -DWITH_LAPACK=OFF \
    -DWITH_DC1394=OFF -DWITH_GPHOTO2=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_videoio=OFF \
    -DBUILD_opencv_superres=OFF \
    -DBUILD_opencv_videostab=OFF \
    -DBUILD_opencv_stitching=OFF \
    -DBUILD_opencv_objdetect=OFF \
    -DBUILD_opencv_calib3d=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_video=OFF \
    -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_photo=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/opencv
cmake --build . --target install
popd


CVI_PY_TOOLKIT=$PROJECT_ROOT/python/cvi_toolkit
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
cp $CVI_PY_TOOLKIT/inference/postprocess/*.py $TPU_PYTHON_PATH/
cp -ar $CVI_PY_TOOLKIT/test/onnx_ir_test/*.py $TPU_PYTHON_PATH/
cp -ar $PROJECT_ROOT/python/python_codegen/*.py $TPU_PYTHON_PATH/


cp -ar $CVI_PY_TOOLKIT/retinaface/ $TPU_PYTHON_PATH/
pushd $TPU_PYTHON_PATH/retinaface; make; popd
cp -ar $TPU_PYTHON_PATH/retinaface/* $TPU_PYTHON_PATH/

# Build rcnn cython
pushd $TPU_PYTHON_PATH/rcnn/cython
python3 setup.py build_ext --inplace
python3 setup.py clean
popd

# TFLite flatbuffer Schema
${INSTALL_PATH}/flatbuffers/bin/flatc \
    -o $TPU_PYTHON_PATH --python \
    $PROJECT_ROOT/python/tflite_schema/schema.fbs

# calibration tool
mkdir -p $BUILD_PATH/calibration
pushd $BUILD_PATH/calibration
cmake $CVI_PY_TOOLKIT/calibration && make
cp calibration_math.so $INSTALL_PATH/tpuc/lib
popd

# build cmodel
mkdir -p $BUILD_PATH/cmodel
pushd $BUILD_PATH/cmodel
cmake -G Ninja -DCHIP=BM1880v2 $BUILD_FLAG \
    -DCVIKERNEL_PATH=$INSTALL_PATH/tpuc \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    $PROJECT_ROOT/externals/cmodel
cmake --build . --target install
popd

# build cviruntime
mkdir -p $BUILD_PATH/cviruntime
pushd $BUILD_PATH/cviruntime
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL $BUILD_FLAG \
    -DCVIKERNEL_PATH=$INSTALL_PATH/tpuc \
    -DCMODEL_PATH=$INSTALL_PATH/tpuc \
    -DENABLE_PYRUNTIME=ON \
    -DFLATBUFFERS_PATH=$INSTALL_PATH/flatbuffers \
    -DCVIBUILDER_PATH=$INSTALL_PATH/tpuc \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    -DENABLE_TEST=ON \
    $PROJECT_ROOT/externals/cviruntime
cmake --build . --target install
#ctest --progress || true
popd

# build cvimath
mkdir -p $BUILD_PATH/cvimath
pushd $BUILD_PATH/cvimath
cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE=-O3 \
    -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
    -DTPU_SDK_ROOT=$INSTALL_PATH/tpuc \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    $PROJECT_ROOT/externals/cvimath
cmake --build . --target install
#ctest --progress || true
popd


# build systemc (for profiling)
# building has some issue, has to build in place for now
# copy the source dir to build dir
mkdir -p $BUILD_PATH/systemc
pushd $BUILD_PATH/systemc
cp $PROJECT_ROOT/third_party/systemc-2.3.3/* . -ur
autoreconf -iv
./configure CXXFLAGS=-std=c++11
make -j`nproc`
make install
mkdir -p $INSTALL_PATH/systemc
cp -a include $INSTALL_PATH/systemc
cp -a lib-linux64 $INSTALL_PATH/systemc
popd

# build profiling
mkdir -p $BUILD_PATH/profiling
pushd $BUILD_PATH/profiling
cmake -G Ninja  \
    -DSYSTEMC_PATH=$INSTALL_PATH/systemc \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    $PROJECT_ROOT/externals/profiling
cmake --build . --target install
popd
cp $PROJECT_ROOT/externals/profiling/tool/performance.html $INSTALL_PATH/tpuc/bin/

# Clean up some files for release build
if [ "$1" = "RELEASE" ]; then
  if [ -z $INSTALL_PATH ]; then
    echo "INSTALL_PATH not specified"
    exit 1
  fi
  rm -f $INSTALL_PATH/tpuc/regression_models.sh
  rm -f $INSTALL_PATH/tpuc/regression_models_e2e.sh
  rm -f $INSTALL_PATH/tpuc/regression_models_e2e_skip_preprocess.sh
  # rm all test prgram
  rm -f $INSTALL_PATH/tpuc/bin/test_*
  rm -f $INSTALL_PATH/tpuc/bin/sample_*
  # strip mlir tools
  strip $INSTALL_PATH/tpuc/bin/tpuc-opt
  strip $INSTALL_PATH/tpuc/bin/tpuc-interpreter
  strip $INSTALL_PATH/tpuc/bin/tpuc-translate
  strip $INSTALL_PATH/tpuc/bin/cvi_profiling

  # install regression
  mkdir -p $INSTALL_PATH/tpuc/regression
  pushd $INSTALL_PATH/tpuc/regression
  cp -a $PROJECT_ROOT/regression/generic ./
  # cp -a $PROJECT_ROOT/regression/parallel ./
  cp -a $PROJECT_ROOT/regression/data ./
  cp -a $PROJECT_ROOT/regression/convert_model.sh ./
  cp -a $PROJECT_ROOT/regression/mlir_to_cvimodel.sh ./
  cp -a $PROJECT_ROOT/regression/generate_all_cvimodels.sh ./
  cp -a $PROJECT_ROOT/regression/run_regression.sh ./
  popd

  # install env script
  cp $PROJECT_ROOT/cvitek_envs.sh $INSTALL_PATH/

  # generate models for release and samples
  pushd $BUILD_PATH

  $PROJECT_ROOT/regression/generate_all_cvimodels.sh
  mkdir -p cvimodel_samples

  sample_models_list=(
    mobilenet_v2.cvimodel
    mobilenet_v2_fused_preprocess.cvimodel
    yolo_v3_416_with_detection.cvimodel
    yolo_v3_416_fused_preprocess_with_detection.cvimodel
    alphapose.cvimodel
    alphapose_fused_preprocess.cvimodel
    retinaface_mnet25_600_with_detection.cvimodel
    retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel
    arcface_res50.cvimodel
    arcface_res50_fused_preprocess.cvimodel
  )

  for sample_model in ${sample_models_list[@]}
  do
    cp cvimodel_release/${sample_model} cvimodel_samples/
  done

  popd
fi
