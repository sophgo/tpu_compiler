#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi


BUILD_TYPE="RELEASE"
if [ "$BUILD_TYPE" == "RELEASE" ]; then
  CXXFLAGS="-O3 -fopenmp"
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=${CXXFLAGS}"
else
  CXXFLAGS="-ggdb -fopenmp"
  BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=${CXXFLAGS}"
  BUILD_PATH=${BUILD_PATH}_debug
fi

TPU_PYTHON_PATH=$INSTALL_PATH/tpuc/python

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
echo "TPU_PYTHON_PATH: $TPU_PYTHON_PATH"

is_valid_cached_path()
{
  if [ -n "$BUILD_CACHED_PATH" ]; then
    if [ -d $BUILD_CACHED_PATH ]; then
      valid_cached_path=1
      return
    fi
  fi

  valid_cached_path=0
}

is_clean_llvm_source()
{
  if [ -d $PROJECT_ROOT/third_party/llvm-project/llvm ]; then
    pushd $PROJECT_ROOT/third_party/llvm-project/llvm
    if [[ $(git diff --stat) == '' ]]; then
      popd
      cleaned_llvm_source=1
      return
    fi
  fi

  cleaned_llvm_source=0
}

get_llvm_version()
{
  pushd $PROJECT_ROOT/third_party/llvm-project/llvm
  LLVM_VER="$(git show -s --format=%H)"
  popd
}

is_valid_cached_llvm()
{
  local CACHED_LLVM_BUILD_PATH=$1
  if [ -e $CACHED_LLVM_BUILD_PATH/bin/llvm-readobj ]; then
    valid_cached_llvm_exec=1
    return
  fi

  valid_cached_llvm_exec=0
}

build_llvm_fn()
{
  local src_dir=$1
  local build_path=$2

  if [ ! -d $build_path ]; then
    mkdir $build_path
  fi

  pushd $build_path
  cmake -G Ninja \
    $BUILD_FLAG \
    $src_dir/llvm \
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
}

install_llvm_fun()
{
  BUILD_LLVM_PATH=$1
  mkdir -p $INSTALL_PATH/tpuc/python
  mkdir -p $INSTALL_PATH/tpuc/lib
  cp -rf $BUILD_LLVM_PATH/python/* $INSTALL_PATH/tpuc/python/
  cp -rf $BUILD_LLVM_PATH/lib/libMLIRPublicAPI* $INSTALL_PATH/tpuc/lib
}

get_os_version_id()
{
  local ver_id="$(lsb_release -sr)"
  major=$(echo $ver_id | cut -d. -f1)
  minor=$(echo $ver_id | cut -d. -f2)

  OS_VER_ID=$major$minor
}

build_install_llvm()
{
  valid_cached_path=0
  valid_cached_llvm_exec=0
  cleaned_llvm_source=0

  LLVM_SRC_DIR=$PROJECT_ROOT/third_party/llvm-project
  BUILD_MLIR_DIR=$BUILD_PATH/llvm/lib/cmake/mlir
  BUILD_LLVM_DIR=$BUILD_PATH/llvm/lib/cmake/llvm

  is_valid_cached_path

  if [ $valid_cached_path = "1" ]; then
    echo "  BUILD_CACHED_PATH $BUILD_CACHED_PATH"

    get_llvm_version
    get_os_version_id
    echo "  LLVM_VER $LLVM_VER"
    echo "  OS_VER_ID $OS_VER_ID"
    CACHED_LLVM_BUILD_PATH=$BUILD_CACHED_PATH/"build_llvm_"$LLVM_VER"_"$OS_VER_ID

    is_clean_llvm_source

    if [ $cleaned_llvm_source = "1" ]; then
      is_valid_cached_llvm $CACHED_LLVM_BUILD_PATH
      if [ $valid_cached_llvm_exec = "0" ]; then
        CACHED_LLVM_SRC_DIR=$BUILD_CACHED_PATH/"llvm_"$LLVM_VER"_"$OS_VER_ID
        if [ ! -d $CACHED_LLVM_SRC_DIR ]; then
          echo "  start cached build from source"
          cp -a $LLVM_SRC_DIR $CACHED_LLVM_SRC_DIR
          build_llvm_fn $CACHED_LLVM_SRC_DIR $CACHED_LLVM_BUILD_PATH
        fi
      fi
    fi

    is_valid_cached_llvm $CACHED_LLVM_BUILD_PATH
  fi

  if [ $valid_cached_llvm_exec = "1" ]; then
    echo "  install llvm from cached"
    install_llvm_fun $CACHED_LLVM_BUILD_PATH

    mkdir -p $BUILD_PATH/llvm
    mkdir -p $BUILD_PATH/llvm/bin
    cp $CACHED_LLVM_BUILD_PATH/bin/llvm-symbolizer $BUILD_PATH/llvm/bin/

    BUILD_MLIR_DIR=$CACHED_LLVM_BUILD_PATH/lib/cmake/mlir
    BUILD_LLVM_DIR=$CACHED_LLVM_BUILD_PATH/lib/cmake/llvm
  else
    echo "  start local build from source"
    build_llvm_fn $LLVM_SRC_DIR $BUILD_PATH/llvm
    install_llvm_fun $BUILD_PATH/llvm
  fi

  echo "BUILD_MLIR_DIR $BUILD_MLIR_DIR"
  echo "BUILD_LLVM_DIR $BUILD_LLVM_DIR"
}

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
build_install_llvm

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
    -DCVIKERNEL_PATH=$INSTALL_PATH/cvikernel \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/tpuc \
    -Dpybind11_DIR=$INSTALL_PATH/pybind11/share/cmake/pybind11 \
    -DMLIR_DIR=$BUILD_MLIR_DIR \
    -DLLVM_DIR=$BUILD_LLVM_DIR \
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
cp $CVI_PY_TOOLKIT/inference/tensorflow/*.py $TPU_PYTHON_PATH/
cp $CVI_PY_TOOLKIT/inference/tflite_int8/*.py $TPU_PYTHON_PATH/
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
rm -f $INSTALL_PATH/tpuc/README.md
rm -f $INSTALL_PATH/tpuc/envs_tpu_sdk.sh
rm -f $INSTALL_PATH/tpuc/regression_models.sh
rm -f $INSTALL_PATH/tpuc/regression_models_e2e.sh
rm -f $INSTALL_PATH/tpuc/regression_models_fused_preprocess.sh
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
  rm -f $INSTALL_PATH/tpuc/regression_models.sh
  rm -f $INSTALL_PATH/tpuc/regression_models_e2e.sh
  rm -f $INSTALL_PATH/tpuc/regression_models_fused_preprocess.sh
  # rm all test prgram
  rm -f $INSTALL_PATH/tpuc/bin/test_*
  rm -f $INSTALL_PATH/tpuc/bin/sample_*
  # strip mlir tools
  pushd $INSTALL_PATH/tpuc/
  find ./ -name "*.so" |xargs strip
  find ./ -name "*.a" |xargs rm
  popd
  pushd $INSTALL_PATH/tpuc/bin
  find ./ -type f ! -name "*.html" |xargs strip
  ln -sf tpuc-opt mlir-opt
  ln -sf tpuc-interpreter mlir-tpu-interpreter
  ln -sf tpuc-translate mlir-translate
  popd

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

fi
