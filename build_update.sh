#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/envsetup.sh

# build caffe
pushd $BUILD_PATH/build_caffe
cmake --build . --target install
popd

# build flatbuffers
pushd $BUILD_PATH/build_flatbuffers
cmake --build . --target install
popd
cp -a $MLIR_SRC_PATH/third_party/flatbuffers/python $FLATBUFFERS_PATH/

# build cvikernel
pushd $BUILD_PATH/build_cvikernel
cmake --build . --target install
popd

# build mlir-tpu
pushd $BUILD_PATH
cmake --build . --target check-mlir
cmake --build . --target pymlir
cmake --build . --target pybind
cmake --build . --target install
cp lib/pymlir.so $TPU_PYTHON_PATH
cp lib/pybind.so $TPU_PYTHON_PATH
popd
cp $MLIR_SRC_PATH/bindings/python/tools/*.py $TPU_PYTHON_PATH/
# python utils
cp -a $MLIR_SRC_PATH/python/utils/* $TPU_PYTHON_PATH/
pushd $TPU_PYTHON_PATH/model/retinaface; make; popd
# calibration tool
pushd $BUILD_PATH/build_calibration
cmake $MLIR_SRC_PATH/python/calibration && make
cp calibration_math.so $INSTALL_PATH/lib
popd
cp $MLIR_SRC_PATH/python/calibration/*.py $TPU_PYTHON_PATH/

# cvibuilder
pushd $BUILD_PATH/build_cvimodel/include
$INSTALL_PATH/flatbuffers/bin/flatc --cpp --gen-object-api \
    $MLIR_SRC_PATH/externals/cvibuilder/src/cvimodel.fbs
popd
cp -a $MLIR_SRC_PATH/externals/cvibuilder/python/* $TPU_PYTHON_PATH/

# build cmodel
pushd $BUILD_PATH/build_cmodel
cmake --build . --target install
popd

# build runtime
pushd $BUILD_PATH/build_runtime
cmake --build . --target install
popd

# build profiling
pushd $BUILD_PATH/build_profiling
cmake --build . --target install
popd
cp $MLIR_SRC_PATH/externals/profiling/tool/performance.html $PROFILING_PATH/bin/

# SoC build
# TODO
