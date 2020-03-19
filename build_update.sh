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

pushd $TPU_PYTHON_PATH/model/retinaface; make; popd
# calibration tool
pushd $BUILD_PATH/build_calibration
cmake $MLIR_SRC_PATH/python/calibration && make
cp calibration_math.so $INSTALL_PATH/lib
popd

# python package
cp -ar $MLIR_SRC_PATH/python/dataset_util $TPU_PYTHON_PATH/
cp -ar $MLIR_SRC_PATH/python/model $TPU_PYTHON_PATH/
cp -ar $MLIR_SRC_PATH/python/transform $TPU_PYTHON_PATH/
cp -ar $MLIR_SRC_PATH/python/utils $TPU_PYTHON_PATH/
cp -ar $MLIR_SRC_PATH/python/numpy_helper $TPU_PYTHON_PATH/

# python script
cp $MLIR_SRC_PATH/python/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/binary_helper/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/calibration/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/eval/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/inference/caffe/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/inference/mlir/*.py $TPU_PYTHON_PATH/
cp $MLIR_SRC_PATH/python/inference/onnx/*.py $TPU_PYTHON_PATH/

# cvibuilder
pushd $BUILD_PATH/build_cvimodel
cmake --build . --target install
popd

# build cmodel
pushd $BUILD_PATH/build_cmodel
cmake --build . --target install
popd

# build cviruntime
pushd $BUILD_PATH/build_cviruntime
cmake --build . --target install
popd

# build profiling
pushd $BUILD_PATH/build_profiling
cmake --build . --target install
popd
cp $MLIR_SRC_PATH/externals/profiling/tool/performance.html $PROFILING_PATH/bin/

