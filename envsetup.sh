#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}
export MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/../models}
export DATASET_PATH=${DATASET_PATH:-$PROJECT_ROOT/../dataset}

echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "BUILD_PATH   : ${BUILD_PATH}"
echo "INSTALL_PATH : ${INSTALL_PATH}"
echo "MODEL_PATH   : ${MODEL_PATH}"
echo "DATASET_PATH : ${DATASET_PATH}"

# python path
export TPU_PYTHON_PATH=$INSTALL_PATH/tpuc/python
# regression path
export REGRESSION_PATH=$PROJECT_ROOT/regression
# run path
export PATH=$INSTALL_PATH/tpuc/bin:$PATH
export PATH=$INSTALL_PATH/tpuc/python:$PATH
export PATH=$INSTALL_PATH/tpuc/python/cvi_toolkit:$PATH
export PATH=$INSTALL_PATH/tpuc/python/cvi_toolkit/eval:$PATH
export PATH=$INSTALL_PATH/tpuc/python/cvi_toolkit/tool:$PATH
export PATH=$INSTALL_PATH/tpuc/python/cvi_toolkit/performance/performance_viewer:$PATH
export PATH=$PROJECT_ROOT:$PATH
export PATH=$PROJECT_ROOT/regression:$PATH
export PATH=$PROJECT_ROOT/regression/generic:$PATH
export PATH=$PROJECT_ROOT/build/llvm/bin:$PATH
export PATH=$PROJECT_ROOT/python/cvi_toolkit/test/onnx_ir_test:$PATH
export PATH=$PROJECT_ROOT/python/cvi_toolkit/test/torch_ir_test:$PATH

export LD_LIBRARY_PATH=$INSTALL_PATH/tpuc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/caffe/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/opencv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/systemc/lib-linux64:$LD_LIBRARY_PATH

export PYTHONPATH=$TPU_PYTHON_PATH:$PYTHONPATH
export PYTHONPATH=$INSTALL_PATH/caffe/python:$PYTHONPATH
export PYTHONPATH=$INSTALL_PATH/flatbuffers/python:$PYTHONPATH

export GLOG_minloglevel=2
export SET_CHIP_NAME="cv183x"
# export OMP_NUM_THREADS=$(nproc --all)
export OMP_NUM_THREADS=4
