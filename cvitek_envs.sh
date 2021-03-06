#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export MLIR_PATH=$SCRIPT_DIR

# set MODEL_PATH
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH=~/data/models
fi

export MODEL_PATH=$MODEL_PATH

# set DATASET_PATH
if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH=~/data/dataset
fi
export DATASET_PATH=$DATASET_PATH

# python path
export TPU_PYTHON_PATH=$MLIR_PATH/tpuc/python

# regression path
export REGRESSION_PATH=$MLIR_PATH/tpuc/regression

# run path
export PATH=$MLIR_PATH/tpuc/bin:$PATH
export PATH=$MLIR_PATH/tpuc/python:$PATH
export PATH=$MLIR_PATH/tpuc/python/cvi_toolkit:$PATH
export PATH=$MLIR_PATH/tpuc/python/cvi_toolkit/eval:$PATH
export PATH=$MLIR_PATH/tpuc/python/cvi_toolkit/tool:$PATH
export PATH=$MLIR_PATH/tpuc/python/cvi_toolkit/performance/performance_viewer:$PATH
export PATH=$MLIR_PATH/tpuc/regression:$PATH
export PATH=$MLIR_PATH/tpuc/regression/generic:$PATH
export PATH=$MLIR_PATH/tpuc/python/cvi_toolkit/test:$PATH

export LD_LIBRARY_PATH=$MLIR_PATH/tpuc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/caffe/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/flatbuffers/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/systemc-2.3.3/lib-linux64:$LD_LIBRARY_PATH

export PYTHONPATH=$TPU_PYTHON_PATH:$PYTHONPATH
export PYTHONPATH=$MLIR_PATH/caffe/python:$PYTHONPATH
export PYTHONPATH=$MLIR_PATH/flatbuffers/python:$PYTHONPATH

export GLOG_minloglevel=2
export SET_CHIP_NAME="cv183x"
# export OMP_NUM_THREADS=$(nproc --all)
export OMP_NUM_THREADS=4
