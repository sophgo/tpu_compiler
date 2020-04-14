#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export MLIR_PATH=$SCRIPT_DIR

# set MODEL_PATH
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH=~/data/models
fi
if [ ! -e $MODEL_PATH ]; then
  echo "MODEL_PATH $MODEL_PATH does not exist"
  echo "  Please export MODEL_PATH='YOUR_MODEL_PATH'"
  echo "  Or ln -s 'YOUR_MODEL_PATH' ~/data/models"
  echo "  Please read README.md in each regression dirs on where to download the models"
  return 1
fi
export MODEL_PATH=$MODEL_PATH

# set DATASET_PATH
if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH=~/data/dataset
fi
export DATASET_PATH=$DATASET_PATH

# python path
export TPU_PYTHON_PATH=$MLIR_PATH/python

# regression path
export REGRESSION_PATH=$MLIR_PATH/regression

# run path
export PATH=$MLIR_PATH/bin:$PATH
export PATH=$MLIR_PATH/python:$PATH
export PATH=$MLIR_PATH/regression:$PATH
export PATH=$MLIR_PATH/regression/generic:$PATH

export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/caffe/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/flatbuffers/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MLIR_PATH/systemc-2.3.3/lib-linux64:$LD_LIBRARY_PATH

export PYTHONPATH=$TPU_PYTHON_PATH:$PYTHONPATH
export PYTHONPATH=$MLIR_PATH/caffe/python:$PYTHONPATH
export PYTHONPATH=$MLIR_PATH/flatbuffers/python:$PYTHONPATH

export GLOG_minloglevel=2
