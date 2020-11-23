#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PROJECT_ROOT=$SCRIPT_DIR
export PROJECT_BASE=$PROJECT_ROOT/../../../..
export OMP_NUM_THREADS=`grep 'core id' /proc/cpuinfo | sort -u | wc -l`
export INSTALL_PATH=$PROJECT_ROOT/install

# set MODEL_PATH
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH=$PROJECT_BASE/models
fi
if [ ! -e $MODEL_PATH ]; then
  echo "MODEL_PATH $MODEL_PATH does not exist"
  echo "  Please export MODEL_PATH='YOUR_MODEL_PATH'"
  echo "  Or ln -s 'YOUR_MODEL_PATH' ~/data/models"
  echo "  Please read README.md in each regression dirs on where to download the models"
  return 1
fi
echo "MODEL_PATH set to $MODEL_PATH"
export MODEL_PATH=$MODEL_PATH

# set DATASET_PATH
if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH=~/data/dataset
fi
echo "DATASET_PATH set to $DATASET_PATH"
export DATASET_PATH=$DATASET_PATH

# set PATH for all projects
export MKLDNN_PATH=$INSTALL_PATH/mkldnn
export CAFFE_PATH=$INSTALL_PATH/caffe
export OPENCV_PATH=$INSTALL_PATH/opencv
export FLATBUFFERS_PATH=$INSTALL_PATH/flatbuffers
export SYSTEMC_PATH=$INSTALL_PATH/systemc-2.3.3
export CVIKERNEL_PATH=$INSTALL_PATH
export CMODEL_PATH=$INSTALL_PATH
export RUNTIME_PATH=$INSTALL_PATH
export PROFILING_PATH=$INSTALL_PATH
export MLIR_PATH=$INSTALL_PATH

# set build python version
export PYTHON_VERSION=3

# python path
export TPU_PYTHON_PATH=$INSTALL_PATH/tpuc/python

# regression path
export REGRESSION_PATH=$PROJECT_ROOT/regression

# run path
export PATH=$INSTALL_PATH/tpuc/bin:$PATH
export PATH=$INSTALL_PATH/tpuc/python:$PATH
export PATH=$PROJECT_ROOT:$PATH
export PATH=$PROJECT_ROOT/regression:$PATH
export PATH=$PROJECT_ROOT/regression/generic:$PATH

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
