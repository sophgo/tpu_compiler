#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export MLIR_SRC_PATH=$SCRIPT_DIR
export TPU_BASE=$MLIR_SRC_PATH/../../../..
export OMP_NUM_THREADS=`grep 'core id' /proc/cpuinfo | sort -u | wc -l`

# set MODEL_PATH
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH=$TPU_BASE/models
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

# set MLIR_INSTALL_PATH
if [[ -z "$MLIR_INSTALL_PATH" ]]; then
  MLIR_INSTALL_PATH=$TPU_BASE/cvitek_mlir
fi
echo "INSTALL_PATH set to $MLIR_INSTALL_PATH"
export INSTALL_PATH=$MLIR_INSTALL_PATH

# set BUILD_PATH
if [[ -z "$BUILD_PATH" ]]; then
  BUILD_PATH=$TPU_BASE/build_out_host
fi
echo "BUILD_PATH set to $BUILD_PATH"
export BUILD_PATH=$BUILD_PATH

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
export TPU_PYTHON_PATH=$INSTALL_PATH/python

# regression path
export REGRESSION_PATH=$MLIR_SRC_PATH/regression

# run path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$INSTALL_PATH/python:$PATH
# export PATH=$FLATBUFFERS_PATH/bin:$PATH
export PATH=$MLIR_SRC_PATH:$PATH
export PATH=$MLIR_SRC_PATH/regression:$PATH
export PATH=$MLIR_SRC_PATH/regression/generic:$PATH

export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNN_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CAFFE_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENCV_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FLATBUFFERS_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SYSTEMC_PATH/lib-linux64:$LD_LIBRARY_PATH

export PYTHONPATH=$TPU_PYTHON_PATH:$PYTHONPATH
export PYTHONPATH=$CAFFE_PATH/python:$PYTHONPATH
export PYTHONPATH=$FLATBUFFERS_PATH/python:$PYTHONPATH

# soc build and path
if [[ -z "$SDK_INSTALL_PATH" ]]; then
  SDK_INSTALL_PATH=$TPU_BASE/cvitek_tpu_sdk
fi
echo "INSTALL_SOC_PATH set to $SDK_INSTALL_PATH"
export INSTALL_SOC_PATH=$SDK_INSTALL_PATH
if [[ -z "$BUILD_SOC_PATH" ]]; then
  BUILD_SOC_PATH=$TPU_BASE/build_out_soc
fi
echo "BUILD_SOC_PATH set to $BUILD_SOC_PATH"
export BUILD_SOC_PATH=$BUILD_SOC_PATH

# soc 32bit build and path
if [[ -z "$SDK_INSTALL_PATH_32BIT" ]]; then
  SDK_INSTALL_PATH_32BIT=$TPU_BASE/cvitek_tpu_sdk_32bit
fi
echo "INSTALL_SOC_PATH_32BIT set to $SDK_INSTALL_PATH_32BIT"
export INSTALL_SOC_PATH_32BIT=$SDK_INSTALL_PATH_32BIT
if [[ -z "$BUILD_SOC_PATH_32BIT" ]]; then
  BUILD_SOC_PATH_32BIT=$TPU_BASE/build_out_soc_32bit
fi
echo "BUILD_SOC_PATH_32BIT set to $BUILD_SOC_PATH_32BIT"
export BUILD_SOC_PATH_32BIT=$BUILD_SOC_PATH_32BIT

export GLOG_minloglevel=2
