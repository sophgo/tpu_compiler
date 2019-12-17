#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export MLIR_SRC_PATH=$SCRIPT_DIR
export TPU_BASE=$MLIR_SRC_PATH/../../../..
# export MLIR_SRC_PATH=$TPU_BASE/llvm-project/llvm/projects/mlir
export DATA_PATH=$MLIR_SRC_PATH/data

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

if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH=~/data/dataset
fi
export DATASET_PATH=$DATASET_PATH

export CAFFE_PATH=$TPU_BASE/install_caffe
export MKLDNN_PATH=$TPU_BASE/install_mkldnn
export BMKERNEL_PATH=$TPU_BASE/install_bmkernel
export CMODEL_PATH=$TPU_BASE/install_cmodel
export SUPPORT_PATH=$TPU_BASE/install_support
export BMBUILDER_PATH=$TPU_BASE/install_bmbuilder
export RUNTIME_PATH=$TPU_BASE/install_runtime

export MLIR_PATH=$TPU_BASE/install_mlir

export LD_LIBRARY_PATH=$BMKERNEL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CMODEL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SUPPORT_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$BMBUILDER_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLDNN_PATH/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CAFFE_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CAFFE_PATH/mkl/mklml/lib:$LD_LIBRARY_PATH

export PATH=$MLIR_SRC_PATH/externals/python_tools:$PATH
export PATH=$TPU_BASE/llvm-project/build/bin:$PATH

# for pymlir wrapper and tools
export PYTHONPATH=$TPU_BASE/llvm-project/build/lib:$PYTHONPATH
export PATH=$MLIR_SRC_PATH/bindings/python/tools:$PATH
