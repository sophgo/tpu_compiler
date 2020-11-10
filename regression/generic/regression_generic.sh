#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z "$CVIMODEL_REL_PATH" ]; then
  CVIMODEL_REL_PATH=$PWD/cvimodel_regression
fi
export CVIMODEL_REL_PATH=$CVIMODEL_REL_PATH
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

if [ -z "$2" ]; then
  DO_BATCHSIZE=1
else
  echo "$0 DO_BATCHSIZE=$DO_BATCHSIZE"
  DO_BATCHSIZE=$2
fi

export DO_BATCHSIZE=$DO_BATCHSIZE
export NET=$NET
source $DIR/generic_models.sh

WORKDIR=${NET}_bs${DO_BATCHSIZE}

if [ ! -e $WORKDIR ]; then
  mkdir $WORKDIR
fi
pushd $WORKDIR

# run tests
/bin/bash $FP32_INFERENCE_SCRIPT
$DIR/regression_1_fp32.sh
if [ $DO_QUANT_INT8 -eq 1 ]; then
  if [ $DO_CALIBRATION -eq 1 ]; then
    $DIR/regression_2_int8_calibration.sh
  fi
  $DIR/regression_3_int8.sh
fi
if [ $DO_QUANT_BF16 -eq 1 ]; then
  $DIR/regression_4_bf16.sh
fi
if [ $DO_QUANT_MIX -eq 1 ]; then
  $DIR/regression_5_mix.sh
fi
if [ $DO_FUSED_PREPROCESS -eq 1 ]; then
  $DIR/regression_6_fuse_preprocess.sh
  if [ $DO_QUANT_BF16 -eq 1 ]; then
    $DIR/regression_7_fuse_preprocess_bf16.sh
  fi
fi
if [ $DO_TPU_SOFTMAX_INFERENCE -eq 1 ]; then
  $DIR/regression_8_tpu_softmax.sh
fi
#if [ $DO_NN_TOOLKIT -eq 1 ]; then
#  gen_cvi_nn_tool_template.py $NET
#  cvi_nn_converter.py $NET.yml
#fi
popd

unset DO_BATCHSIZE
unset NET

# VERDICT
echo $0 PASSED
