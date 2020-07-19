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
  DO_BATCHSIZE=$2
fi

if [ -z "$3" ]; then
  ENABLE_PREPROCESS=0
else
  ENABLE_PREPROCESS=$3
fi
export DO_BATCHSIZE=$DO_BATCHSIZE
export ENABLE_PREPROCESS=$ENABLE_PREPROCESS
export NET=$NET
source $DIR/generic_models.sh

WORKDIR=${NET}_bs${DO_BATCHSIZE}
if [ $ENABLE_PREPROCESS -eq 1 ]; then
  if [ $DO_BATCHSIZE -ne 1 ]; then
    echo "$NET: Not support batch on preprocess for now"
    exit 1
  fi
  if [ $DO_PREPROCESS -ne 1 ]; then
    echo "$NET: Not support DO_PREPROCESS, can not ENABLE_PREPROCESS"
    exit 1
  fi
  WORKDIR=${NET}_preprocess_bs${DO_BATCHSIZE}
fi

if [ ! -e $WORKDIR ]; then
  mkdir $WORKDIR
fi
pushd $WORKDIR

# run tests
/bin/bash $FP32_INFERENCE_SCRIPT
$DIR/regression_1_fp32.sh
$DIR/regression_2_int8_calibration.sh
$DIR/regression_3_int8_per_tensor.sh
$DIR/regression_3_int8_rshift_only.sh
$DIR/regression_3_int8_multiplier.sh
if [ $DO_DEEPFUSION -eq 1 ]; then
  $DIR/regression_4_int8_cmdbuf_deepfusion.sh
fi
if [ $DO_LAYERGROUP -eq 1 ]; then
  $DIR/regression_5_int8_cmdbuf_layergroup.sh
fi
if [ $DO_QUANT_BF16 -eq 1 ]; then
  $DIR/regression_6_bf16.sh
fi
if [ $DO_E2E -eq 1 ]; then
  $DIR/regression_e2e.sh
fi
if [ $DO_NN_TOOLKIT -eq 1 ]; then
  gen_cvi_nn_tool_template.py $NET
  cvi_nn_converter.py $NET.yml
fi
if [ $DO_QUANT_MIX -eq 1 ]; then
  $DIR/regression_7_mix.sh
fi
popd

# VERDICT
echo $0 PASSED
