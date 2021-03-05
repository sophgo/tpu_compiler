#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export WORKING_PATH=${WORKING_PATH:-$DIR/regression_out}
export CVIMODEL_REL_PATH=${WORKING_PATH}/cvimodel_regression
if [ $SET_CHIP_NAME == "cv182x" ]; then
  export CVIMODEL_REL_PATH=${WORKING_PATH}/cvimodel_regression_cv182x
fi

echo "WORKING_PATH: ${WORKING_PATH}"
echo "CVIMODEL_REL_PATH: ${CVIMODEL_REL_PATH}"

if [ -z "$2" ]; then
  DO_BATCHSIZE=1
else
  echo "$0 DO_BATCHSIZE=$DO_BATCHSIZE"
  DO_BATCHSIZE=$2
fi

export DO_BATCHSIZE=$DO_BATCHSIZE
export NET=$NET
source $DIR/generic_models.sh

WORKDIR=${WORKING_PATH}/${NET}_bs${DO_BATCHSIZE}
mkdir -p $WORKDIR
pushd $WORKDIR

# run tests
if [ $INT8_MODEL -eq 1 ]; then
  /bin/bash $INT8_INFERENCE_SCRIPT
  $DIR/regression_9_int8_quantized_model.sh
else
  /bin/bash $FP32_INFERENCE_SCRIPT
  $DIR/regression_1_fp32.sh
  if [ $DO_QUANT_INT8 -eq 1 ]; then
    if [ $DO_CALIBRATION -eq 1 ]; then
      $DIR/regression_2_int8_calibration.sh
    fi
    $DIR/regression_3_int8.sh
    if [ $DO_FUSED_PREPROCESS -eq 1 ]; then
      $DIR/regression_4_int8_fuse_preprocess.sh
      if [ $DO_YUV420_FUSED_PREPROCESS -eq 1 ]; then
        $DIR/regression_5_int8_yuv420_fuse_preprocess.sh
      fi
    fi
  fi
  if [ $DO_QUANT_BF16 -eq 1 ]; then
    $DIR/regression_6_bf16.sh
    if [ $DO_FUSED_PREPROCESS -eq 1 ]; then
      $DIR/regression_7_bf16_fuse_preprocess.sh
    fi
  fi
  if [ $DO_QUANT_MIX -eq 1 ]; then
    $DIR/regression_8_mix.sh
  fi
fi
popd

if [ ! -z $CLEAN_WORKDIR ] && [ $CLEAN_WORKDIR -eq 1 ]; then
  echo "#### rm workspace $WORKDIR"
  rm -rf $WORKDIR
fi

unset DO_BATCHSIZE
unset NET

# VERDICT
echo $0 PASSED
