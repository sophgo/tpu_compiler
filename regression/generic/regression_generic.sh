#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RUN_IN_PARALLEL=0
if [ ! -z "$RUN_IN_PARALLEL" ]; then
  RUN_IN_PARALLEL=$RUN_IN_PARALLEL
fi

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
$DIR/regression_2_int8_calibration.sh

if [ $RUN_IN_PARALLEL -eq 1 ]; then
  parallel -j4 --delay 2.5 --joblog job_$NET.log /bin/bash {} ::: \
    $DIR/regression_3_int8_per_tensor.sh \
    $DIR/regression_3_int8_rshift_only.sh \
    $DIR/regression_3_int8_multiplier.sh \
    $DIR/regression_6_bf16.sh
else
  $DIR/regression_3_int8_per_tensor.sh
  $DIR/regression_3_int8_rshift_only.sh
  $DIR/regression_3_int8_multiplier.sh
  $DIR/regression_6_bf16.sh
fi

if [ $DO_DEEPFUSION -eq 1 ]; then
  $DIR/regression_4_int8_cmdbuf_deepfusion.sh
fi
if [ $DO_MEMOPT -eq 1 ]; then
  $DIR/regression_4_int8_cmdbuf_memopt.sh
fi
if [ $DO_LAYERGROUP -eq 1 ]; then
  $DIR/regression_5_int8_cmdbuf_layergroup.sh
fi
if [ $DO_QUANT_MIX -eq 1 ]; then
  $DIR/regression_7_mix.sh
fi
if [ $DO_E2E -eq 1 ]; then
  $DIR/regression_e2e.sh
fi
popd

# VERDICT
echo $0 PASSED
