#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
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
source $DIR/cvitek_zoo_models.sh

pushd $NET
# clear previous output
# rm -f *.mlir *.bin *.npz *.csv *.cvimodel *.npy

# run tests
/bin/bash $FP32_INFERENCE_SCRIPT
$REGRESSION_PATH/generic/regression_1_fp32.sh
$REGRESSION_PATH/generic/regression_2_int8_calibration.sh

$REGRESSION_PATH/generic/regression_3_int8_per_tensor.sh
$REGRESSION_PATH/generic/regression_3_int8_rshift_only.sh
$REGRESSION_PATH/generic/regression_3_int8_multiplier.sh
$REGRESSION_PATH/generic/regression_6_bf16.sh

if [ $DO_DEEPFUSION -eq 1 ]; then
  $REGRESSION_PATH/generic/regression_4_int8_cmdbuf_deepfusion.sh
fi
if [ $DO_MEMOPT -eq 1 ]; then
  $REGRESSION_PATH/generic/regression_4_int8_cmdbuf_memopt.sh
fi
if [ $DO_LAYERGROUP -eq 1 ]; then
  $REGRESSION_PATH/generic/regression_5_int8_cmdbuf_layergroup.sh
fi
if [ $DO_QUANT_MIX -eq 1 ]; then
  $REGRESSION_PATH/generic/regression_7_mix.sh
fi
if [ $DO_E2E -eq 1 ]; then
  $REGRESSION_PATH/generic/regression_e2e.sh
fi
popd

# VERDICT
echo $0 PASSED
