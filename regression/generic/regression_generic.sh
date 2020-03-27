#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

if [ -z "$CVIMODEL_REL_PATH" ]; then
  CVIMODEL_REL_PATH=$PWD/cvimodel_out
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

pushd $NET
# clear previous output
# rm -f *.mlir *.bin *.npz *.csv *.cvimodel *.npy

# run tests
/bin/bash $FP32_INFERENCE_SCRIPT
$DIR/regression_1_fp32.sh
$DIR/regression_2_int8_calibration.sh

parallel -j4 --delay 2.5 --joblog job_$NET.log /bin/bash {} ::: \
  $DIR/regression_3_int8_per_tensor.sh \
  $DIR/regression_3_int8_rshift_only.sh \
  $DIR/regression_3_int8_multiplier.sh \
  $DIR/regression_6_bf16.sh

if [ $DO_DEEPFUSION -eq 1 ]; then
  $DIR/regression_4_int8_cmdbuf_deepfusion.sh
fi
if [ $DO_QUANT_MIX -eq 1 ]; then
  $DIR/regression_7_mix.sh
fi
if [ $DO_LAYERGROUP -eq 1 ]; then
  $DIR/regression_5_int8_cmdbuf_layergroup.sh
fi

popd

# VERDICT
echo $0 PASSED
