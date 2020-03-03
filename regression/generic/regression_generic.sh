#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

export NET=$NET
source $DIR/generic_models.sh

pushd $NET
# clear previous output
# rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_0_caffe.sh
$DIR/regression_1_fp32.sh
if [ $DO_QUANT_INT8 -eq 1 ]; then
  $DIR/regression_2_int8.sh
  if [ $DO_CMDBUF_INT8 -eq 1 ]; then
    $DIR/regression_3_int8_cmdbuf.sh
  fi
fi
if [ $DO_QUANT_BF16 -eq 1 ]; then
  $DIR/regression_4_bf16.sh
  if [ $DO_CMDBUF_BF16 -eq 1 ]; then
    $DIR/regression_5_bf16_cmdbuf.sh
  fi
fi
if [ $DO_DEEPFUSION -eq 1 ]; then
  $DIR/regression_6_int8_cmdbuf_deepfusion.sh
fi

popd

# VERDICT
echo $0 PASSED
