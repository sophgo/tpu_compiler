#!/bin/bash
set -e

NET=efficientnet_b0
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *.cvimodel

# run tests
$DIR/regression_efficientnet_b0_0_caffe.sh
$DIR/regression_efficientnet_b0_1_fp32.sh
# $DIR/regression_efficientnet_b0_2_int8_per_layer.sh
$DIR/regression_efficientnet_b0_2_int8.sh
$DIR/regression_efficientnet_b0_3_int8_cmdbuf.sh
# $DIR/regression_efficientnet_b0_4_bf16.sh
# $DIR/regression_efficientnet_b0_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
