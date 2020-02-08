#!/bin/bash
set -e

NET=mobilenet_v2
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_mobilenet_v2_0_caffe.sh
$DIR/regression_mobilenet_v2_1_fp32.sh
$DIR/regression_mobilenet_v2_2_int8.sh
$DIR/regression_mobilenet_v2_3_int8_cmdbuf.sh
$DIR/regression_mobilenet_v2_4_bf16.sh
$DIR/regression_mobilenet_v2_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
