#!/bin/bash
set -e

NET=squeezenet
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
#rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_squeezenet_0_caffe.sh
$DIR/regression_squeezenet_1_fp32.sh
$DIR/regression_squeezenet_2_int8.sh
$DIR/regression_squeezenet_3_int8_cmdbuf.sh
$DIR/regression_squeezenet_4_int8_cmdbuf_deepfusion.sh
$DIR/regression_squeezenet_5_bf16.sh
$DIR/regression_squeezenet_6_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
