#!/bin/bash
set -e

NET=vgg16
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_vgg16_0_caffe.sh
$DIR/regression_vgg16_1_fp32.sh
$DIR/regression_vgg16_2_int8.sh
$DIR/regression_vgg16_3_int8_cmdbuf.sh
$DIR/regression_vgg16_4_bf16.sh
$DIR/regression_vgg16_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
