#!/bin/bash
set -e

NET=resnet50
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
# rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_resnet50_0_caffe.sh
$DIR/regression_resnet50_1_fp32.sh
$DIR/regression_resnet50_2_int8.sh
$DIR/regression_resnet50_3_int8_cmdbuf.sh
$DIR/regression_resnet50_4_bf16.sh
$DIR/regression_resnet50_5_bf16_cmdbuf.sh
# $DIR/regression_resnet50_6_int8_cmdbuf_deepfusion.sh

popd

# VERDICT
echo $0 PASSED
