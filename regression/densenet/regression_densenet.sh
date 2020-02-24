#!/bin/bash
set -e

NET=densenet
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_densenet_0_caffe.sh
$DIR/regression_densenet_1_fp32.sh
$DIR/regression_densenet_2_int8.sh
$DIR/regression_densenet_3_int8_cmdbuf.sh
$DIR/regression_densenet_4_fp16.sh
$DIR/regression_densenet_5_fp16_cmdbuf.sh
popd

# VERDICT
echo $0 PASSED