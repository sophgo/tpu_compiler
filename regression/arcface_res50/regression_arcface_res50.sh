#!/bin/bash
set -e

NET=arcface
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_arcface_res50_0_caffe.sh
$DIR/regression_arcface_res50_1_fp32.sh
$DIR/regression_arcface_res50_2_int8.sh
$DIR/regression_arcface_res50_3_int8_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
