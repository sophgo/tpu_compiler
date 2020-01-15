#!/bin/bash
set -e

NET=inception_v4
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_inception_v4_0_caffe.sh
# $DIR/regression_inception_v4_1_fp32.sh
# $DIR/regression_inception_v4_2_int8.sh
# $DIR/regression_inception_v4_3_int8_cmdbuf.sh
# $DIR/regression_inception_v4_4_bf16.sh
# $DIR/regression_inception_v4_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
