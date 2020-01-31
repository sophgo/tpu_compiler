#!/bin/bash
set -e

NET=liveness
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_liveness_0_fp32.sh
$DIR/regression_liveness_1_int8.sh
$DIR/regression_liveness_2_int8_cmdbuf.sh
#$DIR/regression_resnet50_4_bf16.sh
#$DIR/regression_resnet50_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
