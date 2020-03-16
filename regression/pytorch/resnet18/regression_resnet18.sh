#!/bin/bash
set -e

NET=resnet18_pytorch
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
NEED_REMOVE_AFTER_FIX_CPU_LAYER=1
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

#run tests
$DIR/regression_resnet18_0_onnx.sh
$DIR/regression_resnet18_1_fp32.sh
$DIR/regression_resnet18_2_int8.sh
$DIR/regression_resnet18_3_int8_cmdbuf.sh
popd

# VERDICT
echo $0 PASSED
