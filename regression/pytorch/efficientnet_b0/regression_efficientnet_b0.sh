#!/bin/bash
set -e

NET=efficientnet_b0_pytorch
MODEL=$MODEL_PATH/imagenet/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
NEED_REMOVE_AFTER_FIX_CPU_LAYER=1
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

#run tests
$DIR/regression_efficientnet_b0_0_onnx.sh ${NET} ${MODEL}
$DIR/regression_efficientnet_b0_1_fp32.sh
$DIR/regression_efficientnet_b0_2_int8.sh
$DIR/regression_efficientnet_b0_3_int8_cmdbuf.sh
popd

# VERDICT
echo $0 PASSED
