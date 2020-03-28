#!/bin/bash
set -e

NET=mobilenetv3_pytorch
MODEL=$MODEL_PATH/imagenet/mobilenetv3_rw/onnx/mobilenetv3_rw.onnx
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

#run tests
$DIR/regression_0_onnx.sh $NET $MODEL
$DIR/regression_1_fp32.sh $NET $MODEL
#$DIR/regression_2_int8.sh $NET
#$DIR/regression_3_int8_cmdbuf.sh $NET
popd

# VERDICT
echo $0 PASSED
