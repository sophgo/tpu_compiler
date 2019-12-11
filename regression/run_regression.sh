#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e regression_out ]; then
  mkdir regression_out
fi

pushd regression_out
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/resnet50/regression_resnet50_1_fp32.sh
$DIR/resnet50/regression_resnet50_2_int8.sh
$DIR/resnet50/regression_resnet50_3_int8_cmdbuf.sh
$DIR/resnet50/regression_resnet50_4_bf16.sh
$DIR/resnet50/regression_resnet50_5_bf16_cmdbuf.sh

$DIR/mobilenet_v1/regression_mobilenet_v1_1_fp32.sh

$DIR/mobilenet_v2/regression_mobilenet_v2_1_fp32.sh
$DIR/mobilenet_v2/regression_mobilenet_v2_2_int8.sh
$DIR/mobilenet_v2/regression_mobilenet_v2_3_int8_cmdbuf.sh
$DIR/mobilenet_v2/regression_mobilenet_v2_4_bf16.sh
$DIR/mobilenet_v2/regression_mobilenet_v2_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
