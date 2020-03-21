#!/bin/bash
set -e

NET=retinaface_res50
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *threshold_table *.txt

# run tests
$DIR/regression_retinaface_res50_0_caffe.sh
$DIR/regression_retinaface_res50_1_fp32.sh
$DIR/regression_retinaface_res50_2_int8.sh
$DIR/regression_retinaface_res50_3_int8_cmdbuf.sh
$DIR/regression_retinaface_res50_3_int8_cmdbuf_with_detection.sh

popd

# VERDICT
echo $0 PASSED
