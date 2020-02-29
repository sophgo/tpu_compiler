#!/bin/bash
set -e

NET=ssd300
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
NEED_REMOVE_AFTER_FIX_CPU_LAYER=1
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
 $DIR/regression_ssd300_0_caffe.sh
 $DIR/regression_ssd300_1_fp32.sh
 $DIR/regression_ssd300_2_int8.sh
 if [ $NEED_REMOVE_AFTER_FIX_CPU_LAYER -eq 1 ]; then
 $DIR/regression_ssd300_3_int8_cmdbuf_before.sh
 fi
 $DIR/regression_ssd300_3_int8_cmdbuf.sh
 $DIR/regression_ssd300_4_bf16.sh
# $DIR/regression_ssd300_5_bf16_cmdbuf.sh

popd

# VERDICT
echo $0 PASSED
