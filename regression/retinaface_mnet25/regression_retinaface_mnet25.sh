#!/bin/bash
set -e

NET=retinaface_mnet25
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *threshold_table *.txt

# run tests
$DIR/regression_retinaface_mnet25_0_caffe.sh
$DIR/regression_retinaface_mnet25_1_fp32.sh
$DIR/regression_retinaface_mnet25_2_int8.sh

popd

# VERDICT
echo $0 PASSED
