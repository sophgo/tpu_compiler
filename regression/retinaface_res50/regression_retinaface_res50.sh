#!/bin/bash
set -e

NET=retinaface_res50
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

pushd $NET
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_retinaface_res50_fp32.sh

popd

# VERDICT
echo $0 PASSED
