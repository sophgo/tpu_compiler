#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e regression_out ]; then
  mkdir regression_out
fi

pushd regression_out
# clear previous output
rm -f *.mlir *.bin *.npz *.csv

$DIR/resnet50/regression_resnet50.sh
$DIR/yolo_v3/regression_yolo_v3.sh

popd

# VERDICT
echo $0 PASSED
