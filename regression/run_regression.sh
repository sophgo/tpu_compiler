#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# please keep in alphabetical order
net_list=(
  "resnet50"
  "yolo_v3"
  "retinaface_res50"
  "efficientnet_b0"
  "inception_v4"
  #"ssd300"
)

if [ ! -z "$1" ]; then
  $DIR/$1/regression_$1.sh
  echo "regression $1 PASSED"
  exit 0
fi

if [ ! -e regression_out ]; then
  mkdir regression_out
fi

pushd regression_out
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *.cvimodel

for net in ${net_list[@]}
do
  echo "regression $net"
  $DIR/$net/regression_$net.sh 2>&1 | tee $net.log
done

popd

# VERDICT
echo $0 PASSED
