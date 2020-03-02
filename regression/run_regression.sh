#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# please keep in alphabetical order
net_list=(
  "resnet50"
  "yolo_v3"
  # "inception_v4"
  "ssd300"
  "retinaface_res50"
  "retinaface_mnet25"
  "efficientnet_b0"
)

generic_net_list=(
  # "resnet50"
  "vgg16"
  # "inception_v3"
  "inception_v4"
  # "efficientnet_b0"
  # "mobilenet_v2"
)

if [ ! -z "$1" ]; then
  $DIR/$1/regression_$1.sh
  echo "regression $1 PASSED"
  exit 0
fi

if [ ! -e regression_out ]; then
  mkdir regression_out
fi

ERR=0
pushd regression_out
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *.cvimodel

# normal
for net in ${net_list[@]}
do
  echo "regression $net"
  $DIR/$net/regression_$net.sh 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net FAILED" >> verdict.log
    ERR=1
  else
    echo "$net PASSED" >> verdict.log
  fi
done

# generic
for net in ${generic_net_list[@]}
do
  echo "regression $net"
  $DIR/generic/regression_generic.sh $net 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net FAILED" >> verdict.log
    ERR=1
  else
    echo "$net PASSED" >> verdict.log
  fi
done

popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi
