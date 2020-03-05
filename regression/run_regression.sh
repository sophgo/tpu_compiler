#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# please keep in alphabetical order
net_list=(
  "bmface_v3"
  "liveness"
  # "resnet50"
  "retinaface_mnet25"
  "retinaface_res50"
  "ssd300"
  "yolo_v3"
)

generic_net_list=(
  "resnet50"
  "vgg16"
  "mobilenet_v2"
  "inception_v3"
  "inception_v4"
  "efficientnet_b0"
  "shufflenet_v2"
)

generic_accuracy_net_list=(
  "mobilenet_v2"
  # "shufflenet_v2"
)

ERR=0

if [ ! -z "$1" ]; then
  $DIR/$1/regression_$1.sh 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "regression $1 FAILED"
    ERR=1
  else
    echo "regression $1 PASSED"
  fi
  exit $ERR
fi

if [ ! -e regression_out ]; then
  mkdir regression_out
fi

pushd regression_out
# clear previous output
rm -f *.mlir *.bin *.npz *.csv *.cvimodel

# normal
for net in ${net_list[@]}
do
  echo "regression $net"
  $DIR/$net/regression_$net.sh 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net regression FAILED" >> verdict.log
    ERR=1
  else
    echo "$net regression PASSED" >> verdict.log
  fi
done

# generic
for net in ${generic_net_list[@]}
do
  echo "generic regression $net"
  $DIR/generic/regression_generic.sh $net 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net generic regression FAILED" >> verdict.log
    ERR=1
  else
    echo "$net generic regression PASSED" >> verdict.log
  fi
done

# generic accuracy
for net in ${generic_accuracy_net_list[@]}
do
  echo "accuracy $net"
  $DIR/generic/accuracy_generic.sh $net 100 2>&1 | tee ${net}_accuracy.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net accuracy test FAILED" >> verdict.log
    ERR=1
  else
    echo "$net accuracy test PASSED" >> verdict.log
  fi
done

popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED

fi
exit $ERR
