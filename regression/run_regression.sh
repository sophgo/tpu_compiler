#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# please keep in alphabetical order
net_list=(
  # "resnet50"
)

generic_net_list=(
  "resnet50"
  "vgg16"
  "mobilenet_v2"
  "googlenet"
  "inception_v3"
  "inception_v4"
  "shufflenet_v2"
  "squeezenet"
  "arcface_res50"
  "retinaface_mnet25"
  "retinaface_res50"
  "ssd300"
  "yolo_v3_416"
  "yolo_v3_320"
  "resnet18"
  "efficientnet_b0"
  "alphapose"
)

generic_accuracy_net_list=(
  # "resnet50"
  # "mobilenet_v2"
)

ERR=0

helpFunction()
{
   echo ""
   echo "Usage: $0 -batch Batchsize"
   echo -e "\t-b Description of batch size for test"
   echo -e "\t-n Description of Net Name for test "
   exit 1
}

while getopts "n:b:" opt
do
  case "$opt" in
    n ) net="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    h ) helpFunction ;;
  esac
done

if [ -z "$net" ]; then
  net=$1
fi
if [ -z "$bs" ]; then
  bs=1
fi

if [ ! -z "$net" ]; then
  export CVIMODEL_REL_PATH=$PWD/cvimodel_regression
  if [ ! -e $CVIMODEL_REL_PATH ]; then
    mkdir $CVIMODEL_REL_PATH
  fi
  $DIR/$net/regression_$net.sh 2>&1 | tee $net.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "regression $net FAILED"
    ERR=1
  else
    echo "regression $net PASSED"
  fi
  exit $ERR
fi

if [ ! -e regression_out ]; then
  mkdir regression_out
fi
export CVIMODEL_REL_PATH=$PWD/regression_out/cvimodel_regression
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
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
  echo "generic regression $net batch=$bs"
  $DIR/generic/regression_generic.sh $net $bs 2>&1 | tee $net\_bs$bs.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net batch=$bs generic regression FAILED" >> verdict.log
    ERR=1
  else
    echo "$net batch=$bs generic regression PASSED" >> verdict.log
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
