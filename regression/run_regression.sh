#!/bin/bash
# set -e
# set -o pipefail

net_list_generic=(
  "resnet50"
  "vgg16"
  "mobilenet_v1"
  "mobilenet_v2"
  "googlenet"
  "inception_v3"
  "inception_v4"
  "squeezenet"
  "shufflenet_v2"
  "densenet_121"
  "densenet_201"
  # "senet_res50"
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

net_list_batch=(
  "resnet50"
  "mobilenet_v2"
)

net_list_accuracy=(
  # "resnet50"
  # "mobilenet_v2"
)

usage()
{
   echo ""
   echo "Usage: $0 -batch Batchsize"
   echo -e "\t-b Description of batch size for test"
   echo -e "\t-n Description of Net Name for test "
   exit 1
}

run_generic()
{
  net=$1
  bs=$2
  echo "generic regression $net batch=$bs"
  regression_generic.sh $net $bs > $1\_bs$bs.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net batch=$bs generic regression FAILED" >> verdict.log
    return 1
  else
    echo "$net batch=$bs generic regression PASSED" >> verdict.log
    return 0
  fi
}
export -f run_generic

run_generic_all()
{
  ERR=0
  # bs = 1
  for net in ${net_list_generic[@]}
  do
    run_generic $net 1
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  done
  # bs = 4
  for net in ${net_list_batch[@]}
  do
    run_generic $net 4
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  done
  # return
  return $ERR
}

run_generic_all_parallel()
{
  rm -f regression.txt
  for net in ${net_list_generic[@]}
  do
    echo "run_generic $net 1" >> regression.txt
  done
  for net in ${net_list_batch[@]}
  do
    echo "run_generic $net 4" >> regression.txt
  done
  cat regression.txt
  parallel -j0 --delay 0.5  --joblog job_regression.log < regression.txt
  return $?
}

run_accuracy()
{
  net=$1
  count=$2
  echo "generic accuracy $net"
  accuracy_generic.sh $net $count > accuracy_$1\_$count\.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net count=$count generic accuracy FAILED" >> verdict.log
    return 1
  else
    echo "$net count=$count generic accuracy PASSED" >> verdict.log
    return 0
  fi
}
export -f run_accuracy

run_accuracy_all()
{
  count=$1
  ERR=0
  for net in ${net_list_accuracy[@]}
  do
    run_accuracy $net $count
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  done
  return $ERR
}

run_accuracy_all_parallel()
{
  count=$1
  rm -f accuracy.txt
  for net in ${net_list_accuracy[@]}
  do
    echo "run_accuracy $net $count" >> accuracy.txt
  done
  if [ -f accuracy.txt ]; then
    cat accuracy.txt
    parallel -j0 --delay 0.5  --joblog job_accuracy.log < accuracy.txt
    return $?
  fi
}

while getopts "n:b:" opt
do
  case "$opt" in
    n ) net="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    h ) usage ;;
  esac
done

if [ -z "$net" ]; then
  net=$1
fi
if [ -z "$bs" ]; then
  bs=1
fi

# default run in parallel
if [ -z "$PARALLEL" ]; then
  PARALLEL=1
fi

# run single and exit
if [ ! -z "$net" ]; then
  export CVIMODEL_REL_PATH=$PWD/cvimodel_regression
  if [ ! -e $CVIMODEL_REL_PATH ]; then
    mkdir $CVIMODEL_REL_PATH
  fi
  run_generic $net $bs
  exit $?
fi

# run regression for all
if [ ! -e regression_out ]; then
  mkdir regression_out
fi
export CVIMODEL_REL_PATH=$PWD/regression_out/cvimodel_regression
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

pushd regression_out

ERR=0
if [ $PARALLEL -eq 0 ]; then
  run_generic_all
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
  run_accuracy_all 100
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
else
  run_generic_all_parallel
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
  run_accuracy_all_parallel 100
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
fi

popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

exit $ERR
