#!/bin/bash
# set -e
# set -o pipefail

net_list_generic=(
  "resnet50"
  # "vgg16"
  "mobilenet_v1"
  "mobilenet_v2"
  "googlenet"
  # "inception_v3"
  "inception_v4"
  "squeezenet"
  "shufflenet_v2"
  "densenet_121"
  # "densenet_201"
  # "senet_res50"
  "arcface_res50"
  "retinaface_mnet25"
  # "retinaface_res50"
  # "ssd300"
  # "yolo_v2_1080"
  # "yolo_v2_416"
  # "yolo_v3_608"
  "yolo_v3_416"
  # "yolo_v3_320"
  "resnet18"
  "efficientnet_b0"
  # "alphapose"
)

net_list_batch=(
  "resnet50"
  # "vgg16"
  # "mobilenet_v1"
  "mobilenet_v2"
  # "googlenet"
  # "inception_v3"
  # "inception_v4"
  "squeezenet"
  # "shufflenet_v2"
  # "densenet_121"
  # "densenet_201"
  # "senet_res50"
  # "arcface_res50"
  "retinaface_mnet25"
  # "retinaface_res50"
  # "ssd300"
  # "yolo_v3_416"
  # "yolo_v3_320"
  # "resnet18"
  "efficientnet_b0"
  # "alphapose"
)

net_list_accuracy=(
  # "resnet50"
  "mobilenet_v2"
)

net_list_generic_extra=(
  # "resnet50"
  "vgg16"
  # "mobilenet_v1"
  # "mobilenet_v2"
  # "googlenet"
  "inception_v3"
  # "inception_v4"
  # "squeezenet"
  # "shufflenet_v2"
  # "densenet_121"
  "densenet_201"
  "senet_res50"
  # "arcface_res50"
  # "retinaface_mnet25"
  "retinaface_mnet25_600"
  "retinaface_res50"
  "ssd300"
  "yolo_v2_1080"
  "yolo_v2_416"
  "yolo_v3_608"
  # "yolo_v3_416"
  "yolo_v3_320"
  # "resnet18"
  # "efficientnet_b0"
  "alphapose"
  "mobilenet_v3"
)

net_list_batch_extra=(
  # "resnet50"
  # "mobilenet_v2"
  "vgg16"
  "mobilenet_v1"
  "googlenet"
  "inception_v3"
  "inception_v4"
  # "squeezenet"
  "shufflenet_v2"
  "densenet_121"
  "densenet_201"
  "senet_res50"
  "arcface_res50"
  # "retinaface_mnet25"
  ## "retinaface_res50"
  "ssd300"
  ## "yolo_v2_1080"
  ## "yolo_v2_416"
  ## "yolo_v3_608"
  "yolo_v3_416"
  "yolo_v3_320"
  "resnet18"
  # "efficientnet_b0"
  "alphapose"
)

net_list_accuracy_extra=(
  "resnet50"
  ## "vgg16"
  "mobilenet_v1"
  # "mobilenet_v2"
  "googlenet"
  ### "inception_v3"  # 2015
  "inception_v4"
  "squeezenet"
  "shufflenet_v2"
  "densenet_121"
  # "densenet_201"
  "senet_res50"
  ## "arcface_res50"
  ## "retinaface_mnet25"
  ## "retinaface_res50"
  ## "ssd300"
  ## "yolo_v2_1080"
  ## "yolo_v2_416"
  "yolo_v3_608"
  "yolo_v3_416"
  "yolo_v3_320"
  ## "resnet18"
  ## "efficientnet_b0"
  ## "alphapose"
)


net_list_onnx=(
  "resnet50"
  "squeezenet"
  "vgg19"
  "sub_pixel_cnn_2016"
  "mobilenet"
  "densenet-121"
  "caffenet"
  "googlenet"
  "inception_v1"
  ## "inception_v2" # todo: not same output with onnx runtime
  "zfnet-512"
)


run_generic()
{
  local net=$1
  local bs=$2
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
  local run_extra=$1
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
  # extra
  if [ $run_extra -eq 1 ]; then
    # bs = 1
    for net in ${net_list_generic_extra[@]}
    do
      run_generic $net 1
      if [ "$?" -ne 0 ]; then
        ERR=1
      fi
    done
    # bs = 4
    for net in ${net_list_batch_extra[@]}
    do
      run_generic $net 4
      if [ "$?" -ne 0 ]; then
        ERR=1
      fi
    done
  fi
  return $ERR
}

run_generic_all_parallel()
{
  local run_extra=$1
  rm -f regression.txt
  for net in ${net_list_generic[@]}
  do
    echo "run_generic $net 1" >> regression.txt
  done
  for net in ${net_list_batch[@]}
  do
    echo "run_generic $net 4" >> regression.txt
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    for net in ${net_list_generic_extra[@]}
    do
      echo "run_generic $net 1" >> regression.txt
    done
    for net in ${net_list_batch_extra[@]}
    do
      echo "run_generic $net 4" >> regression.txt
    done
  fi
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
  local count=$1
  local run_extra=$2
  ERR=0
  for net in ${net_list_accuracy[@]}
  do
    run_accuracy $net $count
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    for net in ${net_list_accuracy_extra[@]}
    do
      run_accuracy $net $count
      if [ "$?" -ne 0 ]; then
        ERR=1
      fi
    done
  fi
  return $ERR
}

run_accuracy_all_parallel()
{
  local count=$1
  local run_extra=$2
  rm -f accuracy.txt
  for net in ${net_list_accuracy[@]}
  do
    echo "run_accuracy $net $count" >> accuracy.txt
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    for net in ${net_list_accuracy_extra[@]}
    do
      echo "run_accuracy $net $count" >> accuracy.txt
    done
  fi
  if [ -f accuracy.txt ]; then
    cat accuracy.txt
    parallel -j4 --delay 0.5  --joblog job_accuracy.log < accuracy.txt
    return $?
  fi
}

run_onnx_ir_test()
{
  # IR test
  ERR=0
  onnx_ir_test.sh all_ir > onnx_all_ir\.log | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "onnx all ir test FAILED" >> verdict.log
    return 1
  else
    echo "onnx all ir test PASSED" >> verdict.log
  fi

  # Net test
  for net in ${net_list_onnx[@]}
  do
    onnx_ir_test.sh $net > onnx_$net\.log 2>&1 | true
    if [ "${PIPESTATUS[0]}" -ne "0" ]; then
      echo "$net onnx test FAILED" >> verdict.log
    ERR=1
    else
      echo "$net onnx test PASSED" >> verdict.log
  fi
  done

  return $ERR
}

usage()
{
   echo ""
   echo "Usage: $0 [-b batch_size] [-n net_name] [-e] [-a count]"
   echo -e "\t-b Description of batch size for test"
   echo -e "\t-n Description of net name for test"
   echo -e "\t-e Enable extra net list"
   echo -e "\t-a Enable run accuracy, with given image count"
   exit 1
}

run_extra=0
bs=1
run_accuracy=0
run_onnx_test=1
while getopts "n:b:a:e" opt
do
  case "$opt" in
    n ) net="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    e ) run_extra=1 ;;
    a ) run_accuracy="$OPTARG" ;;
    h ) usage ;;
  esac
done

# default run in parallel
if [ -z "$RUN_IN_PARALLEL" ]; then
  export RUN_IN_PARALLEL=1
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
echo "" > verdict.log
# run single and exit
if [ ! -z "$net" ]; then
  export CVIMODEL_REL_PATH=$PWD/cvimodel_regression
  if [ ! -e $CVIMODEL_REL_PATH ]; then
    mkdir $CVIMODEL_REL_PATH
  fi
  run_generic $net $bs
  ERR=$?
  if [ $ERR -eq 0 ]; then
    echo $net TEST PASSED
  else
    echo $net FAILED
  fi
  popd
  exit $ERR
fi

ERR=0
if [ $RUN_IN_PARALLEL -eq 0 ]; then
  run_generic_all $run_extra
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
  if [ $run_accuracy -ne 0 ]; then
    run_accuracy_all $run_accuracy $run_extra
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  fi
else
  run_generic_all_parallel $run_extra
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
  if [ $run_accuracy -ne 0 ]; then
    run_accuracy_all_parallel $run_accuracy $run_extra
    if [ "$?" -ne 0 ]; then
      ERR=1
    fi
  fi
fi

if [ $run_onnx_test -ne 0 ]; then
  run_onnx_ir_test
fi

cat verdict.log

popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

exit $ERR
