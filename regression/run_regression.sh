#!/bin/bash
# set -e
# set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

run_generic()
{
  local net=$1
  local bs=$2
  local preprocess=$3
  echo "generic regression $net batch=$bs preprocess=$preprocess"
  regression_generic.sh $net $bs $preprocess > $1\_bs$bs.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net batch=$bs preprocess=$preprocess generic regression FAILED" >> verdict.log
    return 1
  else
    echo "$net batch=$bs preprocess=$preprocess generic regression PASSED" >> verdict.log
    return 0
  fi
}
export -f run_generic

run_generic_all()
{
  local run_extra=$1
  local err=0
  # bs = 1
  for net in ${net_list_generic[@]}
  do
    run_generic $net 1 0
    if [ "$?" -ne 0 ]; then
      err=1
    fi
  done
  # bs = 4
  for net in ${net_list_batch[@]}
  do
    run_generic $net 4 0
    if [ "$?" -ne 0 ]; then
      err=1
    fi
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    # bs = 1
    for net in ${net_list_generic_extra[@]}
    do
      run_generic $net 1 0
      if [ "$?" -ne 0 ]; then
        err=1
      fi
    done
    # bs = 4
    for net in ${net_list_batch_extra[@]}
    do
      run_generic $net 4 0
      if [ "$?" -ne 0 ]; then
        err=1
      fi
    done
    # fused preprocess
    for net in ${net_list_do_preprocess[@]}
    do
      run_generic $net 1 1
      if [ "$?" -ne 0 ]; then
        err=1
      fi
    done
  fi
  return $err
}

run_generic_all_parallel()
{
  local run_extra=$1

  rm -f regression.txt
  for net in ${net_list_generic[@]}
  do
    echo "run_generic $net 1 0" >> regression.txt
  done
  for net in ${net_list_batch[@]}
  do
    echo "run_generic $net 4 0" >> regression.txt
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    for net in ${net_list_generic_extra[@]}
    do
      echo "run_generic $net 1 0" >> regression.txt
    done
    for net in ${net_list_batch_extra[@]}
    do
      echo "run_generic $net 4 0" >> regression.txt
    done
    # fused preprocess
    for net in ${net_list_do_preprocess[@]}
    do
      echo "run_generic $net 1 1" >> regression.txt
    done
  fi
  cat regression.txt
  parallel -j8 --delay 0.5  --joblog job_regression.log < regression.txt
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
  local err=0
  for net in ${net_list_accuracy[@]}
  do
    run_accuracy $net $count
    if [ "$?" -ne 0 ]; then
      err=1
    fi
  done
  # extra
  if [ $run_extra -eq 1 ]; then
    for net in ${net_list_accuracy_extra[@]}
    do
      run_accuracy $net $count
      if [ "$?" -ne 0 ]; then
        err=1
      fi
    done
  fi
  return $err
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
  local err=0
  onnx_ir_test.sh all_ir > onnx_all_ir\.log | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "onnx all ir test FAILED" >> verdict.log
    return 1
  else
    echo "onnx all ir test PASSED" >> verdict.log
  fi

  return $err
}

usage()
{
   echo ""
   echo "Usage: $0 [-b batch_size] [-n net_name] [-e] [-a count]"
   echo -e "\t-b Description of batch size for test"
   echo -e "\t-n Description of net name for test"
   echo -e "\t-e Enable extra net list"
   echo -e "\t-a Enable run accuracy, with given image count"
   echo -e "\t-f Model list filename"
   exit 1
}

run_extra=0
bs=1
run_accuracy=0
run_onnx_test=1

while getopts "n:b:a:f:e" opt
do
  case "$opt" in
    n ) network="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    e ) run_extra=1 ;;
    a ) run_accuracy="$OPTARG" ;;
    f ) model_list_file="$OPTARG" ;;
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

net_list_generic=()
net_list_batch=()
net_list_accuracy=()
net_list_generic_extra=()
net_list_batch_extra=()
net_list_accuracy_extra=()
net_list_do_preprocess=()

if [ -z $model_list_file ]; then
  model_list_file=$SCRIPT_DIR/generic/model_list.txt
fi

while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext fused_preprocess
do
  [[ $net =~ ^#.* ]] && continue
  # echo "net='$net' bs1='$bs1' bs4='$bs4' acc='$acc' bs1_ext='$bs1_ext' bs4_ext='$bs4_ext' acc_ext='$acc_ext' fused_preprocess='$fused_preprocess'"
  if [ "$bs1" = "Y" ]; then
    # echo "bs1 add $net"
    net_list_generic+=($net)
  fi
  if [ "$bs4" = "Y" ]; then
    # echo "bs4 add $net"
    net_list_batch+=($net)
  fi
  if [ "$acc" = "Y" ]; then
    # echo "acc add $net"
    net_list_accuracy+=($net)
  fi
  if [ "$bs1_ext" = "Y" ]; then
    # echo "bs1_ext add $net"
    net_list_generic_extra+=($net)
  fi
  if [ "$bs4_ext" = "Y" ]; then
    # echo "bs4_ext add $net"
    net_list_batch_extra+=($net)
  fi
  if [ "$acc_ext" = "Y" ]; then
    # echo "acc_ext add $net"
    net_list_accuracy_extra+=($net)
  fi
  if [ "$fused_preprocess" = "Y" ]; then
    # echo "fused_preprocess add $net"
    net_list_do_preprocess+=($net)
  fi
done < ${model_list_file}

# printf '%s\n' "${net_list_generic[@]}"
# printf '%s\n' "${net_list_batch[@]}"
# printf '%s\n' "${net_list_accuracy[@]}"
# printf '%s\n' "${net_list_generic_extra[@]}"
# printf '%s\n' "${net_list_batch_extra[@]}"
# printf '%s\n' "${net_list_accuracy_extra[@]}"
# printf '%s\n' "${net_list_do_preprocess[@]}"

pushd regression_out
echo "" > verdict.log
# run specified network and exit
if [ ! -z "$network" ]; then
  export CVIMODEL_REL_PATH=$PWD/cvimodel_regression
  if [ ! -e $CVIMODEL_REL_PATH ]; then
    mkdir $CVIMODEL_REL_PATH
  fi
  run_generic $network $bs 0
  ERR=$?
  if [ $ERR -eq 0 ]; then
    echo $network TEST PASSED
  else
    echo $network FAILED
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
  if [ "$?" -ne 0 ]; then
    ERR=1
  fi
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
