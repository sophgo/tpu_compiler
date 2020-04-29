#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export GLOG_minloglevel=0

# please keep in alphabetical order
net_list=(
  "bmface_v3"
  "liveness"
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

if [ ! -z $net ]; then
  echo "cvitek zoo regression $net batch=$bs"
  $DIR/cvitek_zoo/cvitek_zoo_regression.sh $net $bs 2>&1 | tee $net\_bs$bs.log
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net batch=$bs generic regression FAILED" >> verdict.log
    ERR=1
  else
    echo "$net batch=$bs generic regression PASSED" >> verdict.log
  fi
else
  for net in ${net_list[@]}
  do
    echo "cvitek zoo regression $net batch=$bs"
    $DIR/cvitek_zoo/cvitek_zoo_regression.sh $net $bs 2>&1 | tee $net\_bs$bs.log
    if [ "${PIPESTATUS[0]}" -ne "0" ]; then
      echo "$net batch=$bs generic regression FAILED" >> verdict.log
      ERR=1
    else
      echo "$net batch=$bs generic regression PASSED" >> verdict.log
    fi
  done
fi
popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

exit $ERR
