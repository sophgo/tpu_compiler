#!/bin/bash
# set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GLOG_minloglevel=0

if [ ! -e regression_out ]; then
  mkdir regression_out
fi
export CVIMODEL_REL_PATH=$PWD/regression_out/cvimodel_regression
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

echo -e "\033[1;32mBegin to test regression:\033[0m"
cat $DIR/regression.txt
echo ""

ERR=0

pushd regression_out

parallel -j0 --delay 2.5  --joblog job_regression.log < $DIR/regression.txt
if [ "$?" -ne "0" ]; then
  ERR=1
fi
parallel -j0 --delay 2.5 --joblog job_accuracy.log < $DIR/accuracy.txt
if [ "$?" -ne "0" ]; then
  ERR=1
fi

echo -e "\033[1;32mDone! $?\n\033[0m"
cat job_regression.log
cat job_accuracy.log
echo ""
cat verdict.log

popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

export GLOG_minloglevel=2

exit $ERR
