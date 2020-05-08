#!/bin/bash
# set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
_REGRESSION_PATH=regression_out
GLOG_minloglevel=0

if [ ! -e ${_REGRESSION_PATH} ]; then
  mkdir ${_REGRESSION_PATH}
fi
export CVIMODEL_REL_PATH=$PWD/${_REGRESSION_PATH}/cvimodel_regression
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

echo -e "\033[1;32mBegin to test regression:\033[0m"
cat $DIR/regression.txt
echo ""

ERR=0

pushd ${_REGRESSION_PATH}

start_time="$(date -u +%s)"

#parallel -j$(nproc) --delay 2.5  --joblog job_regression.log < $DIR/regression.txt
#if [ "$?" -ne "0" ]; then
#  ERR=1
#fi
#parallel -j$(nproc) --delay 2.5 --joblog job_accuracy.log < $DIR/accuracy.txt
#if [ "$?" -ne "0" ]; then
#  ERR=1
#fi

echo -e "\033[1;32mDone! $?\n\033[0m"

cat job_regression.log
cat job_accuracy.log
echo ""
cat verdict.log

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo -e "Total of \033[1;32m$elapsed\033[0m seconds elapsed for process"

popd

echo "Parsing log"
pushd $DIR/tool
python ./cvi_accuracy.py --acc_log_path ${DIR}/../../${_REGRESSION_PATH} --input_acc_file=$DIR/accuracy.txt
echo -e "\033[1;32mexport $DIR/tool/accuracy.xls success\033[0m"
popd

# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

export GLOG_minloglevel=2

exit $ERR
