#!/bin/bash
# set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export GLOG_minloglevel=0

if [ ! -e regression_out ]; then
  mkdir regression_out
fi
export CVIMODEL_REL_PATH=regression_out/cvimodel_release
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

echo -e "\033[1;32mBegin to test regression:\033[0m"
cat $DIR/regression.txt
echo ""

ERR=0

pushd regression_out

#set BATCH_SIZE="1 4 8 16 32" for batch test
BATCH_SIZE="1 2 4"
for d in ${BATCH_SIZE}
do
  cp $DIR/regression.txt $DIR/regression_b.txt
  sed -e "s/$/ $d/" -i $DIR/regression_b.txt
  #delete previos regression folder under regression_out
  find $PWD -type d -name "*" ! -path "$PWD" |xargs rm -rf

  parallel -j13 --delay 2.5 --ungroup --joblog job_regression_b$d.log < $DIR/regression_b.txt
  if [ "$?" -ne "0" ]; then
    ERR=1
  fi
done
parallel -j13 --delay 2.5 --ungroup --joblog job_accuracy.log < $DIR/accuracy.txt
if [ "$?" -ne "0" ]; then
  ERR=1
fi

echo -e "\033[1;32mDone! $?\n\033[0m"
for d in ${BATCH_SIZE}
do
  cat job_regression_b$d.log
done
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

exit $ERR
