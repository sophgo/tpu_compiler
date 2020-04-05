#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export GLOG_minloglevel=0

if [ ! -e regression_out ]; then
  mkdir regression_out
fi
CVIMODEL_REL_ROOT_PATH=$PWD/regression_out/cvimodel_regression
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

echo -e "\033[1;32mBegin to test regression:\033[0m"
cat $DIR/regression.txt
echo ""

ERR=0

pushd regression_out

# set BATCH_SIZE="1 4 8 16 32" for batch test
# execute script and set batch size such as "paralle_wk_test.sh 1 2 4"
# we can easily to set batch size for CI without changing this script
for i in $@
do
 BATCH_SIZE+=" $i"
done
if [[ -z "$BATCH_SIZE" ]]; then
 BATCH_SIZE="1"
fi

for d in ${BATCH_SIZE}
do
  export CVIMODEL_REL_PATH=$CVIMODEL_REL_ROOT_PATH\_bs$d
  if [ ! -e $CVIMODEL_REL_PATH ]; then
    mkdir $CVIMODEL_REL_PATH
  fi
  cp $DIR/regression.txt $DIR/regression_b.txt
  sed -e "s/$/ $d/" -i $DIR/regression_b.txt
  #delete previos regression folder under regression_out and keep CVIMODEL_REL_PATH folder
  find $PWD -type d -name "*" ! -path "$PWD" ! -path "$CVIMODEL_REL_ROOT_PATH*" |xargs rm -rf

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
