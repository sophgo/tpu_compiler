#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $DIR/../regression_out ]; then
  mkdir -p $DIR/../regression_out
else
  rm -rf $DIR/../regression_out/
fi
export CVIMODEL_REL_PATH=$DIR/../regression_out/cvimodel_release
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir -p $CVIMODEL_REL_PATH
else
  rm -rf $CVIMODEL_REL_PATH/
fi

echo -e "\033[1;32mBegin to test regression:\033[0m"
cat $DIR/regression.txt
echo ""

parallel -j13 --delay 2.5 --ungroup --joblog $DIR/../regression_out/job.log < $DIR/regression.txt

echo -e "\033[1;32mDone! $?\n\033[0m"
cat $DIR/../regression_out/job.log
echo ""
cat $DIR/../regression_out/verdict.log
