#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ERR=0

if [ ! -e regression_out ]; then
  mkdir regression_out
fi
export CVIMODEL_REL_PATH=regression_out/cvimodel_release
if [ ! -e $CVIMODEL_REL_PATH ]; then
  mkdir $CVIMODEL_REL_PATH
fi

pushd regression_out

echo "accuracy $1"
$DIR/../generic/accuracy_generic.sh $1 100 2>&1 | tee ${1}_accuracy.log
if [ "${PIPESTATUS[0]}" -ne "0" ]; then
  echo "$1 accuracy test FAILED" >> verdict.log
  ERR=1
else
  echo "$1 accuracy test PASSED" >> verdict.log
fi

popd

exit $ERR
