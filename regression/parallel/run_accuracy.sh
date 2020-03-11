#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ERR=0

pushd $DIR/../regression_out

echo -e "\033[1;32mAccuracy Regression -> $1\033[0m"
$DIR/../generic/accuracy_generic.sh $1 100 &> ${1}_accuracy.log
if [ "${PIPESTATUS[0]}" -ne "0" ]; then
  echo "$1 accuracy test FAILED" >> verdict.log
  ERR=1
else
  echo "$1 accuracy test PASSED" >> verdict.log
fi

popd

exit $ERR
