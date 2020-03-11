#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ERR=0

echo -e "\033[1;32mAccuracy Regression -> $1\033[0m"
$DIR/../generic/accuracy_generic.sh $1 $2 > ${1}_accuracy.log 2>&1 | true
if [ "${PIPESTATUS[0]}" -ne "0" ]; then
  echo "$1 accuracy test FAILED" >> verdict.log
  ERR=1
else
  echo "$1 accuracy test PASSED" >> verdict.log
fi

exit $ERR
