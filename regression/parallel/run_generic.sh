#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ERR=0

if [ -z $2 ]; then
  bs=1
else
  bs=$2
fi

# generic
echo -e "\033[1;32mGeneric Regression -> $1 bs $bs\033[0m"
$DIR/../generic/regression_generic.sh $1 $bs> $1\_bs$bs.log 2>&1 | true
if [ "${PIPESTATUS[0]}" -ne "0" ]; then
  echo "$1 bs $bs generic regression FAILED" >> verdict.log
  ERR=1
else
  echo "$1 bs $bs generic regression PASSED" >> verdict.log
fi

exit $ERR
