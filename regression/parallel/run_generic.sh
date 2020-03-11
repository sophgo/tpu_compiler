#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ERR=0

# generic
echo -e "\033[1;32mGeneric Regression -> $1\033[0m"
$DIR/../generic/regression_generic.sh $1 > $1.log 2>&1 | true
if [ "${PIPESTATUS[0]}" -ne "0" ]; then
  echo "$1 generic regression FAILED" >> verdict.log
  ERR=1
else
  echo "$1 generic regression PASSED" >> verdict.log
fi

exit $ERR
