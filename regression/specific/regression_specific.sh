#!/bin/bash
set -e

#test yuv420_csc
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export WORKING_PATH=${WORKING_PATH:-$DIR/regression_out}

WORKDIR=${WORKING_PATH}/specific
mkdir -p $WORKDIR
pushd $WORKDIR
  $DIR/yuv420_test.sh
popd

# VERDICT
echo $0 PASSED
