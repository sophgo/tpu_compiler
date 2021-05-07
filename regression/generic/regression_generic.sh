#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z $SET_CHIP_NAME ]; then
  echo "please set SET_CHIP_NAME"
  exit 1
fi
export WORKING_PATH=${WORKING_PATH:-$DIR/regression_out}
export WORKSPACE_PATH=${WORKING_PATH}/${SET_CHIP_NAME}
export CVIMODEL_REL_PATH=$WORKSPACE_PATH/cvimodel_regression
export OMP_NUM_THREADS=4

echo "WORKING_PATH: ${WORKING_PATH}"
echo "WORKSPACE_PATH: ${WORKSPACE_PATH}"
echo "CVIMODEL_REL_PATH: ${CVIMODEL_REL_PATH}"

if [ -z "$2" ]; then
  DO_BATCHSIZE=1
else
  echo "$0 DO_BATCHSIZE=$DO_BATCHSIZE"
  DO_BATCHSIZE=$2
fi

export DO_BATCHSIZE=$DO_BATCHSIZE
export NET=$NET
source $DIR/generic_models.sh

WORKDIR=${WORKSPACE_PATH}/${NET}_bs${DO_BATCHSIZE}
mkdir -p $CVIMODEL_REL_PATH
mkdir -p $WORKDIR
pushd $WORKDIR

# run tests
if [ $INT8_MODEL -eq 1 ]; then
  /bin/bash $INT8_INFERENCE_SCRIPT
  $DIR/regression_9_int8_quantized_model.sh
else
  $DIR/regression_all_in_one.sh
fi
popd

if [ ! -z $CLEAN_WORKDIR ] && [ $CLEAN_WORKDIR -eq 1 ]; then
  echo "#### rm workspace $WORKDIR"
  rm -rf $WORKDIR
fi

unset DO_BATCHSIZE
unset NET

# VERDICT
echo $0 PASSED
