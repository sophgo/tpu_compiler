#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export NET=$NET
source $DIR/generic_models.sh

WORKDIR=${NET}_bs1
if [ ! -e $WORKDIR ]; then
  echo "$WORKDIR does not exist, run regression first"
  exit 1
fi
pushd $WORKDIR

if [ -z $EVAL_SCRIPT ]; then
  if [ $MODEL_TYPE = "caffe" ]; then
    $DIR/accuracy_0_caffe.sh $2
  elif [ $MODEL_TYPE = "onnx" ]; then
    $DIR/accuracy_0_onnx.sh $2
  elif [ $MODEL_TYPE = "tensorflow" ]; then
    $DIR/accuracy_0_tensorflow.sh $2
  else
    echo "Invalid MODEL_TYPE $MODEL_TYPE"
    exit 1
  fi
  $DIR/accuracy_1_interpreter.sh $2 pytorch
  # $DIR/accuracy_1_interpreter.sh $2 gluoncv
else
  if [ ! -e $EVAL_SCRIPT ]; then
    echo "$EVAL_SCRIPT not exist"
    exit 1
  fi
  $EVAL_SCRIPT $2
fi

popd

echo $0 DONE
