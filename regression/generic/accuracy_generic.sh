#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  exit 1
fi

export NET=$NET
source $DIR/generic_models.sh

pushd $NET

if [ -z $EVAL_SCRIPT ]; then
  if [ $DO_ACCURACY_CAFFE -eq 1 ]; then
    $DIR/accuracy_0_caffe.sh $2
  fi
  if [ $DO_ACCURACY_ONNX -eq 1 ]; then
    $DIR/accuracy_0_onnx.sh $2
  fi
  if [ $DO_ACCURACY_INTERPRETER -eq 1 ]; then
    $DIR/accuracy_1_interpreter.sh $2 pytorch
    # $DIR/accuracy_1_interpreter.sh $2 gluoncv
  fi
else
  if [ ! -e $EVAL_SCRIPT ]; then
    echo "$EVAL_SCRIPT not exist"
    exit 1
  fi
  $EVAL_SCRIPT $2
fi

popd

echo $0 DONE
