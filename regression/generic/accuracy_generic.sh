#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  return 1
fi

export NET=$NET
source $DIR/generic_models.sh

pushd $NET

if [ $DO_ACCURACY_CAFFE -eq 1 ]; then
  $DIR/accuracy_0_caffe.sh $2
fi
if [ $DO_ACCURACY_INTERPRETER -eq 1 ]; then
  $DIR/accuracy_1_interpreter.sh $2 pytorch
  # $DIR/accuracy_1_interpreter.sh $2 gluoncv
fi
popd

echo $0 DONE
