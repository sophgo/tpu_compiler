#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

NET=mobilenet_v2
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  return 1
fi

pushd $NET

$DIR/accuracy_mobilenet_v2_0_caffe.sh $1
$DIR/accuracy_mobilenet_v2_1_interpreter.sh $1 pytorch
# $DIR/accuracy_mobilenet_v2_1_interpreter.sh $1 gluoncv

popd

echo $0 DONE
