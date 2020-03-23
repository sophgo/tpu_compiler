#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


NET=efficientnet_b0
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  return 1
fi

pushd $NET

$DIR/accuracy_efficientnet_b0_0_caffe.sh $1
$DIR/accuracy_efficientnet_b0_1_interpreter.sh $1 pytorch
# $DIR/accuracy_efficientnet_b0_1_interpreter.sh $1 gluoncv

popd

echo $0 DONE
