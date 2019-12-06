#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e regression_out ]; then
  echo "regression_out dir not exist, please run regression first"
  exit 1
fi

pushd regression_out

# run tests
$DIR/resnet50/accuracy_resnet50.sh $1 pytorch
$DIR/resnet50/accuracy_resnet50.sh $1 gluoncv

$DIR/mobilenet_v1/accuracy_mobilenet_v1.sh $1 pytorch
$DIR/mobilenet_v1/accuracy_mobilenet_v1.sh $1 gluoncv

$DIR/mobilenet_v2/accuracy_mobilenet_v2.sh $1 pytorch
$DIR/mobilenet_v2/accuracy_mobilenet_v2.sh $1 gluoncv

popd

echo $0 DONE
