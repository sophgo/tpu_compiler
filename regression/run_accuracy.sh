#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ -z "$DATASET_PATH" ]]; then
  echo "DATASET_PATH not defined"
  return 1
fi
if [ ! -e $DATASET_PATH ]; then
  echo "DATASET_PATH $DATASET_PATH does not exist"
  return 1
fi

if [ ! -e regression_out ]; then
  echo "regression_out dir not exist, please run regression first"
  return 1
fi
pushd regression_out

# run tests
# $DIR/resnet50/accuracy_resnet50.sh $1 
# $DIR/yolo_v3/accuracy_yolo_v3.sh $1 
$DIR/ssd300/accuracy_ssd300.sh $1 $2
popd

echo $0 DONE
