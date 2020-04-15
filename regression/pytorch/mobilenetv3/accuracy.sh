#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
COUNT=$1


NET=mobilenetv3_pytorch
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  return 1
fi

pushd $NET

$DIR/accuracy_1_interpreter.sh $COUNT pytorch $NET

popd

echo $0 DONE
