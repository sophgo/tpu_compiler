#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


NET=retinaface_res50
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  echo "$NET does not exist, run regression first"
  return 1
fi

pushd $NET

# Remove previous result
rm result_caffe_fp32 interpreter_result_fp32 interpreter_result_int8 -rf

$DIR/accuracy_retinaface_res50_0_caffe.sh
$DIR/accuracy_retinaface_res50_1_fp32.sh
$DIR/accuracy_retinaface_res50_2_int8.sh

popd

echo $0 DONE
