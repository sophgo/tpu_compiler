#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1

source $DIR/onnx_models.sh

# remove previous result
rm -rf onnx_test

if [ $NET = "all_ir" ]; then
    test_onnx.py
else
    test_onnx.py $INPUT_SHAPE $MODEL_DEF $INPUT_NAME
fi

# VERDICT
echo $0 PASSED
