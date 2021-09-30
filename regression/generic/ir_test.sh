#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# remove previous result
rm -rf onnx_test

test_onnx.py

#rm -rf torch_test

#test_torch.py

# VERDICT
echo $0 PASSED
