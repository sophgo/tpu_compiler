#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


test_onnx.py $1 $2 $3


# VERDICT
echo $0 PASSED
