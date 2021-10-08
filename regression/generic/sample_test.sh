#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ -z "$MLIR_PATH" ]]; then
  export MLIR_PATH=$INSTALL_PATH
fi
# add samples here
${PROJECT_ROOT}/examples/custom_op/build.sh

# VERDICT
echo $0 PASSED
