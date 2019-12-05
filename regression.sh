# !/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_1_fp32.sh
$DIR/regression_2_int8.sh
$DIR/regression_3_int8_cmdbuf.sh
$DIR/regression_4_bf16.sh
$DIR/regression_5_bf16_cmdbuf.sh

$DIR/regression_mobilenet_v2_1_fp32.sh
$DIR/regression_mobilenet_v2_2_int8.sh
$DIR/regression_mobilenet_v2_3_int8_cmdbuf.sh
$DIR/regression_mobilenet_v2_4_bf16.sh
$DIR/regression_mobilenet_v2_5_bf16_cmdbuf.sh

$DIR/regression_mobilenet_v1_1_fp32.sh

# VERDICT
echo $0 PASSED
