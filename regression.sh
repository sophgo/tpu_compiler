# !/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# clear previous output
rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_1_fp32.sh
$DIR/regression_2_int8.sh
$DIR/regression_3_cmdbuf.sh
$DIR/regression_4_bf16.sh

# VERDICT
echo $0 PASSED
