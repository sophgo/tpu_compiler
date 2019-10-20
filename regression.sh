# !/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/regression_1_fp32.sh
$DIR/regression_2_int8.sh
$DIR/regression_3_cmdbuf.sh

# VERDICT
echo $0 PASSED
