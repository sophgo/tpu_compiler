#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh
pushd $DIR


# clear previous output
#rm -f *.mlir *.bin *.npz *.csv *.cvimodel *calibration_table

## run test
bash ./testbench_0_fp32.sh
bash ./testbench_1_int8.sh
bash ./testbench_1_int8_cmd.sh "fake_weight"

# VERDICT
echo $0 PASSED

popd
