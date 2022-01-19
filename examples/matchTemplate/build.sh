#!/bin/bash
set -xe

export SET_CHIP_NAME="cv182x"
mkdir -p tmp

pushd tmp

tpuc-opt ../ccoeff_normed.mlir \
    --convert-bn-to-scale \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ccoeff_normed_op_info.csv \
    -o ccoeff_normed_fp32.mlir

model_deploy.py --model_name ccoeff_normed \
    --mlir ccoeff_normed_fp32.mlir \
    --calibration_table ../ccoeff_normed_cali_table \
    --mix_precision_table ../ccoeff_normed_mix_table \
    --chip cv182x \
    --inputs_type KEEP \
    --outputs_type KEEP \
    --tolerance 0.8,0.8,0.67 \
    --correctness 0.9,0.9,0.9 \
    --debug \
    --cvimodel ccoeff_normed.cvimodel

popd
