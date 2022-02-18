#!/bin/bash
set -xe

export SET_CHIP_NAME="cv182x"
mkdir -p tmp

pushd tmp

tpuc-opt ../sqdiff.mlir \
    --convert-bn-to-scale \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename sqdiff_op_info.csv \
    -o sqdiff_fp32.mlir

model_deploy.py --model_name sqdiff \
    --mlir sqdiff_fp32.mlir \
    --chip cv182x \
    --quantize bf16 \
    --inputs_type SAME \
    --outputs_type SAME \
    --tolerance 0.8,0.8,0.67 \
    --correctness 0.9,0.9,0.9 \
    --debug \
    --cvimodel sqdiff.cvimodel
popd
