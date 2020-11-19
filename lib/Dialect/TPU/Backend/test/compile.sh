#!/bin/bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MLIR_MODEL=$DIR/$1
BATCH_SIZE=`sed -r -n 's/^.+?tpu_func\(%arg0: tensor<([[:digit:]]+?)x.+?$/\1/p' $MLIR_MODEL`
echo "to compile $MLIR_MODEL batch=$BATCH_SIZE"

export SET_CHIP_NAME="cv183x"
if [ ! -e "$DIR/tmp" ]; then
  mkdir -p $DIR/tmp
fi

#DEBUG=-debug

pushd $DIR/tmp

mlir-opt \
     --gen-pseudo-weight-npz \
     --pseudo-calibration-table test_calibration_table \
     ${MLIR_MODEL} \
     -o tmp.mlir
mv input.npz test_in_fp32_bs${BATCH_SIZE}.npz

# opt

mlir-opt ${MLIR_MODEL} \
    --convert-bn-to-scale \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    -o test_opt_fp32.mlir

# quantization.
mlir-opt \
     --import-calibration-table \
     --calibration-table test_calibration_table \
     test_opt_fp32.mlir \
     -o test_cali.mlir
mlir-opt \
     --assign-chip-name \
     --chipname $SET_CHIP_NAME \
     --tpu-quant \
     --print-tpu-op-info \
     --tpu-op-info-filename test_op_info_int8.csv \
     test_cali.mlir \
     -o test_int8.mlir

# optimization for int8 mlir model
mlir-opt \
     --tpu-lower \
     --reorder-op \
     --tg-fuse-leakyrelu \
     --conv-ic-alignment \
     --group-ops \
     --dce \
     --deep-fusion-group-slice \
     --deep-fusion-opt \
     test_int8.mlir \
     -o test_int8_opt.mlir
mlir-opt \
     --assign-weight-address \
     --tpu-weight-address-align=16 \
     --tpu-weight-map-filename=weight_map_int8_lg.csv \
     --tpu-weight-bin-filename=weight.bin \
     --assign-neuron-address \
     --tpu-neuron-memory-reuse \
     --tpu-neuron-address-align=64 \
     --tpu-neuron-map-filename=neuron_map.csv \
     test_int8_opt.mlir \
     -o test_int8_addr.mlir
mlir-opt \
     --divide-ops-to-func \
     test_int8_addr.mlir \
     -o test_int8_func.mlir

# codegen
mlir-translate ${DEBUG} \
     --mlir-to-cvimodel \
     --weight-file weight.bin \
     test_int8_func.mlir \
     -o test_bs${BATCH_SIZE}.cvimodel

# run cvimodel on emulator
model_runner \
     --dump-all-tensors \
     --input test_in_fp32_bs${BATCH_SIZE}.npz \
     --model test_bs${BATCH_SIZE}.cvimodel \
     --batch-num $BATCH_SIZE \
     --output test_cmdbuf_out_bs${BATCH_SIZE}.npz

# inference with int8 model and get outputs.
mlir-tpu-interpreter test_int8.mlir \
    --tensor-in test_in_fp32_bs${BATCH_SIZE}.npz \
    --tensor-out test_out_int8.npz \
    --dump-all-tensor=test_tensor_all_int8.npz

# compare results
cvi_npz_tool.py compare \
    test_cmdbuf_out_bs${BATCH_SIZE}.npz \
    test_tensor_all_int8.npz \
    --op_info test_op_info_int8.csv

popd
