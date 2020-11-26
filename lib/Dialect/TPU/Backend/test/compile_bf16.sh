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
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv \
    ${MLIR_MODEL} \
    -o test_fp32_opt.mlir

mlir-opt \
     --gen-pseudo-weight-npz \
     --pseudo-calibration-table test_calibration_table \
     test_fp32_opt.mlir \
     -o tmp.mlir
mv input.npz test_in_fp32_bs${BATCH_SIZE}.npz

# quantization.
mlir-opt \
     --assign-chip-name \
     --chipname $SET_CHIP_NAME \
     --tpu-quant --quant-full-bf16 \
     --quant-bf16-softmax \
     --print-tpu-op-info \
     --tpu-op-info-filename test_op_info_bf16.csv \
     test_fp32_opt.mlir \
     -o test_bf16.mlir

# optimization for bf16 mlir model
mlir-opt \
     --tpu-lower \
     --tg-fuse-leakyrelu \
     --conv-ic-alignment \
     --group-ops \
     --dce \
     --deep-fusion-group-slice \
     --deep-fusion-opt \
     test_bf16.mlir \
     -o test_bf16_opt.mlir
mlir-opt \
     --assign-weight-address \
     --tpu-weight-address-align=16 \
     --tpu-weight-map-filename=weight_map_bf16_lg.csv \
     --tpu-weight-bin-filename=weight.bin \
     --assign-neuron-address \
     --tpu-neuron-memory-reuse \
     --tpu-neuron-address-align=64 \
     --tpu-neuron-map-filename=neuron_map.csv \
     test_bf16_opt.mlir \
     -o test_bf16_addr.mlir
mlir-opt \
     --divide-ops-to-func \
     test_bf16_addr.mlir \
     -o test_bf16_func.mlir

# codegen
mlir-translate ${DEBUG} \
     --mlir-to-cvimodel \
     --weight-file weight.bin \
     test_bf16_func.mlir \
     -o test_bs${BATCH_SIZE}.cvimodel

# run cvimodel on emulator
model_runner \
     --dump-all-tensors \
     --input test_in_fp32_bs${BATCH_SIZE}.npz \
     --model test_bs${BATCH_SIZE}.cvimodel \
     --batch-num $BATCH_SIZE \
     --output test_cmdbuf_out_bs${BATCH_SIZE}.npz

# inference with bf16 model and get outputs.
mlir-tpu-interpreter test_bf16.mlir \
    --tensor-in test_in_fp32_bs${BATCH_SIZE}.npz \
    --tensor-out test_out_bf16.npz \
    --dump-all-tensor=test_tensor_all_bf16.npz

# compare results
cvi_npz_tool.py compare \
    test_cmdbuf_out_bs${BATCH_SIZE}.npz \
    test_tensor_all_bf16.npz \
    --op_info test_op_info_bf16.csv \
    --tolerance 0.99,0.99,0.99 -vv

popd
