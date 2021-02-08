#!/bin/bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MLIR_MODEL=$DIR/$1
BATCH_SIZE=`sed -r -n 's/^.+?tpu_func\(%arg0: tensor<([[:digit:]]+?)x.+?$/\1/p' $MLIR_MODEL`
OP_NAME=`echo $1 | sed -r -n 's/^(.+?)\.mlir$/\1/p'`_bf16
echo "to compile $MLIR_MODEL batch=$BATCH_SIZE"

export SET_CHIP_NAME="cv183x"
if [ ! -e "$DIR/tmp" ]; then
  mkdir -p $DIR/tmp
fi

#DEBUG=-debug

pushd $DIR/tmp

tpuc-opt \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename ${OP_NAME}_op_info_fp32.csv \
    ${MLIR_MODEL} \
    -o ${OP_NAME}_fp32_opt.mlir

tpuc-opt \
     --gen-pseudo-weight-npz \
     --pseudo-calibration-table ${OP_NAME}_calibration_table \
     ${OP_NAME}_fp32_opt.mlir \
     -o ${OP_NAME}_tmp.mlir
mv input.npz ${OP_NAME}_in_fp32.npz

# fp32 result
tpuc-interpreter ${OP_NAME}_tmp.mlir \
    --tensor-in ${OP_NAME}_in_fp32.npz \
    --tensor-out ${OP_NAME}_out_fp32.npz \
    --dump-all-tensor=${OP_NAME}_tensor_all_fp32.npz

# quantization.
tpuc-opt \
     --assign-chip-name \
     --chipname $SET_CHIP_NAME \
     --tpu-quant --quant-full-bf16 \
     --quant-bf16-softmax \
     --print-tpu-op-info \
     --tpu-op-info-filename ${OP_NAME}_op_info_bf16.csv \
     ${OP_NAME}_tmp.mlir \
     -o ${OP_NAME}.mlir

${REGRESSION_PATH}/mlir_to_cvimodel.sh \
   -i ${OP_NAME}.mlir \
   -o ${OP_NAME}.cvimodel

# run cvimodel on emulator
model_runner \
     --dump-all-tensors \
     --input ${OP_NAME}_in_fp32.npz \
     --model ${OP_NAME}.cvimodel \
     --batch-num $BATCH_SIZE \
     --output ${OP_NAME}_cmdbuf_out_bf16.npz

# inference with bf16 model and get outputs.
tpuc-interpreter ${OP_NAME}.mlir \
    --tensor-in ${OP_NAME}_in_fp32.npz \
    --tensor-out ${OP_NAME}_out.npz \
    --dump-all-tensor=${OP_NAME}_tensor_all_bf16.npz

# compare bf16 and fp32 interpreter
cvi_npz_tool.py compare \
    ${OP_NAME}_tensor_all_bf16.npz \
    ${OP_NAME}_tensor_all_fp32.npz \
    --op_info ${OP_NAME}_op_info_bf16.csv \
    --tolerance 0.8,0.8,0.8 -vv

# compare cmdbuf and interpreter by bf16
cvi_npz_tool.py compare \
    ${OP_NAME}_cmdbuf_out_bf16.npz \
    ${OP_NAME}_tensor_all_bf16.npz \
    --op_info ${OP_NAME}_op_info_bf16.csv \
    --tolerance 0.99,0.99,0.99 -vv

popd
