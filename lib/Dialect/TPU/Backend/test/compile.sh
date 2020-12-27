#!/bin/bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MLIR_MODEL=$DIR/$1
BATCH_SIZE=`sed -r -n 's/^.+?tpu_func\(%arg0: tensor<([[:digit:]]+?)x.+?$/\1/p' $MLIR_MODEL`
OP_NAME=`echo $1 | sed -r -n 's/^(.+?)\.mlir$/\1/p'`
echo "to compile $MLIR_MODEL batch=$BATCH_SIZE"

export SET_CHIP_NAME="cv183x"
if [ ! -e "$DIR/tmp" ]; then
  mkdir -p $DIR/tmp
fi

#DEBUG=-debug
cp -f $DIR/simple_cali.py $DIR/tmp/

pushd $DIR/tmp

tpuc-opt ${MLIR_MODEL} \
    --convert-bn-to-scale \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${OP_NAME}_op_info.csv \
    -o ${OP_NAME}_opt_fp32.mlir

tpuc-opt \
     --gen-pseudo-weight-npz \
     ${OP_NAME}_opt_fp32.mlir \
     -o ${OP_NAME}_tmp.mlir
mv input.npz ${OP_NAME}_in_fp32.npz

tpuc-interpreter ${OP_NAME}_tmp.mlir \
    --tensor-in ${OP_NAME}_in_fp32.npz \
    --tensor-out ${OP_NAME}_out_fp32.npz \
    --dump-all-tensor=${OP_NAME}_tensor_all_fp32.npz

python simple_cali.py ${OP_NAME}_tensor_all_fp32.npz > ${OP_NAME}_calibration_table

# quantization.
tpuc-opt \
     --import-calibration-table \
     --calibration-table ${OP_NAME}_calibration_table \
     ${OP_NAME}_tmp.mlir \
     -o ${OP_NAME}_cali.mlir
tpuc-opt \
     --assign-chip-name \
     --chipname $SET_CHIP_NAME \
     --tpu-quant \
     --print-tpu-op-info \
     --tpu-op-info-filename ${OP_NAME}_op_info_int8.csv \
     ${OP_NAME}_cali.mlir \
     -o ${OP_NAME}_int8.mlir

# optimization for int8 mlir model
${REGRESSION_PATH}/mlir_to_cvimodel.sh \
   -i ${OP_NAME}_int8.mlir \
   -o ${OP_NAME}.cvimodel

# run cvimodel on emulator
model_runner \
     --dump-all-tensors \
     --input ${OP_NAME}_in_fp32.npz \
     --model ${OP_NAME}.cvimodel \
     --batch-num $BATCH_SIZE \
     --output ${OP_NAME}_cmdbuf_out_int8.npz

# inference with int8 model and get outputs.
tpuc-interpreter ${OP_NAME}_int8.mlir \
    --tensor-in ${OP_NAME}_in_fp32.npz \
    --tensor-out ${OP_NAME}_out_int8.npz \
    --dump-all-tensor=${OP_NAME}_tensor_all_int8.npz

# compare results
cvi_npz_tool.py compare \
    ${OP_NAME}_cmdbuf_out_int8.npz \
    ${OP_NAME}_tensor_all_int8.npz \
    --op_info ${OP_NAME}_op_info_int8.csv

popd
