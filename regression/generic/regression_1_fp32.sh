#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


CHECK_NON_OPT_VERSION=0

if [ $MODEL_TYPE = "caffe" ]; then
  # translate from caffe model
  mlir-translate \
      --caffe-to-mlir $MODEL_DEF \
      --caffemodel $MODEL_DAT \
      --static-batchsize $BATCH_SIZE \
      -o ${NET}.mlir
elif [[ $MODEL_TYPE = "onnx" || $MODEL_TYPE = "tflite" || $MODEL_TYPE = "tensorflow" ]]; then
  cvi_model_convert.py \
      --model_path $MODEL_DEF \
      --model_name ${NET} \
      --model_type $MODEL_TYPE \
      --batch_size $BATCH_SIZE \
      --mlir_file_path ${NET}.mlir
else
  echo "Invalid MODEL_TYPE=$MODEL_TYPE"
  return 1
fi


# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --assign-layer-id \
    ${MLIR_OPT_FE_PRE} \
    --canonicalize \
    ${MLIR_OPT_FE_POST} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter ${NET}_opt.mlir \
    -debug \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

#cvi_npz_tool.py compare ${NET}_out_fp32.npz ${NET}_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info.csv \
    --excepts $EXCEPTS \
    --tolerance=0.999,0.999,0.998 -vv

cvi_npz_tool.py to_bin ${NET}_in_fp32.npz $INPUT ${NET}_in_fp32.bin

# VERDICT
echo $0 PASSED
