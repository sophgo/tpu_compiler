#!/bin/bash
set -e

# use python to mlir , gen golden too
# default is <1, 4, 3, 4>, factor is 2
python ./make_mlir.py \
  --output_name test_in_fp32.npz \
  --node_name Transpose \
  -n 1 -c 2048 --height 8 -w 6 \
  --factor 2

# show input
cvi_npz_tool.py dump test_in_fp32.npz arr_0 0

mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename test_op_info.csv \
    --convert-bn-to-scale \
    test.mlir \
    -o test_opt.mlir

# test frontend optimizations
#gdb --args \
mlir-tpu-interpreter test_opt.mlir \
    --tensor-in test_in_fp32.npz \
    --tensor-out test_out_fp32.npz \
    --dump-all-tensor=test_tensor_all_fp32.npz

# show output
cvi_npz_tool.py dump test_out_fp32.npz Y_Transpose 0
