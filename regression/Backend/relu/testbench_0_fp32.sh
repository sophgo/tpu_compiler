#!/bin/bash
set -e

# use python to mlir , gen golden too
python ./make_mlir.py

# show input
npz_dump.py test_in_fp32.npz arr_0 0

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
npz_dump.py test_out_fp32.npz Y 0
