#!/bin/bash

# gen mlir
python test_conv.py

# inference
mlir-tpu-interpreter test_conv.mlir \
    --tensor-in test_in_fp32.npz \
    --tensor-out test_out_fp32.npz \
    --dump-all-tensor=test_tensor_all_fp32.npz

# Compare with golden
cvi_npz_tool.py compare \
    test_out_fp32.npz \
    test_output_golden.npz -vv

echo $0 PASSED