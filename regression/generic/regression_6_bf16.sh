#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

# quantization
if [ $DO_QUANT_BF16 -eq 1 ] && [ $DO_NOT_BF16_UNDER_182x -eq 0 ]; then
  mlir-opt \
      --assign-chip-name \
      --chipname ${SET_CHIP_NAME} \
      --tpu-quant --quant-full-bf16 \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_bf16.csv \
      ${NET}_opt.mlir \
      -o ${NET}_quant_bf16.mlir

  # bf16 inference
  mlir-tpu-interpreter ${NET}_quant_bf16.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_bf16.npz \
      --dump-all-tensor=${NET}_tensor_all_bf16.npz

  cvi_npz_tool.py compare \
      ${NET}_out_bf16.npz \
      ${NET}_out_fp32.npz \
      --tolerance $TOLERANCE_BF16 -vv

  cvi_npz_tool.py compare \
      ${NET}_tensor_all_bf16.npz \
      ${NET}_tensor_all_fp32.npz \
      --op_info ${NET}_op_info_bf16.csv \
      --tolerance $TOLERANCE_BF16 -vv

  if [ $DO_CMDBUF_BF16 -eq 1 ]; then
    ################################
    # Lower
    ################################
    mlir-opt \
        --tpu-lower --reorder-op --tg-op-tile \
        ${NET}_quant_bf16.mlir \
        -o ${NET}_quant_bf16_tg.mlir

    if [ $DO_LG_WITH_BF16 -eq 1 ]; then
        mlir-opt \
            --group-ops \
            ${NET}_quant_bf16_tg.mlir \
            -o ${NET}_quant_bf16_lg.mlir

        mlir-opt \
            --compress-weight \
            --tpu-compressed-weight-map-filename=${NET}_quant_bf16_lg_compressed_weight_stats.csv \
            ${NET}_quant_bf16_lg.mlir \
            -o ${NET}_quant_bf16_lg_compressed.mlir

        # assign weight address & neuron address
        mlir-opt \
            --assign-weight-address \
            --tpu-weight-address-align=16 \
            --tpu-weight-map-filename=${NET}_weight_map_bf16.csv \
            --tpu-weight-bin-filename=weight_bf16.bin \
            --assign-neuron-address \
            --tpu-neuron-address-align=64 \
            --tpu-neuron-map-filename=${NET}_neuron_map_bf16.csv \
            ${NET}_quant_bf16_lg_compressed.mlir \
            -o ${NET}_quant_bf16_addr.mlir
    else
        # assign weight address & neuron address
        mlir-opt \
            --assign-weight-address \
            --tpu-weight-address-align=16 \
            --tpu-weight-map-filename=${NET}_weight_map_bf16.csv \
            --tpu-weight-bin-filename=weight_bf16.bin \
            --assign-neuron-address \
            --tpu-neuron-address-align=64 \
            --tpu-neuron-map-filename=${NET}_neuron_map_bf16.csv \
            ${NET}_quant_bf16_tg.mlir \
            -o ${NET}_quant_bf16_addr.mlir
    fi

    # backend translate into cmdbuf
    mlir-opt \
        --divide-ops-to-func \
        ${NET}_quant_bf16_addr.mlir \
        -o ${NET}_quant_bf16_addr_func.mlir

    mlir-translate \
        --mlir-to-cvimodel \
        --weight-file weight_bf16.bin \
        ${NET}_quant_bf16_addr_func.mlir \
        -o ${NET}_bf16.cvimodel

    # run cvimodel
    model_runner \
        --dump-all-tensors \
        --input ${NET}_in_fp32.npz \
        --model ${NET}_bf16.cvimodel \
        --batch-num $BATCH_SIZE \
        --output ${NET}_cmdbuf_out_all_bf16.npz

    # compare all tensors
    cvi_npz_tool.py compare \
        ${NET}_cmdbuf_out_all_bf16.npz \
        ${NET}_tensor_all_bf16.npz \
        --op_info ${NET}_op_info_bf16.csv \
        --tolerance=$TOLERANCE_BF16_CMDBUF -vv
  fi
fi
# VERDICT
echo $0 PASSED
