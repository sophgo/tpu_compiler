#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mlir_list=(
  relu.mlir
  permute.mlir
  permute_large.mlir
  crop.mlir
  concat.mlir
  conv.mlir
  reverse.mlir
  lrn.mlir
  fc.mlir
  fc_large.mlir
  eltwise_add.mlir
  square.mlir
  deconv.mlir
  avg_pool.mlir
  max_pool.mlir
  broadcast_add.mlir
  broadcast_sub.mlir
  quadratic_sum.mlir
  yuv420.mlir
#  conv3d.mlir
#  matmul_transpose.mlir
#  matmul_with_big_k.mlir
#  matmul.mlir
#  max_pool3d.mlir
#  search_model.mlir
)

sample_mlir=reverse.mlir

if [ x$1 = x1 ]; then
     for mlir in ${mlir_list[@]}
     do
     echo "========== Test $mlir =========="
     $SCRIPT_DIR/compile.sh $mlir
     if [ "$?" -ne 0 ]; then
          echo "### compile.sh $mlir failed"
          exit 1
     fi
     $SCRIPT_DIR/compile_bf16.sh $mlir
     if [ "$?" -ne 0 ]; then
          echo "### compile_bf16.sh $mlir failed"
          exit 1
     fi
     done
else
     echo "========== Test ${sample_mlir} =========="
     $SCRIPT_DIR/compile.sh ${sample_mlir}
     if [ "$?" -ne 0 ]; then
          echo "### compile.sh ${sample_mlir} failed"
          exit 1
     fi
fi

echo "test_backend.sh success"
exit 0