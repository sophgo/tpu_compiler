#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mlir_list=(
  concat.mlir
  relu.mlir
  permute.mlir
  permute_large.mlir
#  avg_pool.mlir
#  broadcast_add.mlir
#  broadcast_sub.mlir
#  conv.mlir
#  conv3d.mlir
#  deconv.mlir
#  eltwise_add.mlir
#  matmul_transpose.mlir
#  matmul_with_big_k.mlir
#  matmul.mlir
#  max_pool.mlir
#  max_pool3d.mlir
#  quadrstic_sum.mlir
#  search_model.mlir
#  square.mlir
)

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
echo "test_backend.sh success"
exit 0