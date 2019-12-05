# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# for run cmdbuf
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmkernel/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_support/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmbuilder/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_cmodel/lib:$LD_LIBRARY_PATH

# translate from caffe
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir

# apply all possible pre-calibration optimizations
./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    --fuse-relu \
    resnet-50.mlir \
    -o resnet-50-opt.mlir

################################
# prepare bf16 input
################################
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_bf16.py \
    $DATA_DIR/test_cat_in_fp32.bin \
    in_bf16.bin
# check
diff in_bf16.bin $DATA_DIR/test_cat_in_bf16.bin

################################
# quantization
################################
./bin/mlir-opt \
    --quant-bf16 \
    resnet-50-opt.mlir \
    -o resnet-50-quant-bf16.mlir

# run interpreter, to generate reference tensor all npz
./bin/mlir-tpu-interpreter resnet-50-quant-bf16.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-bf16.bin \
    --dump-all-tensor=tensor_all_quant-bf16.npz

# assign weight address & neuron address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    resnet-50-quant-bf16.mlir \
    -o resnet-50-quant-bf16-addr.mlir

# backend translate into cmdbuf
./bin/mlir-translate \
    --mlir-to-cmdbuf \
    resnet-50-quant-bf16-addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all.bin \
    32921552 0 32921552 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000_bf16.bin bf16 0x00049800 1000
diff out_fc1000_bf16.bin $DATA_DIR/test_cat_out_fc1000-bf16.bin
# python ../llvm/projects/mlir/externals/python_tools/bin_bf16_to_fp32.py out_fc1000_bf16.bin out_fc1000_fp32.bin
# python ../llvm/projects/mlir/externals/python_tools/bin_dump.py out_fc1000_fp32.bin float32 1 1 1 1000 5

# compare all tensors
python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map_bf16.csv out_all_bf16.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all_bf16.npz tensor_all_quant-bf16.npz show 5

# VERDICT
echo $0 PASSED
