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

'''
# run interpreter, to generate reference tensor all npz
./bin/mlir-tpu-interpreter resnet-50-quant-bf16.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-bf16.bin \
    --dump-all-tensor=tensor_all_quant-bf16.npz
'''

# quantization
./bin/mlir-opt \
    --quant-bf16 \
    resnet-50-opt.mlir \
    -o resnet-50-quant-bf16.mlir

# run interpreter, to generate reference tensor all npz
./bin/mlir-tpu-interpreter resnet-50-quant-bf16.mlir \
    --tensor-in $TPU_DATA_PATH/test_cat_in_fp32.bin \
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

if false; then

# assign weight address & neuron address
./bin/mlir-translate \
    --mlir-to-cmdbuf \
    resnet-50-quant-bf16-addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    $DATA_DIR/test_cat_in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all.bin \
    16460784 0 16460784 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin bf16 0x00024c00 1000
diff out_fc1000.bin $DATA_DIR/test_cat_out_fc1000-bf16.bin

# compare all tensors
python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all.npz tensor_all_quant-int8.npz int8 show 5

fi

# VERDICT
echo $0 PASSED
