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
    resnet-50.mlir \
    -o resnet-50-opt.mlir

# import calibration table
./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_DIR/bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt.mlir \
    -o resnet-50-cali.mlir

# apply all possible post-calibration optimizations
./bin/mlir-opt \
    --fuse-relu \
    resnet-50-cali.mlir \
    -o resnet-50-opt-post-cali.mlir

################################
# prepare int8 input
################################
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    $DATA_DIR/test_cat_in_fp32.bin \
    in_int8.bin \
    1.0 \
    161.008057
# check
diff in_int8.bin $DATA_DIR/test_cat_in_int8.bin

################################
# quantization 1: per-layer int8
################################
./bin/mlir-opt \
    --quant-int8 \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-quant-int8.mlir

# assign weight address & neuron address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    16460784 0 16460784 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin $DATA_DIR/test_cat_out_fc1000-int8.bin

# run interpreter, to generate reference tensor all npz
./bin/mlir-tpu-interpreter resnet-50-quant-int8.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz

# compare all tensors
python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all.npz tensor_all_quant-int8.npz show 5

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: per-channel multiplier int8
################################
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir

# assign weight address & neuron address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight-multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-multiplier.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf-multiplier.bin

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    in_int8.bin \
    weight-multiplier.bin \
    cmdbuf-multiplier.bin \
    out_all.bin \
    16460784 0 16460784 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin $DATA_DIR/test_cat_out_fc1000-int8-multiplier.bin

# run interpreter, to generate reference tensor all npz
./bin/mlir-tpu-interpreter \
    resnet-50-quant-int8-multiplier.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin \
    --dump-all-tensor=tensor_all_quant-int8-multiplier.npz

# compare all tensors
python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all.npz tensor_all_quant-int8-multiplier.npz show 5

# VERDICT
echo $0 PASSED
