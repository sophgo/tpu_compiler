# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# translate from caffe, apply all possible pre-calibration optimizations
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir
./bin/mlir-opt \
    --convert-bn-to-scale \
    resnet-50.mlir \
    -o resnet-50-opt1.mlir
./bin/mlir-opt \
    --fold-scale \
    resnet-50-opt1.mlir \
    -o resnet-50-opt2.mlir
./bin/mlir-opt \
    --fuse-scale-into-conv \
    resnet-50-opt2.mlir \
    -o resnet-50-opt3.mlir
cp ResNet-50-model.npz ResNet-50-model-opt3.npz

# import calibration table
./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_DIR/bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt3.mlir \
    -o resnet-50-cali.mlir

# before run cmdbuf
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmkernel/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_support/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmbuilder/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_cmodel/lib:$LD_LIBRARY_PATH

################################
# quantization 1: per-layer int8
################################
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8.mlir

# assign weight address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    resnet-50-quant-int8.mlir \
    -o resnet-50-quant-int8-addr1.mlir

# assign neuron address
./bin/mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-addr1.mlir \
    -o resnet-50-quant-int8-addr2.mlir

# run interpreter, and generate cmdbuf at the same time
./bin/mlir-tpu-interpreter resnet-50-quant-int8-addr2.mlir \
    --generate-cmdbuf=cmdbuf.bin \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    $DATA_DIR/test_cat_in_int8.bin \
    ResNet-50-model.bin \
    cmdbuf.bin \
    out_all.bin \
    25542640 0 25542640 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin $DATA_DIR/test_cat_out_fc1000-int8.bin

python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all.npz tensor_all_quant-int8.npz int8 show 5

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: per-channel multiplier int8
################################
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir

# assign weight address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    resnet-50-quant-int8-multiplier.mlir \
    -o resnet-50-quant-int8-multiplier-addr1.mlir

# assign neuron address
./bin/mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-multiplier-addr1.mlir \
    -o resnet-50-quant-int8-multiplier-addr2.mlir

# run interpreter, to generate cmdbuf
./bin/mlir-tpu-interpreter resnet-50-quant-int8-multiplier-addr2.mlir \
    --generate-cmdbuf=cmdbuf-multiplier.bin \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin

# run interpreter, to generate reference tensor all npz
cp ResNet-50-model_quant_int8_multiplier.npz ResNet-50-model.npz
./bin/mlir-tpu-interpreter \
    resnet-50-quant-int8-multiplier.mlir \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin \
    --dump-all-tensor=tensor_all_quant-int8-multiplier.npz

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    $DATA_DIR/test_cat_in_int8.bin \
    ResNet-50-model.bin \
    cmdbuf-multiplier.bin \
    out_all.bin \
    25542640 0 25542640 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin $DATA_DIR/test_cat_out_fc1000-int8-multiplier.bin

python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all.npz tensor_all_quant-int8-multiplier.npz int8 show 5

# VERDICT
echo $0 PASSED
