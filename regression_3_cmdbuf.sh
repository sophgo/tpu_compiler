# !/bin/bash
set -e

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
    --calibration-table bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt3.mlir \
    -o resnet-50-cali.mlir

# quantization 1: per-layer int8
cp ResNet-50-model-opt3.npz ResNet-50-model.npz
./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8.mlir
cp ResNet-50-model.npz ResNet-50-model_quant_int8.npz

# run interpreter, and generate cmdbuf at the same time
./bin/mlir-tpu-interpreter resnet-50-quant-int8.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz

# run cmdbuf
export LD_LIBRARY_PATH=~/work_cvitek/install_bmkernel/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/work_cvitek/install_support/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/work_cvitek/install_bmbuilder/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/work_cvitek/install_cmodel/lib:$LD_LIBRARY_PATH

~/work_cvitek/install_runtime/bin/test_bmnet \
    test_cat_in_int8.bin \
    ~/work_cvitek/llvm-project/build/ResNet-50-model.bin \
    ~/work_cvitek/llvm-project/build/cmdbuf.bin \
    out_all.bin \
    25542640 0 25542640 1
python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin test_cat_out_fc1000-int8.bin

python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map.csv out_all.npz
python ../llvm/projects/mlir/externals/python_tools//npz_compare.py \
    out_all.npz tensor_all_quant-int8.npz int8 show 5

# VERDICT
echo $0 PASSED
