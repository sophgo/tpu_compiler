#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    -o resnet50.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    resnet50.mlir \
    -o resnet50_opt.mlir

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/bmnet_resnet50_calibration_table.1x10 \
    resnet50_opt.mlir \
    -o resnet50_cali.mlir

# apply all possible post-calibration optimizations
mlir-opt \
    --fuse-relu \
    --fuse-eltwise \
    resnet50_cali.mlir \
    -o resnet50_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_per_layer.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_layer.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_per_layer.bin \
    --dump-all-tensor=tensor_all.npz
# bin_compare.py out.bin out_int8_per_layer.bin float32 1 1 1 1000 5
npz_to_bin.py tensor_all.npz fc1000 out_fc1000.bin
bin_fp32_to_int8.py out_fc1000.bin out_fc1000_int8.bin
diff out_fc1000_int8.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_per_layer.bin

# quantization 2: per-channel int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_per_channel.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_channel.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_per_channel.bin \
    --dump-all-tensor=tensor_all.npz
# bin_compare.py out.bin out_int8_per_channel.bin float32 1 1 1 1000 5
npz_to_bin.py tensor_all.npz fc1000 out_fc1000.bin
bin_fp32_to_int8.py out_fc1000.bin out_fc1000_int8.bin
diff out_fc1000_int8.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_per_channel.bin

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_multiplier.mlir

mlir-tpu-interpreter resnet50_quant_int8_multiplier.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_multiplier.bin \
    --dump-all-tensor=tensor_all.npz
# bin_compare.py out.bin out_int8_multiplier.bin float32 1 1 1 1000 5
npz_to_bin.py tensor_all.npz fc1000 out_fc1000.bin
bin_fp32_to_int8.py out_fc1000.bin out_fc1000_int8.bin
diff out_fc1000_int8.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_multiplier.bin

# VERDICT
echo $0 PASSED
