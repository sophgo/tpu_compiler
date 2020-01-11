#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/VGG_ILSVRC_16_layers.caffemodel \
    -o vgg16.mlir

# apply all possible pre-calibration optimizations


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/bmnet_vgg16_calibration_table.1X10 \
    vgg16.mlir \
    -o vgg16_cali.mlir

# apply all possible post-calibration optimizations
mlir-opt \
    --fuse-relu \
    vgg16_cali.mlir \
    -o vgg16_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_per_layer.mlir


mlir-tpu-interpreter vgg16_quant_int8_per_layer.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_per_layer.bin \
    --dump-all-tensor=tensor_all.npz
# bin_compare.py out.bin out_int8_per_layer.bin float32 1 1 1 1000 5
npz_to_bin.py tensor_all.npz fc8 out_fc8.bin
bin_fp32_to_int8.py out_fc8.bin out_fc8_int8.bin

#diff out_fc8_int8.bin $DATA_PATH/test_cat_out_vgg16_fc8_int8_per_layer.bin


# quantization 2: per-channel int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_per_channel.mlir

mlir-tpu-interpreter vgg16_quant_int8_per_channel.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_per_channel.bin \
    --dump-all-tensor=tensor_all.npz
# bin_compare.py out.bin out_int8_per_channel.bin float32 1 1 1 1000 5
#npz_to_bin.py tensor_all.npz fc1000 out_fc8.bin
#bin_fp32_to_int8.py out_fc8.bin out_fc8_int8.bin
#diff out_fc8_int8.bin $DATA_PATH/test_cat_out_vgg16_fc8_int8_per_channel.bin


# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_multiplier.mlir

mlir-tpu-interpreter vgg16_quant_int8_multiplier.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_int8_multiplier.bin \
    --dump-all-tensor=tensor_all.npz
#bin_compare.py out.bin out_int8_multiplier.bin float32 1 1 1 1000 5
npz_to_bin.py tensor_all.npz fc8 out_fc8.bin
bin_fp32_to_int8.py out_fc8.bin out_fc8_int8.bin
#diff out_fc8_int8.bin $DATA_PATH/test_cat_out_vgg16_fc8_int8_multiplier.bin


# VERDICT
echo $0 PASSED
