#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/VGG_ILSVRC_16_layers.caffemodel \
    -o vgg16.mlir

# test mlir interpreter
mlir-tpu-interpreter vgg16.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all_fp32.npz

# apply all possible pre-calibration optimizations
#mlir-opt \
#    --convert-bn-to-scale \
#    --fold-scale \
#    --merge-scale-into-conv \
#    resnet50.mlir \
#    -o resnet50_opt.mlir

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

################################
# prepare int8 input
################################
bin_fp32_to_int8.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    in_int8.bin \
    1.0 \
    161.057006836
# check
#diff in_int8.bin $DATA_PATH/test_cat_in_vgg16_int8.bin

################################
# quantization 1: per-layer int8
################################
mlir-opt \
    --quant-int8 \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_per_layer.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    vgg16_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

#Usage: test_bmnet input.bin weight.bin cmdbuf.bin output.bin
#       output_size output_offset neuron_size batch_size

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    15237616 0 15237616 1

#0x00024c00 fc8 neuron address from neuron_map
#bin_extract.py out_all.bin out_fc8.bin int8 0x00024c00 1000
#diff out_fc1000.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_per_layer.bin

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    vgg16_quant_int8_per_layer.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_int8_per_layer.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map.csv out_all_perlayer.npz
npz_compare.py out_all_perlayer.npz tensor_all_int8_per_layer.npz

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: per-channel multiplier int8
################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_multiplier.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    vgg16_quant_int8_multiplier.mlir  | \
    mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    15237616 0 15237616 1
#bin_extract.py out_all.bin out_fc8.bin int8 0x00024c00 1000
#diff out_fc1000.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_multiplier.bin

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    vgg16_quant_int8_multiplier.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_int8_multiplier.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map.csv out_all_perchannel.npz
npz_compare.py out_all_perchannel.npz tensor_all_int8_multiplier.npz



# VERDICT
echo $0 PASSED
