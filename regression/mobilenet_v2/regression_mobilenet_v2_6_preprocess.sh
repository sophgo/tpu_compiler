#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel \
    --swap_channel\
    --raw_scale 255.0 \
    --mean 103.94,116.78,123.68 \
    --scale 0.017 \
    -o mobilenet_v2_preprocess.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_preprocess_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    mobilenet_v2_preprocess.mlir \
    -o mobilenet_v2_preprocess_opt.mlir

# test 1: fp32
python $REGRESSION_PATH/mobilenet_v2/convert_image.py \
    --image $REGRESSION_PATH/resnet50/data/cat.jpg \
    --save mobilenet_v2_preprocess_in_fp32

mlir-tpu-interpreter mobilenet_v2_preprocess_opt.mlir \
    --tensor-in mobilenet_v2_preprocess_in_fp32.npz \
    --tensor-out mobilenet_v2_preprocess_opt_out_fp32.npz \
    --dump-all-tensor mobilenet_v2_preprocess_opt_all_fp32.npz

# compare with caffe result
cvi_npz_tool.py compare mobilenet_v2_preprocess_opt_all_fp32.npz \
                                    mobilenet_v2_blobs.npz \
                                     --op_info mobilenet_v2_preprocess_op_info.csv \
                                     -vv

# test 2: int8
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_preprocess_threshold_table \
    mobilenet_v2_preprocess_opt.mlir \
    -o mobilenet_v2_preprocess_cali.mlir

mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_preprocess_op_info_int8_multiplier.csv \
    mobilenet_v2_preprocess_cali.mlir \
    -o mobilenet_v2_preprocess_quant_int8_multiplier.mlir

mlir-tpu-interpreter mobilenet_v2_preprocess_quant_int8_multiplier.mlir \
    --tensor-in mobilenet_v2_preprocess_in_fp32.npz \
    --tensor-out mobilenet_v2_preprocess_out_int8_multiplier.npz \
    --dump-all-tensor=mobilenet_v2_preprocess_tensor_all_int8_multiplier.npz

cvi_npz_tool.py compare \
    mobilenet_v2_preprocess_tensor_all_int8_multiplier.npz \
    mobilenet_v2_blobs.npz \
    --op_info mobilenet_v2_preprocess_op_info_int8_multiplier.csv \
    --dequant \
    --excepts prob \
    --tolerance 0.95,0.94,0.69 -v

# test 3: int8 cmdbuf
cvi_npz_tool.py to_bin \
    mobilenet_v2_preprocess_tensor_all_int8_multiplier.npz \
    data \
    mobilenet_v2_preprocess_in_int8.bin \
    int8

cvi_npz_tool.py to_bin mobilenet_v2_preprocess_in_fp32.npz data mobilenet_v2_preprocess_in_fp32.bin

  mlir-opt \
      --tpu-lower \
      mobilenet_v2_preprocess_quant_int8_multiplier.mlir \
      -o mobilenet_v2_preprocess_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=mobilenet_v2_preprocess_weight_map_int8_multiplier.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=mobilenet_v2_preprocess_neuron_map_int8_multiplier.csv \
    mobilenet_v2_preprocess_quant_int8_multiplier_tg.mlir \
    -o mobilenet_v2_preprocess_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2_preprocess_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir mobilenet_v2_preprocess_quant_int8_multiplier_addr.mlir \
    --output=mobilenet_v2_preprocess_int8_multiplier.cvimodel

# run cvimodel
model_runner \
    --dump-all-tensors \
    --input mobilenet_v2_preprocess_in_fp32.npz \
    --model mobilenet_v2_preprocess_int8_multiplier.cvimodel \
    --output mobilenet_v2_preprocess_cmdbuf_out_all_int8_multiplier.npz

cvi_npz_tool.py compare \
    mobilenet_v2_preprocess_cmdbuf_out_all_int8_multiplier.npz \
    mobilenet_v2_preprocess_tensor_all_int8_multiplier.npz \
    --op_info mobilenet_v2_preprocess_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
