#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel \
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
npz_compare.py mobilenet_v2_preprocess_opt_all_fp32.npz \
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

npz_compare.py \
    mobilenet_v2_preprocess_tensor_all_int8_multiplier.npz \
    mobilenet_v2_blobs.npz \
    --op_info mobilenet_v2_preprocess_op_info_int8_multiplier.csv \
    --dequant \
    --excepts prob \
    --tolerance 0.95,0.94,0.69 -v

# VERDICT
echo $0 PASSED
