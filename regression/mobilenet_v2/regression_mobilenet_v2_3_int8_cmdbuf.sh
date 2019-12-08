#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    mobilenet_v2.mlir \
    -o mobilenet_v2_opt.mlir

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/bmnet_mobilenet_v2_calibration_table.1x10 \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_cali.mlir

# apply all possible post-calibration optimizations
mlir-opt \
    --fuse-relu \
    mobilenet_v2_cali.mlir \
    -o mobilenet_v2_opt_post_cali.mlir

################################
# prepare int8 input
################################
bin_fp32_to_int8.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    in_int8.bin \
    0.017 \
    2.56929183
# check
diff in_int8.bin $DATA_PATH/test_cat_in_mobilenet_v2_int8.bin

################################
# quantization 1: per-layer int8
################################
mlir-opt \
    --quant-int8 \
    mobilenet_v2_opt_post_cali.mlir \
    -o mobilenet_v2_quant_per_layer.mlir

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
    mobilenet_v2_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    9405584 0 9405584 1
bin_extract.py out_all.bin out_fc7.bin int8 0x00024c00 1000
diff out_fc7.bin $DATA_PATH/test_cat_out_mobilenet_v2_fc7_int8_per_layer.bin

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    mobilenet_v2_quant_int8_per_layer.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_int8_per_layer.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map.csv out_all.npz
npz_compare.py out_all.npz tensor_all_int8_per_layer.npz show 5

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
    mobilenet_v2_opt_post_cali.mlir \
    -o mobilenet_v2_quant_int8_multiplier.mlir

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
    mobilenet_v2_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    9405584 0 9405584 1
bin_extract.py out_all.bin out_fc7.bin int8 0x00024c00 1000
diff out_fc7.bin $DATA_PATH/test_cat_out_mobilenet_v2_fc7_int8_multiplier.bin

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    mobilenet_v2_quant_int8_multiplier.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_int8_multiplier.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map.csv out_all.npz
npz_compare.py out_all.npz tensor_all_int8_multiplier.npz show 5

# VERDICT
echo $0 PASSED
