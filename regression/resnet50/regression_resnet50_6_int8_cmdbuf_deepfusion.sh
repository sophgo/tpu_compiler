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

################################
# quantization, per-channel multiplier int8
################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_multiplier.mlir

################################
# deepfusion, simple version first
################################
mlir-opt \
    --deep-fusion-simple \
    --deep-fusion-simple-stats=resnet50_deepfusion_stats.csv \
    resnet50_quant_int8_multiplier.mlir \
    -o resnet50_opt_deepfusion.mlir

################################
# backend
################################
if false; then

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
    resnet50_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

################################
# prepare int8 input
################################
bin_fp32_to_int8.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    in_int8.bin \
    1.0 \
    161.008057
# check
diff in_int8.bin $DATA_PATH/test_cat_in_resnet50_int8.bin

################################
# run cmdbuf with cmodel
################################
$RUNTIME_PATH/bin/test_bmnet \
    in_int8.bin \
    weight.bin \
    cmdbuf.bin \
    out_all.bin \
    16460784 0 16460784 1
bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
diff out_fc1000.bin $DATA_PATH/test_cat_out_resnet50_fc1000_int8_multiplier.bin

################################
# verify result
################################
# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter \
    resnet50_quant_int8_multiplier.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out dummy.bin \
    --dump-all-tensor=tensor_all_int8_multiplier.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map.csv out_all.npz
npz_compare.py out_all.npz tensor_all_int8_multiplier.npz show 5

fi

# VERDICT
echo $0 PASSED
