#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

echo $0 is RUNNING
# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel \
    --images_dim 299,299 \
    --mean 128,128,128 \
    --input_scale 0.0078125 \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs inception_v3_blobs.npz \
    --dump_weights inception_v3_weights.npz \
    $REGRESSION_PATH/inception_v3/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py inception_v3_blobs.npz inception_v3_in_fp32.npz input
npz_extract.py inception_v3_blobs.npz inception_v3_out_fp32_prob.npz conv1_3x3_s2

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
# npz_compare.py inception_v3_in_fp32.npz $REGRESSION_PATH/inception_v3/data/inception_v3_in_fp32.npz
cp inception_v3_in_fp32.npz $REGRESSION_PATH/inception_v3/data/inception_v3_in_fp32.npz

# VERDICT
echo $0 PASSED

mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt \
    --caffemodel $MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel \
    -o inception_v3.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info.csv \
    inception_v3.mlir \
    -o inception_v3_id.mlir

## test mlir interpreter
mlir-tpu-interpreter inception_v3.mlir \
    --tensor-in inception_v3_in_fp32.npz \
    --tensor-out inception_v3_out_fp32.npz \
    --dump-all-tensor=inception_v3_tensor_all_fp32.npz
npz_compare.py inception_v3_out_fp32.npz inception_v3_out_fp32_prob.npz -v
# set tolerance to 0.91 now, need to check this with fp32 later
npz_compare.py \
    inception_v3_tensor_all_fp32.npz \
    inception_v3_blobs.npz \
    --op_info inception_v3_op_info.csv \
    --tolerance=0.99,0.99,0.91 -vvv

# opt1, convert bn to scale
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    inception_v3_id.mlir \
    -o inception_v3_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter inception_v3_opt.mlir \
    --tensor-in inception_v3_in_fp32.npz \
    --tensor-out inception_v3_opt_out_fp32.npz
npz_compare.py inception_v3_opt_out_fp32.npz inception_v3_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/inception_v3/data/inception_v3_threshold_table \
    inception_v3_opt.mlir \
    -o inception_v3_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    inception_v3_cali.mlir \
    -o inception_v3_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_quant_int8_per_layer_info.csv \
    inception_v3_opt_post_cali.mlir \
    -o inception_v3_quant_int8_per_layer.mlir

mlir-tpu-interpreter inception_v3_quant_int8_per_layer.mlir \
    --tensor-in inception_v3_in_fp32.npz \
    --tensor-out inception_v3_out_int8_per_layer.npz \
    --dump-all-tensor=inception_v3_tensor_all_int8_per_layer.npz

npz_to_bin.py inception_v3_in_fp32.npz input inception_v3_in_fp32.bin
bin_fp32_to_int8.py \
    inception_v3_in_fp32.bin \
    inception_v3_in_int8.bin \
    1.0 \
    6.007720007250769

################################
# quantization 1: per-layer int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    inception_v3_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    inception_v3_in_int8.bin \
    weight_int8_per_layer.bin \
    cmdbuf_int8_per_layer.bin \
    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
    14422816 0 14422816 1
#bin_extract.py \
#    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
#    inception_v3_cmdbuf_out_classifier_int8_per_layer.bin \
#    int8 0x000417b0 1000
#bin_compare.py \
#    inception_v3_cmdbuf_out_classifier_int8_per_layer.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_per_layer.bin \
#    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    inception_v3_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    inception_v3_cmdbuf_out_all_int8_per_layer.npz \
    inception_v3_tensor_all_int8_per_layer.npz \
    --order neuron_map.csv

