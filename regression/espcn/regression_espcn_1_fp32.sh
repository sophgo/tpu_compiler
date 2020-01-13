#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# generate image
randomn_image.py 1 1 512 512 $DATA_PATH/test_espcn_cat_in_fp32 \
    $MODEL_PATH/caffe/espcn_2x.prototxt \
    $MODEL_PATH/caffe/espcn_2x.caffemodel \
    "Conv2D_2"

python $PYTHON_TOOLS_PATH/run_caffe_inference.py \
--model_def $MODEL_PATH/caffe/espcn_2x.prototxt \
--pretrained_model $MODEL_PATH/caffe/espcn_2x.caffemodel \
--images_dim='512,512' \
--mean_file "" \
--channel_swap 0 \
$DATA_PATH/test_espcn_cat_in_fp32.jpg \
caffe_out.npy

npy_to_bin.py caffe_out.npy caffe_ref.bin


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/espcn_2x.prototxt \
    --caffemodel $MODEL_PATH/caffe/espcn_2x.caffemodel \
    -o espcn.mlir

# test mlir interpreter
mlir-tpu-interpreter espcn.mlir \
    --debug \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out.bin

bin_compare.py out.bin caffe_ref.bin \
    float32 1 4 504 504 5 5


# opt1, convert bn to scale
mlir-opt \
    --convert-bn-to-scale \
    espcn.mlir \
    -o espcn_opt1.mlir

# test opt1
mlir-tpu-interpreter espcn_opt1.mlir \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out_opt1.bin
bin_compare.py out.bin out_opt1.bin float32 1 4 504 504 5 5

# opt2, fold consecutive scales
mlir-opt \
    --fold-scale \
    espcn_opt1.mlir \
    -o espcn_opt2.mlir

# test opt2
mlir-tpu-interpreter espcn_opt2.mlir \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out_opt2.bin
bin_compare.py out.bin out_opt2.bin float32 1 4 504 504 5 5

# opt3, merge scale into conv
mlir-opt \
    --merge-scale-into-conv \
    espcn_opt2.mlir \
    -o espcn_opt3.mlir

# test opt3
mlir-tpu-interpreter espcn_opt3.mlir \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out_opt3.bin
bin_compare.py out.bin out_opt3.bin float32 1 4 504 504 5 5

# opt4, fuse relu with conv
mlir-opt \
    --fuse-relu \
    espcn_opt3.mlir \
    -o espcn_opt4.mlir

# test opt4
mlir-tpu-interpreter espcn_opt4.mlir \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out_opt4.bin
bin_compare.py out.bin out_opt4.bin float32 1 4 504 504 5 5

# opt5, fuse eltwise with conv
mlir-opt \
    --fuse-eltwise \
    espcn_opt4.mlir \
    -o espcn_opt5.mlir

# test opt5
mlir-tpu-interpreter espcn_opt5.mlir \
    --tpu-op-stats-filename espcn_op_stats.csv \
    --tensor-in $DATA_PATH/test_espcn_cat_in_fp32.bin \
    --tensor-out out_opt5.bin
bin_compare.py out.bin out_opt5.bin float32 1 4 504 504 5 5

# VERDICT
echo $0 PASSED
