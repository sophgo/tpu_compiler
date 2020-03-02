#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -e $NET ]; then
  mkdir $NET
fi

export NET=$NET

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/resnet50/data/resnet50_calibration_table
export IMAGE_DIM=224,224
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export RAW_SCALE=255.0
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc1000
export TOLERANCE_PER_TENSOR=0.92,0.91,0.59
export TOLERANCE_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/vgg16/data/vgg16_calibration_table
export IMAGE_DIM=224,224
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc8
export TOLERANCE_PER_TENSOR=0.98,0.98,0.84
export TOLERANCE_RSHIFT_ONLY=0.99,0.99,0.90
export TOLERANCE_MULTIPLER=0.99,0.99,0.92
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/inception_v3/data/inception_v3_threshold_table
export IMAGE_DIM=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
export TOLERANCE_PER_TENSOR=0.98,0.98,0.84
export TOLERANCE_RSHIFT_ONLY=0.99,0.99,0.90
export TOLERANCE_MULTIPLER=0.99,0.99,0.92
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/inception_v4/data/inception_v4_threshold_table
export IMAGE_DIM=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
export TOLERANCE_PER_TENSOR=0.98,0.98,0.84
export TOLERANCE_RSHIFT_ONLY=0.99,0.99,0.90
export TOLERANCE_MULTIPLER=0.99,0.99,0.92
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "efficientnet_b0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.caffemodel
export CALI_TABLE=$REGRESSION_PATH/efficientnet_b0/data/efficientnet_b0_threshold_table
export IMAGE_DIM=256,256
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export INPUT_SCALE=4.45
export INPUT=data
export OUTPUTS_FP32=_fc
export OUTPUTS=_global_avg_pool
export TOLERANCE_PER_TENSOR=" -0.14,-0.24,-1.26"
export TOLERANCE_RSHIFT_ONLY=0.37,0.29,-0.57
export TOLERANCE_MULTIPLER=0.38,0.30,-0.55
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
export CALI_TABLE=$REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_calibration_table
export IMAGE_DIM=299,299
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc7
export TOLERANCE_PER_TENSOR=0.58,0.56,-0.03
export TOLERANCE_RSHIFT_ONLY=0.91,0.89,0.57
export TOLERANCE_MULTIPLER=0.91,0.90,0.57
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

pushd $NET
# clear previous output
# rm -f *.mlir *.bin *.npz *.csv

# run tests
$DIR/regression_0_caffe.sh
$DIR/regression_1_fp32.sh
$DIR/regression_2_int8.sh
$DIR/regression_3_int8_cmdbuf.sh
$DIR/regression_4_bf16.sh
$DIR/regression_5_bf16_cmdbuf.sh
# $DIR/regression_6_int8_cmdbuf_deepfusion.sh

popd

# VERDICT
echo $0 PASSED
