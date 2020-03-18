#!/bin/bash
set -e

#default values
export EXCEPTS=-
export DO_CALIBRATION=0
export CALIBRATION_IMAGE_COUNT=1000
export DO_QUANT_INT8=1
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export DO_QUANT_INT8_MULTIPLER=1
export DO_CMDBUF_INT8=1
export DO_QUANT_BF16=1
export DO_CMDBUF_BF16=1
export DO_DEEPFUSION=0
export DO_QUANT_MIX=0
export DO_ACCURACY_CAFFE=1
export DO_ACCURACY_INTERPRETER=1
export DO_LAYERGROUP=0
if [ -z "$DO_BATCHSIZE" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=$DO_BATCHSIZE
fi
export BATCH_SIZE

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/resnet50/data/resnet50_calibration_table
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc1000
export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.92,0.91,0.59
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.92
export DO_DEEPFUSION=1
export DO_LAYERGROUP=1
# export BATCH_SIZE=4
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/vgg16/data/vgg16_calibration_table
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc8
export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.98,0.98,0.84
export TOLERANCE_INT8_RSHIFT_ONLY=0.99,0.99,0.90
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.92
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_DEEPFUSION=1
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/inception_v3/data/inception_v3_threshold_table
export NET_INPUT_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.84,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.94,0.93,0.64
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.71
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/inception_v4/data/inception_v4_threshold_table
export NET_INPUT_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.84,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.93,0.93,0.62
export TOLERANCE_INT8_MULTIPLER=0.93,0.93,0.64
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "efficientnet_b0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.caffemodel
# export DO_CALIBRATION=1
export CALI_TABLE=$REGRESSION_PATH/efficientnet_b0/data/efficientnet_b0_threshold_table_1000
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export INPUT_SCALE=4.45
export INPUT=data
export OUTPUTS_FP32=_fc
export OUTPUTS=_fc
export DO_QUANT_INT8_PER_TENSOR=0  # need to turn
#export TOLERANCE_INT8_PER_TENSOR=0.04,-0.24,-1.26
export TOLERANCE_INT8_RSHIFT_ONLY=0.37,0.29,-0.57
export TOLERANCE_INT8_MULTIPLER=0.47,0.42,-0.44
export TOLERANCE_BF16=0.99,0.99,0.91
export DO_CMDBUF_BF16=0   # this is a bug to fix
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_QUANT_MIX=1
export TOLERANCE_MIX=0.45,0.40,-0.45
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
# export DO_CALIBRATION=1
export CALI_TABLE=$REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_calibration_table_1000
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc7
export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.61,0.61,0.02
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.94,0.68
export TOLERANCE_INT8_MULTIPLER=0.95,0.94,0.69
export TOLERANCE_BF16=0.99,0.99,0.93
export DO_CMDBUF_BF16=0   # this is a bug to fix
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_DEEPFUSION=1
fi

if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
#export DO_CALIBRATION=1
export CALI_TABLE=$REGRESSION_PATH/shufflenet_v2/data/shufflenet_v2_threshold_table
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=fc
export OUTPUTS=fc
export TOLERANCE_INT8_PER_TENSOR=0.89,0.88,0.50
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.92,0.55
export TOLERANCE_INT8_MULTIPLER=0.92,0.92,0.57
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi
