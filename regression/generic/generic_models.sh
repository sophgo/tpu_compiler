#!/bin/bash
set -e

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export STD=1,1,1
export MODEL_CHANNEL_ORDER="bgr"
export DATA_FORMAT="nchw"
export EXCEPTS=-
export ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD=""
export DO_CALIBRATION=0
export CALIBRATION_IMAGE_COUNT=1000
export MLIR_OPT_FE_PRE="--convert-bn-to-scale"
export MLIR_OPT_FE_POST="--eltwise-early-stride"
export MLIR_OPT_BE="--tg-fuse-leakyrelu --conv-ic-alignment"
export TOLERANCE_FP32=0.999,0.999,0.998
export DO_QUANT_INT8=1
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export DO_CMDBUF_INT8=1
export DO_QUANT_BF16=0
export DO_CMDBUF_BF16=0
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export DO_QUANT_MIX=0
export DO_ACCURACY_CAFFE=1
export DO_ACCURACY_ONNX=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_ACCURACY_INTERPRETER=1
export DO_E2E=1
export USE_LAYERGROUP=1
export EVAL_MODEL_TYPE="imagenet"
export LABEL_FILE=$REGRESSION_PATH/data/synset_words.txt
export DO_NN_TOOLKIT=0
export SET_CHIP_NAME="cv183x"
export SWAP_CHANNEL=0,1,2
export YOLO_PREPROCESS="false"
export BGRAY=0

if [ -z "$DO_BATCHSIZE" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=$DO_BATCHSIZE
fi
export BATCH_SIZE
if [ -z "$ENABLE_PREPROCESS" ]; then
  DO_PREPROCESS=0
else
  DO_PREPROCESS=$ENABLE_PREPROCESS
fi
export DO_PREPROCESS
# default inference and test image, for imagenet models only
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/cat.jpg

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc1000
#export EXCEPTS=prob
#export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu   # for "--eltwise-early-stride"
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.91,0.89,0.56
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.72
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_LAYERGROUP=1
export DO_NN_TOOLKIT=1
export DO_QUANT_MIX=1
export TOLERANCE_MIX_PRECISION=0.96,0.95,0.73
export MIX_PRECISION_BF16_LAYER_NUM=10

# export BATCH_SIZE=4
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,prob,res2c_relu,res3d_relu,res4f_relu
else
  export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
fi
fi

if [ $NET = "resnext50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnext/caffe/resnext50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnext/caffe/resnext50.caffemodel
export LABEL_FILE=$MODEL_PATH/imagenet/resnext/caffe/corresp.txt
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=prob
export OUTPUTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.83,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.69
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.72
export TOLERANCE_BF16=0.99,0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.91
# compare failed in board if open layergroup
export DO_LAYERGROUP=0
export USE_LAYERGROUP=0
export DO_PREPROCESS=0
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc8
export EXCEPTS=prob
export DO_QUANT_BF16=0
export TOLERANCE_INT8_PER_TENSOR=0.98,0.98,0.84
export TOLERANCE_INT8_RSHIFT_ONLY=0.99,0.99,0.90
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.90
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=prob
export OUTPUTS=prob
# export DO_QUANT_INT8_PER_TENSOR=1
# export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.96,0.95,0.72
export TOLERANCE_INT8_RSHIFT_ONLY=0.96,0.96,0.71
export TOLERANCE_INT8_MULTIPLER=0.96,0.96,0.71
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.97
export DO_CMDBUF_BF16=0
#export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,prob
else
  export EXCEPTS=prob
fi
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.84,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.94,0.93,0.64
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.68
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.84,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.93,0.93,0.62
export TOLERANCE_INT8_MULTIPLER=0.93,0.93,0.63
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.89
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "mobilenet_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc7
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.89,0.86,0.52
export TOLERANCE_INT8_RSHIFT_ONLY=0.96,0.95,0.73
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.92
export DO_CMDBUF_BF16=1
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.95
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,prob
else
  export EXCEPTS=prob
fi
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc7
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.61,0.60,0.01
export TOLERANCE_INT8_RSHIFT_ONLY=0.94,0.9,0.66
export TOLERANCE_INT8_MULTIPLER=0.94,0.94,0.67
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.92
export DO_CMDBUF_BF16=0   # this is a bug to fix
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,prob
else
  export EXCEPTS=prob
fi
fi

if [ $NET = "mobilenet_v3" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v3/onnx/mobilenetv3_rw.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_mobilenetv3_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenetv3_calibration_table
export RAW_SCALE=1
export INPUT_SCALE=0.875
export NET_INPUT_DIMS=256,256
export MEAN=0.406,0.456,0.485  # in BGR, pytorch mean=[0.485, 0.456, 0.406]
export STD=0.225,0.224,0.229   # in BGR, pytorch std=[0.229, 0.224, 0.225]
export IMAGE_RESIZE_DIMS=224,224
export INPUT=input
export COMPARE_ALL=1
export TOLERANCE_INT8_MULTIPLER=0.083338,-0.1,-1.0
export DO_QUANT_INT8_MULTIPLER=0
export DO_CMDBUF_INT8=0
export DO_QUANT_BF16=1
export DO_CMDBUF_BF16=1
export DO_NN_TOOLKIT=0
export TOLERANCE_BF16=0.9,0.9,0.9
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export EXCEPTS=371_Add,315_Add,390_Add,401_Add,409_Add,419_Add,427_Add,438_Add,446_Add,457_Add,465_Add,476_Add,484_Add,492_Add,499_Add,509_Add,517_Add,525_Add,532_Add,543_Add,551_Add,559_Add,566_Add,576_Add,584_Add,592_Add,599_Add,610_Add,618_Add,626_Add,633_Add,644_Add,652_Add # cuz relu6 could add 'relu' layer that could mismatch original layer
export MLIR_OPT_FE_PRE="$MLIR_OPT_FE_PRE --skip-mult-used-scale-op --relu6-to-clip"
export MLIR_OPT_FE_INT8_MULTIPLER_PRE="--tpu-quant-clip"
export BF16_QUANT_LAYERS_FILE=${NET}_bf16_quant_layers
export BF16_QUANT_LAYERS="316_Clip 354_Clip 372_Clip 391_Clip 402_Clip 410_Clip 420_Clip 428_Clip 428_Clip 439_Clip 447_Clip 458_Clip 466_Clip 477_Clip 485_Clip 493_Clip 500_Clip 510_Clip 518_Clip 526_Clip 533_Clip 544_Clip 552_Clip 560_Clip 567_Clip 577_Clip 585_Clip 593_Clip 600_Clip 611_Clip 619_Clip 627_Clip 634_Clip 645_Clip 653_Clip #656_Mul 313_BatchNormalization 315_Add 319_Mul 322_Relu 353_Add"
export DO_CMDBUF_BF16=0
export DO_DEEPFUSION=0
export DO_MEMOPT=0
export DO_DEEPFUSION=0
export DO_LAYERGROUP=0
export DO_E2E=0
export USE_LAYERGROUP=0
export DO_PREPROCESS=0
fi

if [ $NET = "nasnet_mobile" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/nasnet_mobile/onnx/nasnet_mobile.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/nasnet_mobile_calibration_table
export RAW_SCALE=1
export INPUT_SCALE=1
export NET_INPUT_DIMS=224,224
export MEAN=0.5,0.5,0.5  # in BGR, pytorch mean=[0.5, 0.5, 0.5]
export STD=0.5,0.5,0.5   # in BGR, pytorch std=[0.5, 0.5, 0.5]
export IMAGE_RESIZE_DIMS=256,256
export INPUT=input
export COMPARE_ALL=1
export TOLERANCE_INT8_MULTIPLER=0.77,0.86,0.277
export ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD="--enable-cali-overwrite-threshold-forward-relu"
export DO_CALIBRATION=0
export CALIBRATION_IMAGE_COUNT=2000
export DO_QUANT_INT8_MULTIPLER=1
export DO_CMDBUF_INT8=0
export DO_QUANT_BF16=1
export DO_CMDBUF_BF16=1
export DO_NN_TOOLKIT=0
export TOLERANCE_BF16=0.9,0.9,0.9
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export MLIR_OPT_FE_PRE="$MLIR_OPT_FE_PRE --skip-mult-used-scale-op --relu6-to-clip"
export MLIR_OPT_FE_INT8_MULTIPLER_PRE="--tpu-quant-clip"
export BF16_QUANT_LAYERS_FILE=${NET}_bf16_quant_layers
export DO_CMDBUF_BF16=0
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export DO_E2E=1
export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=1
export DO_QUANT_INT8_MULTIPLER=1
export DO_PREPROCESS=0
fi

if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
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
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "squeezenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=prob
export OUTPUTS=pool10
export TOLERANCE_INT8_PER_TENSOR=0.9,0.9,0.55
export TOLERANCE_INT8_RSHIFT_ONLY=0.9,0.9,0.6
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.56
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "densenet_121" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=fc6
export OUTPUTS=fc6
# export EXCEPTS=prob
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.74,0.73,0.23
export TOLERANCE_INT8_RSHIFT_ONLY=0.83,0.82,0.39
export TOLERANCE_INT8_MULTIPLER=0.82,0.81,0.37
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "densenet_201" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=fc6
export OUTPUTS=fc6
# export EXCEPTS=prob
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.57,0.55,-0.05
export TOLERANCE_INT8_RSHIFT_ONLY=0.78,0.77,0.30
export TOLERANCE_INT8_MULTIPLER=0.77,0.76,0.28
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.92
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.95
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "senet_res50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
export NET_INPUT_DIMS=225,225
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc6
# export EXCEPTS=prob
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.94,0.93,0.66
export TOLERANCE_INT8_RSHIFT_ONLY=0.96,0.95,0.71
export TOLERANCE_INT8_MULTIPLER=0.96,0.96,0.75
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.99
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "arcface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/Aaron_Eckhart_0001.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export NET_INPUT_DIMS=112,112
export IMAGE_RESIZE_DIMS=112,112
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.0078125
export TOLERANCE_INT8_MULTIPLER=0.45,0.45,-0.39
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# accuracy setting
export EVAL_MODEL_TYPE="lfw"
#export DO_ACCURACY_CAFFE=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "retinaface_mnet25_600" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_retinaface_mnet25_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table_preprocess
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8_MULTIPLER=0.90,0.85,0.54
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "retinaface_mnet25" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_retinaface_mnet25_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table_preprocess
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=320,320
export NET_INPUT_DIMS=320,320
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8_MULTIPLER=0.90,0.85,0.54
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# accuracy setting
export NET_INPUT_DIMS=320,320
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "retinaface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8_MULTIPLER=0.86,0.83,0.49
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "ssd300" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/ssd300/labelmap_coco.prototxt
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_ssd_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export INPUT=data
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=104.0,117.0,123.0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.99,0.99,0.89
export TOLERANCE_INT8_RSHIFT_ONLY=0.98,0.98,0.81
export TOLERANCE_INT8_MULTIPLER=0.98,0.98,0.85
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_CAFFE="eval_caffe_detector_ssd.py"
export EVAL_SCRIPT_INT8="eval_ssd.py"
#export DO_ACCURACY_CAFFE=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,detection_out
else
  export EXCEPTS=detection_out
fi
fi

if [ $NET = "mobilenet_ssd" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/labelmap_voc.prototxt
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_ssd_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export INPUT=data
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.007843
export TOLERANCE_INT8_PER_TENSOR=0.93,0.87,0.62
export TOLERANCE_INT8_RSHIFT_ONLY=0.97,0.97,0.70
export TOLERANCE_INT8_MULTIPLER=0.98,0.96,0.77
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export EVAL_MODEL_TYPE="voc2012"
export EVAL_SCRIPT_VOC="eval_detector_voc.py"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,detection_out
else
  export EXCEPTS=detection_out
fi
fi

if [ $NET = "yolo_v1_448" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v1_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v1_448_calibration_table
export YOLO_PREPROCESS="true"
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=448,448
export NET_INPUT_DIMS=448,448
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.88,0.88,0.46
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v2_1080" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy_1080.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v2_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table_preprocess
export YOLO_PREPROCESS="true"
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=1080,1920
export NET_INPUT_DIMS=1080,1920
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.88,0.88,0.46
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v2_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v2_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table_preprocess
export YOLO_PREPROCESS="true"
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=416,416
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.88,0.88,0.46
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.89,0.90,0.49
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_608" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
export YOLO_PREPROCESS="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=608,608
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.89,0.86,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.92,0.90,0.59
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="false"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
export YOLO_PREPROCESS="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=416,416
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.91,0.60
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="false"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_416_onnx" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/onnx/yolov3-416.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_yolo_v3_0_onnx.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_416_onnx_threshold_table
export NET_INPUT_DIMS=416,416
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.82,0.79,0.29
export DO_QUANT_BF16=0
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export DO_PREPROCESS=0
fi

if [ $NET = "yolo_v3_320" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
export YOLO_PREPROCESS="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=320,320
export NET_INPUT_DIMS=320,320
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="false"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_160" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_160.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
export YOLO_PREPROCESS="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=160,160
export NET_INPUT_DIMS=160,160
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.59
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="false"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_512x288" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_512x288.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
export YOLO_PREPROCESS="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=288,512
export NET_INPUT_DIMS=288,512
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="false"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "yolo_v3_tiny" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_tiny_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_tiny_calibration_table_preprocess
export INPUT=input
export YOLO_PREPROCESS="true"
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=416,416
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export EXCEPTS=output
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.97,0.97,0.76
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export TINY="true"
export SPP_NET="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,output
fi
fi

if [ $NET = "yolo_v3_spp" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_spp_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/yolo_v3_spp_calibration_table_preprocess
export INPUT=input
export YOLO_PREPROCESS="true"
export MODEL_CHANNEL_ORDER="rgb"
export SWAP_CHANNEL=2,1,0
export IMAGE_RESIZE_DIMS=608,608
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export EXCEPTS=output
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.86,0.86,0.34
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export SPP_NET="true"
export TINY="false"
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,output
fi
fi

if [ $NET = "resnet18" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=output
export OUTPUTS=output
export DO_NN_TOOLKIT=1
# export DO_QUANT_INT8_PER_TENSOR=1
# export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.97,0.97,0.78
export TOLERANCE_INT8_RSHIFT_ONLY=0.98,0.98,0.84
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.86
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_PREPROCESS=0
fi

if [ $NET = "efficientnet_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
# export DO_CALIBRATION=1
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientnet_b0_onnx_calibration_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB,
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=output
export OUTPUTS=output
export EXCEPTS=424_Mul,388_Sigmoid
export TOLERANCE_FP32=0.999,0.999,0.97 # we leverage bf16 lut do sigmoid
# export DO_QUANT_INT8_PER_TENSOR=1
# export DO_QUANT_INT8_RFHIFT_ONLY=1
# export TOLERANCE_INT8_PER_TENSOR=0.8,0.8,0.8
# export TOLERANCE_INT8_RSHIFT_ONLY=0.8,0.8,0.8
export TOLERANCE_INT8_MULTIPLER=0.76,0.60,0.26
export TOLERANCE_BF16=0.99,0.99,0.91
export DO_CMDBUF_BF16=0
export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=1
export DO_LAYERGROUP=1
export DO_PREPROCESS=0
export DO_QUANT_MIX=1
export TOLERANCE_MIX_PRECISION=0.76,0.60,0.27
export MIX_PRECISION_BF16_LAYER_NUM=50
fi

if [ $NET = "gru_toy" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/toy/gru_toy.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_gru_toy_0_onnx.sh
export INPUT=input
export OUTPUTS_FP32=output
export OUTPUTS=output
fi

if [ $NET = "efficientnet-lite_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-lite/b0/onnx/efficientnet_lite.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientnet-lite-b0_onnx_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127,127,127
export STD=128,128,128
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=output
export OUTPUTS=output
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.7
export TOLERANCE_BF16=0.99,0.99,0.91

export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=1
export DO_LAYERGROUP=1
export DO_PREPROCESS=0
fi

if [ $NET = "alphapose" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_alphapose_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/alphapose_calibration_table
export IMAGE_RESIZE_DIMS=256,192
export NET_INPUT_DIMS=256,192
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.406,0.457,0.48 # in RGB
export STD=1.0,1.0,1.0
export INPUT_SCALE=1.0
export INPUT=input
export EXCEPTS=404_Relu
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.66
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_PREPROCESS=0
fi

if [ $NET = "espcn_3x" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/super_resolution/espcn/onnx/espcn_3x.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_espcn_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/espcn_3x_calibration_table
export NET_INPUT_DIMS=85,85
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_FP32=0.999,0.999,0.997 # we leverage bf16 lut do sigmoid
export TOLERANCE_INT8_PER_TENSOR=0.97,0.96,0.80
export TOLERANCE_INT8_RSHIFT_ONLY=0.97,0.96,0.80
export TOLERANCE_INT8_MULTIPLER=0.98,0.97,0.81
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
#export TOLERANCE_BF16=0.99,0.99,0.94
#export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_PREPROCESS=0
fi

if [ $NET = "unet" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/segmentation/unet/onnx/unet.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_unet_0_onnx.sh
export IMAGE_PATH=$REGRESSION_PATH/data/0.png
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_0
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=256,256
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=prob
export CALIBRATION_IMAGE_COUNT=30
export DO_CALIBRATION=0
export TOLERANCE_INT8_PER_TENSOR=0.99,0.98,0.91
export TOLERANCE_INT8_RSHIFT_ONLY=0.99,0.98,0.91
export TOLERANCE_INT8_MULTIPLER=0.99,0.98,0.91
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.98,0.97
export DO_CMDBUF_BF16=0
export DO_LAYERGROUP=1
export EVAL_MODEL_TYPE="isbi"
export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=1
export DO_PREPROCESS=0
export BGRAY=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

if [ $NET = "ecanet50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/ecanet/onnx/ecanet50.onnx
export MODEL_DAT=""
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export STD=0.229,0.224,0.225
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=output_Gemm
export OUTPUTS=output_Gemm
export TOLERANCE_INT8_PER_TENSOR=0.91,0.91,0.58
export TOLERANCE_INT8_RSHIFT_ONLY=0.91,0.91,0.58
export TOLERANCE_INT8_MULTIPLER=0.96,0.96,0.72
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_PREPROCESS=0
fi

if [ $NET = "res2net50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/res2net/onnx/res2net50_48w_2s.onnx
export MODEL_DAT=""
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export STD=0.229,0.224,0.225
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=output_Gemm
export OUTPUTS=output_Gemm
export TOLERANCE_INT8_MULTIPLER=0.94,0.94,0.65
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
# export BATCH_SIZE=4
export DO_PREPROCESS=0
fi

if [ $NET = "erfnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_deploy_mergebn.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_cityscapes_mergebn.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_erfnet_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/erfnet_calibration_table
export DO_CALIBRATION=0
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export CALIBRATION_IMAGE_COUNT=60
export INPUT=data
export OUTPUTS_FP32=Deconvolution23_deconv
export OUTPUTS=Deconvolution23_deconv
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.91,0.89,0.56
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_LAYERGROUP=1
export DO_NN_TOOLKIT=0
# export BATCH_SIZE=4
export DO_PREPROCESS=0
fi

if [ $NET = "enet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/enet/caffe/enet_deploy_final.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/enet/caffe/cityscapes_weights.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_erfnet_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/enet_calibration_table
export DO_CALIBRATION=0
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export CALIBRATION_IMAGE_COUNT=60
export INPUT=data
export OUTPUTS_FP32=Deconvolution23_deconv
export OUTPUTS=Deconvolution23_deconv
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export TOLERANCE_INT8_PER_TENSOR=0.91,0.89,0.56
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_LAYERGROUP=1
export DO_NN_TOOLKIT=0
# export BATCH_SIZE=4
export DO_PREPROCESS=0
fi

if [ $NET = "resnet50_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/tensorflow/resnet50
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export MLIR_OPT_FE_POST=""
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/resnet50_tensorflow_calibration_table_1000
export IMAGE_RESIZE_DIMS=224,224
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=116.67,104.01,122.68 # in BGR
export STD=1,1,1
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.97,0.97,0.77
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export DO_NN_TOOLKIT=1
export EXCEPTS=predictions # softmax
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_PREPROCESS=0
fi

if [ $NET = "mobilenet_v1_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/tensorflow/mobilenet_v1
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenet_v1_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.95,0.95,0.6
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.7
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export DO_NN_TOOLKIT=1
export DO_PREPROCESS=0
fi

if [ $NET = "mobilenet_v2_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/tensorflow/mobilenetv2
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenet_v2_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_MULTIPLER=0.93,0.93,0.52
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export EXCEPTS=block_15_project_BN
export DO_PREPROCESS=0
fi

if [ $NET = "vgg16_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/tensorflow/vgg16
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/vgg16_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=103.939,116.779,123.68 # in BGR
export STD=1,1,1
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.90
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export EXCEPTS=block_15_project_BN
export DO_PREPROCESS=0
fi

if [ $NET = "densenet121_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/tensorflow/densenet121/
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/densenet121_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=224,224
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5,127.5,127.5 # in BGR
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_MULTIPLER=0.85,0.85,0.31
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export DO_PREPROCESS=0
fi

if [ $NET = "inception_v3_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/tensorflow/inceptionv3/
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/inceptionv3_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=299,299
export NET_INPUT_DIMS=299,299
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5,127.5,127.5 # in BGR
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_MULTIPLER=0.86,0.83,0.38
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export DO_PREPROCESS=0
fi

# work in progress
if [ $NET = "fcn-8s" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/fcn-8s/caffe/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/fcn-8s/caffe/fcn-8s-pascalcontext.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_fcn-8s_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/fcn-8s_calibration_table
export CALI_TABLE_PREPROCESS=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
export NET_INPUT_DIMS=500,500
export IMAGE_RESIZE_DIMS=500,500
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.92,0.91,0.43
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_LAYERGROUP=1
export DO_NN_TOOLKIT=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data,prob,res2c_relu,res3d_relu,res4f_relu
else
  export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
fi
fi

if [ $NET = "espcn_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export CALI_TABLE=
export IMAGE_RESIZE_DIMS=540,960
export NET_INPUT_DIMS=540,960
export SHAPE_HW=$NET_INPUT_DIMS
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5 # in BGR
export STD=127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_MULTIPLER=0.86,0.83,0.38
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export BGRAY="true"
fi

if [ $NET = "icnet" ]; then
export MODEL_TYPE="caffe"
export MODEL_DEF=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_bnnomerge.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_trainval_90k_bnnomerge.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_caffe.sh
export NET_INPUT_DIMS=1025,2049
export IMAGE_RESIZE_DIMS=1025,2049
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=conv5_3_pool1_interp
export OUTPUTS=conv5_3_pool1_interp
export CALIBRATION_IMAGE_COUNT=30
export DO_CALIBRATION=0
export TOLERANCE_INT8_PER_TENSOR=0.99,0.98,0.91
export TOLERANCE_INT8_RSHIFT_ONLY=0.99,0.98,0.91
export TOLERANCE_INT8_MULTIPLER=0.99,0.98,0.91
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.98,0.97
export DO_CMDBUF_BF16=0
export DO_LAYERGROUP=1
export DO_PREPROCESS=0
export EVAL_MODEL_TYPE="isbi"
export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export IMAGE_PATH=$REGRESSION_PATH/data/0.png
  export EXCEPTS=input
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_0
fi
fi
