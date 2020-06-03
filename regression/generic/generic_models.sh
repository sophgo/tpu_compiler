#!/bin/bash
set -e

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export STD=1,1,1
export MODEL_CHANNEL_ORDER="bgr"
export DATA_FORMAT="nchw"
export EXCEPTS=-
export DO_CALIBRATION=0
export CALIBRATION_IMAGE_COUNT=1000
export MLIR_OPT_FE_PRE="--convert-bn-to-scale"
export MLIR_OPT_FE_POST="--eltwise-early-stride"
export MLIR_OPT_BE="--tg-fuse-leakyrelu --conv-ic-alignment"
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
export DO_PREPROCESS=0
export USE_LAYERGROUP=0
export EVAL_MODEL_TYPE="imagenet"

if [ -z "$DO_BATCHSIZE" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=$DO_BATCHSIZE
fi
export BATCH_SIZE
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_caffe.sh

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.9
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
export DO_NN_TOOLKIT=1
# export BATCH_SIZE=4
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data,prob,res2c_relu,res3d_relu,res4f_relu
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
  export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
fi
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.91
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.96,0.96,0.72
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.97
export DO_CMDBUF_BF16=0
#export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.69
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
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
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "mobilenet_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet.caffemodel
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
export TOLERANCE_INT8_PER_TENSOR=0.9,0.86,0.52
export TOLERANCE_INT8_RSHIFT_ONLY=0.96,0.95,0.73
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.74
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.94
export DO_CMDBUF_BF16=1
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
  export EXCEPTS=data,prob
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
  export EXCEPTS=prob
fi
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
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
export TOLERANCE_INT8_PER_TENSOR=0.61,0.61,0.01
export TOLERANCE_INT8_RSHIFT_ONLY=0.94,0.9,0.66
export TOLERANCE_INT8_MULTIPLER=0.94,0.94,0.67
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.92
export DO_CMDBUF_BF16=0   # this is a bug to fix
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
  export EXCEPTS=data,prob
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
  export EXCEPTS=prob
fi
fi

if [ $NET = "mobilenet_v3" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v3/onnx/2020.04.17.01/mobilenetv3_rw.onnx
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
export DO_QUANT_MIX_CMD=1
export BF16_QUANT_LAYERS_FILE=${NET}_bf16_quant_layers
export BF16_QUANT_LAYERS="316_Clip 354_Clip 372_Clip 391_Clip 402_Clip 410_Clip 420_Clip 428_Clip 428_Clip 439_Clip 447_Clip 458_Clip 466_Clip 477_Clip 485_Clip 493_Clip 500_Clip 510_Clip 518_Clip 526_Clip 533_Clip 544_Clip 552_Clip 560_Clip 567_Clip 577_Clip 585_Clip 593_Clip 600_Clip 611_Clip 619_Clip 627_Clip 634_Clip 645_Clip 653_Clip #656_Mul 313_BatchNormalization 315_Add 319_Mul 322_Relu 353_Add"
export DO_CMDBUF_BF16=0
export DO_DEEPFUSION=0
export DO_MEMOPT=0
export DO_DEEPFUSION=0
export DO_LAYERGROUP=0
export DO_E2E=0
export USE_LAYERGROUP=0
fi

if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MODEL_CHANNEL_ORDER="rgb"
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
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "squeezenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel
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
export USE_LAYERGROUP=1
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
fi
fi

if [ $NET = "densenet_121" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.83,0.82,0.39
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=0
export USE_LAYERGROUP=0
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
fi
fi

if [ $NET = "densenet_201" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.caffemodel
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
export TOLERANCE_INT8_MULTIPLER=0.78,0.77,0.30
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.92
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.95
export DO_LAYERGROUP=0
export USE_LAYERGROUP=0
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
fi
fi

if [ $NET = "senet_res50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1.caffemodel
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
export TOLERANCE_INT8_RSHIFT_ONLY=0.96,0.96,0.73
export TOLERANCE_INT8_MULTIPLER=0.96,0.96,0.75
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.99
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_LAYERGROUP=0
export USE_LAYERGROUP=0
export DO_PREPROCESS=0
if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000_preprocess
  export EXCEPTS=data
else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
fi
fi

if [ $NET = "arcface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.59,0.59,-0.13
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
# accuracy setting
export EVAL_MODEL_TYPE="lfw"
#export DO_ACCURACY_CAFFE=0
fi

if [ $NET = "retinaface_mnet25_600" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_retinaface_mnet25_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export INPUT=data
export NET_INPUT_DIMS=600,600
export TOLERANCE_INT8_MULTIPLER=0.90,0.85,0.54
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
fi

if [ $NET = "retinaface_mnet25" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_retinaface_mnet25_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export INPUT=data
export NET_INPUT_DIMS=320,320
export TOLERANCE_INT8_MULTIPLER=0.90,0.85,0.54
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
# accuracy setting
export NET_INPUT_DIMS=320,320
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
fi

if [ $NET = "retinaface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.86,0.83,0.49
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
#export DO_ACCURACY_CAFFE=0
#export DO_ACCURACY_ONNX=0
fi

if [ $NET = "ssd300" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/ssd300/labelmap_coco.prototxt
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_ssd_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=104.0,117.0,123.0
export INPUT_SCALE=1.0
export TOLERANCE_INT8_PER_TENSOR=0.99,0.99,0.89
export TOLERANCE_INT8_RSHIFT_ONLY=0.98,0.98,0.81
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.88
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=1
export NET_INPUT_DIMS=300,300
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_CAFFE="eval_caffe_detector_ssd.py"
export EVAL_SCRIPT_INT8="eval_ssd.py"
#export DO_ACCURACY_CAFFE=0
fi

if [ $NET = "mobilenet_ssd" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/labelmap_voc.prototxt
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_ssd_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.007843
export EXCEPTS=detection_out
export TOLERANCE_INT8_PER_TENSOR=0.93,0.87,0.62
export TOLERANCE_INT8_RSHIFT_ONLY=0.97,0.97,0.70
export TOLERANCE_INT8_MULTIPLER=0.98,0.96,0.77
export DO_QUANT_BF16=0
#export DO_LAYERGROUP=1
fi

if [ $NET = "yolo_v2_1080" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy_1080.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v2_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export NET_INPUT_DIMS=1080,1920
export INPUT=data
export TOLERANCE_INT8_PER_TENSOR=0.88,0.88,0.46
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v2_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v2_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export NET_INPUT_DIMS=416,416
export INPUT=data
export TOLERANCE_INT8_PER_TENSOR=0.88,0.88,0.46
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v3_608" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=608,608
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.89,0.86,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.92,0.90,0.60
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=416,416
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v3_320" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=320,320
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v3_160" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_160.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=160,160
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

if [ $NET = "yolo_v3_512x288" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_512x288.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=288,512
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
fi

# check if the MODEL exists of caffe
if [ ! -f ${MODEL_DEF} ]; then
  echo "cannot find the file ${MODEL_DEF}"
  exit 1
fi

if [ ! -f ${MODEL_DAT} ]; then
  echo "cannot find the file ${MODEL_DAT}"
  exit 1
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
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.87
export DO_QUANT_BF16=0
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
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
export USE_LAYERGROUP=1
# export DO_QUANT_INT8_PER_TENSOR=1
# export DO_QUANT_INT8_RFHIFT_ONLY=1
# export TOLERANCE_INT8_PER_TENSOR=0.8,0.8,0.8
# export TOLERANCE_INT8_RSHIFT_ONLY=0.8,0.8,0.8
export TOLERANCE_INT8_MULTIPLER=0.77,0.63,0.28
export TOLERANCE_BF16=0.99,0.99,0.91
export DO_CMDBUF_BF16=0
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
# export DO_QUANT_MIX=1
# export TOLERANCE_MIX=0.8,0.8,0.8
export DO_ACCURACY_CAFFE=0
export DO_ACCURACY_ONNX=1
export DO_LAYERGROUP=1
export USE_LAYERGROUP=0
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
export USE_LAYERGROUP=1
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

# check if the MODEL exists of onnx case
if [ ! -f ${MODEL_DEF} ]; then
  echo "cannot find the file ${MODEL_DEF}"
  exit 1
fi

if [ $NET = "resnet50_tensorflow" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/tensorflow/resnet50
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tensorflow.sh
export MLIR_OPT_FE_POST=""
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/resnet50_tensorflow_calibration_table
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export DATA_FORMAT="nhwc"
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER="0.42,0.3,-0.4"
export DO_QUANT_BF16=0
export DO_E2E=0
export DO_DEEPFUSION=0
export DO_NN_TOOLKIT=1
export EXCEPTS=probs # softmax
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

# turn off those optimization when batch_size is larger than 1 temporarily
#if [ $BATCH_SIZE -gt 1 ]; then
#export DO_DEEPFUSION=0
#export DO_MEMOPT=0
#export DO_QUANT_MIX=0
#export DO_E2E=0
#fi

# if [ $DO_LAYERGROUP -eq 1 ]; then
#   # echo "do layer_group, fuse leaky relu"
#   export MLIR_OPT_BE="--tg-fuse-leakyrelu  --conv-ic-alignment"
# fi
