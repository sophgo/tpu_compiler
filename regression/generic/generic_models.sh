#!/bin/bash
set -e

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
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
export DO_QUANT_BF16=1
export DO_CMDBUF_BF16=1
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=0
export DO_QUANT_MIX=0
export DO_ACCURACY_CAFFE=1
export DO_ACCURACY_INTERPRETER=1
export DO_E2E=1

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
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/resnet50_calibration_table
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc1000
#export EXCEPTS=prob
export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu   # for "--eltwise-early-stride"
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.92,0.91,0.59
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.95,0.7
export TOLERANCE_INT8_MULTIPLER=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.92
export DO_LAYERGROUP=0
# export BATCH_SIZE=4
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/vgg16_calibration_table
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
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
export DO_CALIBRATION=1
# export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/googlenet_calibration_table
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=104.0,117.0,123.0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS_FP32=prob
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8_PER_TENSOR=0.84,0.84,0.41
export TOLERANCE_INT8_RSHIFT_ONLY=0.94,0.93,0.64
export TOLERANCE_INT8_MULTIPLER=0.95,0.95,0.71
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/inception_v3_calibration_table
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
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/inception_v4_calibration_table
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
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientnet_b0_calibration_table_1000
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
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenet_v2_calibration_table_1000
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS_FP32=prob
export OUTPUTS=fc7
export EXCEPTS=prob
export DO_QUANT_INT8_PER_TENSOR=1
export DO_QUANT_INT8_RFHIFT_ONLY=1
export TOLERANCE_INT8_PER_TENSOR=0.61,0.61,0.02
export TOLERANCE_INT8_RSHIFT_ONLY=0.95,0.94,0.67
export TOLERANCE_INT8_MULTIPLER=0.95,0.94,0.69
export TOLERANCE_BF16=0.99,0.99,0.93
export DO_CMDBUF_BF16=0   # this is a bug to fix
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
#export DO_CALIBRATION=1
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/shufflenet_v2_calibration_table
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

if [ $NET = "squeezenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_PER_TENSOR=0.9,0.9,0.55
export TOLERANCE_INT8_RSHIFT_ONLY=0.9,0.9,0.6
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.6
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "arcface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.59,0.59,-0.12
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
fi

if [ $NET = "bmface_v3" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/bmface_v3_cali1024_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.6
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
fi

if [ $NET = "liveness" ]; then
export MODEL_DEF=$MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.prototxt
export MODEL_DAT=$MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.7
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_E2E=0
fi

if [ $NET = "retinaface_mnet25" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.90,0.85,0.54
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
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
fi

if [ $NET = "ssd300" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy_tpu.prototxt
# export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_PER_TENSOR=0.99,0.99,0.89
export TOLERANCE_INT8_RSHIFT_ONLY=0.98,0.98,0.81
export TOLERANCE_INT8_MULTIPLER=0.99,0.99,0.88
export DO_QUANT_BF16=0
fi

if [ $NET = "yolo_v3_608" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=608,608
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.89,0.86,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.92,0.90,0.60
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=416,416
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "yolo_v3_320" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_caffe/regression_yolo_v3_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export NET_INPUT_DIMS=320,320
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
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
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
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
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "alphapose" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx
export MODEL_DAT=""
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_onnx/regression_alphapose_0_onnx.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/alphapose_calibration_table
export NET_INPUT_DIMS=256,192
export INPUT=input
export TOLERANCE_INT8_PER_TENSOR=0.9,0.88,0.51
export TOLERANCE_INT8_RSHIFT_ONLY=0.92,0.90,0.58
export TOLERANCE_INT8_MULTIPLER=0.93,0.92,0.61
export DO_QUANT_BF16=0
# export TOLERANCE_BF16=0.99,0.99,0.94
# export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

# turn off those optimization when batch_size is larger than 1 temporarily
if [ $BATCH_SIZE -gt 1 ]; then
export DO_DEEPFUSION=0
export DO_MEMOPT=0
export DO_LAYERGROUP=0
export DO_QUANT_MIX=0
export DO_E2E=0
fi
