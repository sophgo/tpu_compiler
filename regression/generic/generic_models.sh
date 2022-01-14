#!/bin/bash
set -e

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export STD=1,1,1
export MODEL_CHANNEL_ORDER="bgr"
export EXCEPTS=-
export EXCEPTS_BF16=-
export CALIBRATION_IMAGE_COUNT=1000
export DO_QUANT_INT8=1
export DO_FUSED_PREPROCESS=1
export DO_FUSED_POSTPROCESS=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_ACCURACY_FUSED_PREPROCESS=0
export EVAL_MODEL_TYPE="imagenet"
export LABEL_FILE=$REGRESSION_PATH/data/synset_words.txt
export BGRAY=0
export RESIZE_KEEP_ASPECT_RATIO=0
export TOLERANCE_FP32=0.999,0.999,0.998
export DO_QUANT_BF16=1
export INT8_MODEL=0
export MIX_PRECISION_TABLE='-'
export MODEL_DAT="-"

if [ -z "$DO_BATCHSIZE" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=$DO_BATCHSIZE
fi
export BATCH_SIZE

# default inference and test image, for imagenet models only
export IMAGE_PATH=$REGRESSION_PATH/data/cat.jpg

# do postprocess
export DO_POSTPROCESS=0
export POSTPROCESS_SCRIPT=-

# Caffe
if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=fc1000
export TOLERANCE_INT8=0.96,0.95,0.71
export TOLERANCE_MIX_PRECISION=0.96,0.95,0.73
export MIX_PRECISION_BF16_LAYER_NUM=10
export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
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
export OUTPUTS=prob
export TOLERANCE_INT8=0.96,0.95,0.70
export TOLERANCE_BF16=0.99,0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.97
fi

if [ $NET = "resnet_res_blstm" ]; then
export MODEL_DEF=$MODEL_PATH/rnn/resnet_res_blstm/caffe/deploy_fix.prototxt
export MODEL_DAT=$MODEL_PATH/rnn/resnet_res_blstm/caffe/model.caffemodel
export LABEL_MAP=$MODEL_PATH/rnn/resnet_res_blstm/caffe/label.txt
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/poem.jpg
export NET_INPUT_DIMS=32,280
export IMAGE_RESIZE_DIMS=32,280
export RAW_SCALE=255.0
export MEAN=152,152,152
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=fc1x
export TOLERANCE_INT8=0.99,0.99,0.79
export TOLERANCE_BF16=0.99,0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_POSTPROCESS=1
export POSTPROCESS_SCRIPT=$REGRESSION_PATH/data/run_postprocess/ctc_greedy_decoder.sh
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=fc8
export EXCEPTS=prob
export TOLERANCE_INT8=0.99,0.99,0.90
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS=prob
export TOLERANCE_INT8=0.93,0.93,0.62
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.95,0.95,0.68
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.93,0.93,0.63
export TOLERANCE_BF16=0.99,0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "icnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_bnnomerge.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/ICNet/caffe/icnet_cityscapes_trainval_90k_bnnomerge.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/0.png
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_0
export NET_INPUT_DIMS=1025,2049
export IMAGE_RESIZE_DIMS=1025,2049
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS=conv5_3_pool1_interp
export CALIBRATION_IMAGE_COUNT=30
export TOLERANCE_INT8=0.85,0.84,0.41
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EVAL_MODEL_TYPE="isbi"
fi

if [ $NET = "mobilenet_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.96,0.95,0.73
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.95
export EXCEPTS=prob
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.94,0.94,0.66
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EXCEPTS=prob
fi

if [ $NET = "blazeface" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/blazeface/onnx/blazeface.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export NET_INPUT_DIMS=128,128 # h,w
export IMAGE_RESIZE_DIMS=128,128
export CALIBRATION_IMAGE_COUNT=1
export MEAN=1,1,1
export INPUT_SCALE=1.0
export STD=127.5,127.5,127.5
export RAW_SCALE=255
export INPUT=input
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export TOLERANCE_INT8=0.70,0.70,-0.10
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.96 #
export DO_PREPROCESS=0
export BGRAY=0
# just compare last one
fi

if [ $NET = "blazefacenas" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/blazefacenas/onnx/blazefacenas.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export MIX_PRECISION_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_mix_table
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export NET_INPUT_DIMS=640,640 # h,w
export IMAGE_RESIZE_DIMS=640,640
export CALIBRATION_IMAGE_COUNT=1
export RAW_SCALE=255.0
export INPUT_SCALE=1.0
export MEAN=123,117,104
export STD=127.502231,127.502231,127.502231
export INPUT=input
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export TOLERANCE_INT8=0.70,0.70,-0.10
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.96 #
export EXCEPTS=transpose_0.tmp_0_Transpose,reshape2_3.tmp_0_Reshape,concat_0.tmp_0_Concat,reshape2_13.tmp_0_Reshape,reshape2_13.tmp_0_Reshape_dequant
export DO_PREPROCESS=0
export BGRAY=0
# just compare last one
fi


if [ $NET = "mobilenet_v3" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v3/onnx/mobilenetv3_rw.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenetv3_calibration_table
export RAW_SCALE=1
export INPUT_SCALE=0.875
export NET_INPUT_DIMS=256,256
export MEAN=0.406,0.456,0.485  # in BGR, pytorch mean=[0.485, 0.456, 0.406]
export STD=0.225,0.224,0.229   # in BGR, pytorch std=[0.229, 0.224, 0.225]
export IMAGE_RESIZE_DIMS=224,224
export INPUT=input
export TOLERANCE_INT8=0.083338,-0.1,-1.0
export DO_QUANT_INT8=0
export DO_CMDBUF_BF16=1
export TOLERANCE_BF16=0.9,0.9,0.9
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export EXCEPTS=371_Add,315_Add,390_Add,401_Add,409_Add,419_Add,427_Add,438_Add,446_Add,457_Add,465_Add,476_Add,484_Add,492_Add,499_Add,509_Add,517_Add,525_Add,532_Add,543_Add,551_Add,559_Add,566_Add,576_Add,584_Add,592_Add,599_Add,610_Add,618_Add,626_Add,633_Add,644_Add,652_Add # cuz relu6 could add 'relu' layer that could mismatch original layer
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "nasnet_mobile" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/nasnet_mobile/onnx/nasnet_mobile.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/nasnet_mobile_calibration_table
export RAW_SCALE=1
export INPUT_SCALE=1
export NET_INPUT_DIMS=224,224
export MEAN=0.5,0.5,0.5  # in BGR, pytorch mean=[0.5, 0.5, 0.5]
export STD=0.5,0.5,0.5   # in BGR, pytorch std=[0.5, 0.5, 0.5]
export IMAGE_RESIZE_DIMS=256,256
export INPUT=input
export TOLERANCE_INT8=0.77,0.77,0.277
export CALIBRATION_IMAGE_COUNT=2000
export DO_CMDBUF_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "faceboxes" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/face_detection/faceboxes/onnx/faceboxes.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_onnx.sh
export NET_INPUT_DIMS=915,1347
export IMAGE_RESIZE_DIMS=915,1347
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input.1
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export TOLERANCE_INT8=0.70,0.70,-0.10
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.96 #
export DO_PREPROCESS=0
export BGRAY=0
# just compare last one
fi


if [ $NET = "shufflenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export MODEL_CHANNEL_ORDER="rgb"
export RAW_SCALE=1.0
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS=fc
export TOLERANCE_INT8=0.92,0.92,0.57
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "shufflenet_v1" ]; then
# just for test concat relu
export MODEL_DEF=$MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc1000
export TOLERANCE_INT8=0.96,0.96,0.72
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "squeezenet_v1.0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.0.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS=pool10
export TOLERANCE_INT8=0.9,0.9,0.42
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "squeezenet_v1.1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0
export INPUT=data
export OUTPUTS=pool10
export TOLERANCE_INT8=0.9,0.9,0.55
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
fi

if [ $NET = "densenet_121" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_121.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc6
export TOLERANCE_INT8=0.82,0.81,0.37
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "densenet_201" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc6
export TOLERANCE_INT8=0.77,0.76,0.28
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "senet_res50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/senet/caffe/se_resnet_50_v1.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export NET_INPUT_DIMS=225,225
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017
export INPUT=input
export OUTPUTS=fc6
export TOLERANCE_INT8=0.96,0.96,0.73
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "arcface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/Aaron_Eckhart_0001.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=112,112
export IMAGE_RESIZE_DIMS=112,112
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.0078125
export TOLERANCE_INT8=0.45,0.45,-0.39
# accuracy setting
export EVAL_MODEL_TYPE="lfw"
export EXCEPTS=data
export TOLERANCE_BF16=0.99,0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "retinaface_mnet25_600" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8=0.90,0.85,0.54
export TOLERANCE_BF16=0.99,0.99,0.88
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export EXCEPTS=data
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600_with_detection.prototxt
fi

if [ $NET = "retinaface_mnet25" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=320,320
export NET_INPUT_DIMS=320,320
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8=0.90,0.85,0.54
# accuracy setting
export NET_INPUT_DIMS=320,320
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.92
fi

if [ $NET = "retinaface_res50" ]; then
export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt
export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/parade.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=600,600
export NET_INPUT_DIMS=600,600
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1
export TOLERANCE_INT8=0.86,0.83,0.49
# accuracy setting
export NET_INPUT_DIMS=600,600
export EVAL_MODEL_TYPE="widerface"
export OBJ_THRESHOLD=0.005
export NMS_THRESHOLD=0.45
export DATASET=$DATASET_PATH/widerface/WIDER_val/images
export ANNOTATION=$DATASET_PATH/widerface/wider_face_split
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.99,0.87
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "ssd300" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/ssd300/labelmap_coco.prototxt
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=104.0,117.0,123.0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.91,0.91,0.52
export TOLERANCE_BF16=0.99,0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.91
export EXCEPTS=detection_out
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_CAFFE="eval_caffe_detector_ssd.py"
export EVAL_SCRIPT_INT8="eval_ssd.py"
fi

if [ $NET = "mobilenet_ssd" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/MobileNetSSD_deploy.caffemodel
export LABEL_MAP=$MODEL_PATH/object_detection/ssd/caffe/mobilenet_ssd/labelmap_voc.prototxt
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=data
export IMAGE_RESIZE_DIMS=300,300
export NET_INPUT_DIMS=300,300
export RAW_SCALE=255.0
export MEAN=127.5,127.5,127.5
export INPUT_SCALE=0.007843
export TOLERANCE_INT8=0.96,0.96,0.67
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export EXCEPTS=detection_out
export EVAL_MODEL_TYPE="voc2012"
export EVAL_SCRIPT_VOC="eval_detector_voc.py"
fi

if [ $NET = "yolact" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/segmentation/yolact/onnx/yolact_resnet50_coco_4outputs.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export NET_INPUT_DIMS=550,550
export IMAGE_RESIZE_DIMS=550,550
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=input
export DO_QUANT_BF16=0
export TOLERANCE_INT8=0.82,0.79,0.29
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
fi

if [ $NET = "yolo_v1_448" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v1/caffe/yolo_tiny.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v1_448_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=448,448
export NET_INPUT_DIMS=448,448
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "yolo_v2_1080" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy_1080.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=1080,1920
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=1080,1920
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.90,0.90,0.50
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export DO_QUANT_BF16=0
fi

if [ $NET = "yolo_v2_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v2/caffe/caffe_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v2/caffe/yolov2.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v2_calibration_table
export INPUT=data
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.89,0.89,0.48
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.99
fi

if [ $NET = "yolo_v3_608" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/608/yolov3_608.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.92,0.90,0.59
export TOLERANCE_BF16=0.99,0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_416" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.92,0.90,0.59
export EXCEPTS='output'
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416_with_detection.prototxt
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_416_onnx" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/onnx/yolov3-416.onnx
export MODEL_DAT=""
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_416_onnx_threshold_table
export NET_INPUT_DIMS=416,416
export INPUT=input
export TOLERANCE_INT8=0.82,0.79,0.29
export TOLERANCE_BF16=0.99,0.99,0.97
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_320" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=320,320
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=320,320
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.93,0.92,0.60
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export DO_FUSED_POSTPROCESS=1
export MODEL_DEF_FUSED_POSTPROCESS=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320_with_detection.prototxt
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_160" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_160.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=160,160
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=160,160
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.93,0.92,0.59
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_512x288" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_512x288.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=288,512
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=288,512
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.93,0.92,0.61
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
fi

if [ $NET = "yolo_v3_tiny" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_tiny.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_tiny_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export EXCEPTS=output
export TOLERANCE_INT8=0.93,0.93,0.54
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
export TINY=1
fi

if [ $NET = "yolo_v3_spp" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_spp.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_spp_calibration_table
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export EXCEPTS=output
export TOLERANCE_INT8=0.86,0.86,0.32
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
export SPP_NET=1
fi

if [ $NET = "faster_rcnn" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/frcn/caffe/faster_rcnn.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/frcn/caffe/faster_rcnn.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_frcn.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/faster_rcnn_calibration_table
export INPUT=input
export IMAGE_RESIZE_DIMS=600,800
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=600,800
export RAW_SCALE=255.0
export MEAN=102.9801,115.9465,122.7717
export INPUT_SCALE=1.0
export EXCEPTS=proposal,roi_pool5,bbox_pred,output
export TOLERANCE_INT8=0.84,0.78,0.41
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EXCEPTS_BF16=proposal,roi_pool5,roi_pool5_quant,fc6_reshape,relu6,relu7,cls_score,cls_score_dequant,bbox_pred,bbox_pred_dequant,cls_prob #output is euclidean_similarity   = 0.995603
fi

if [ $NET = "yolo_v4" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v4_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.88,0.88,0.46
# mish layer
export EXCEPTS="layer137-act,layer138-act,layer138-scale,layer142-act,layer149-act,layer149-scale"
export OUTPUTS="layer139-conv,layer150-conv,layer161-conv"
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export YOLO_V4=1
fi

if [ $NET = "yolo_v4_with_detection" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4_with_detection.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v4_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=608,608
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=608,608
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.65,0.61,0.01
# mish layer
export EXCEPTS="layer136-act,layer137-act,layer138-act,layer142-act,layer148-act,layer149-act,layer153-act,output"
export OUTPUTS="layer139-conv,layer150-conv,layer161-conv"
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export YOLO_V4=1
fi


if [ $NET = "yolo_v4_540x960" ]; then
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/caffe/540x960/yolov4.prototxt
export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v4/caffe/yolov4.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v4_calibration_table_autotune
export YOLO_V4="true"
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=540,960
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=540,960
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.65,0.61,0.01
# mish layer
export EXCEPTS="layer136-act,layer137-act,layer138-act,layer142-act,layer148-act,layer149-act,layer153-act"
export OUTPUTS="layer139-conv,layer150-conv,layer161-conv"
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export YOLO_V4=1
fi


if [ $NET = "yolo_v4_tiny" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v4/onnx/yolov4_1_3_416_416_static.onnx
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export EVAL_SCRIPT=$REGRESSION_PATH/data/eval/accuracy_yolo_v3.sh
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v4_tiny_calibration_table_autotune
export INPUT=input
export MODEL_CHANNEL_ORDER="rgb"
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export OUTPUTS="layer30-conv,layer37-conv"
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export TOLERANCE_INT8=0.65,0.1,0.005
#export EXCEPTS="layer6-act,layer7-route,layer9-route,layer10-maxpool,layer11-act,layer17-route,layer18-maxpool,layer27-act"
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V4=1
export TINY=1
if [ $DO_PREPROCESS -eq 1 ]; then
  export EXCEPTS=data
fi
fi

# ONNX

if [ $NET = "resnet18" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export TOLERANCE_INT8=0.99,0.98,0.86
export TOLERANCE_BF16=0.99,0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96

fi

if [ $NET = "resnetv2" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnetv2_tf_50_10.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export EXCEPTS=resnet_v2_50/predictions/Softmax:0_Softmax
export TOLERANCE_INT8=0.83,0.83,0.40
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "ghostnet" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/ghostnet/onnx/ghostnet_pytorch.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export TOLERANCE_INT8=0.89,0.88,0.40
export TOLERANCE_BF16=0.99,0.99,0.94
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "efficientnet_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientnet_b0_onnx_calibration_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB,
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export EXCEPTS=424_Mul,422_Conv,388_Sigmoid
export TOLERANCE_INT8=0.68,0.43,0.13
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "efficientdet_d0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/efficientdet-d0/onnx/efficientdet-d0.onnx
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientdet_d0_onnx_calibration_table_1000
export IMAGE_RESIZE_DIMS=512,512
export NET_INPUT_DIMS=512,512
export RAW_SCALE=1.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.485,0.456,0.406  # in RGB,
export STD=0.229,0.224,0.225   # in RGB
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export DO_QUANT_BF16=0
export EXCEPTS=2367_Mul,2365_Conv,2366_Sigmoid
export TOLERANCE_INT8=0.78,0.77,0.19
fi

if [ $NET = "ocr" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/speech/ocr/onnx/ocr_no_argmax.onnx
export MODEL_DAT=""
export IMAGE_PATH=$REGRESSION_PATH/data/ocr.jpg
export INPUT=input
export OUTPUTS=output
export DO_QUANT_INT8=0
export TOLERANCE_BF16=0.89,0.87,0.51
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export IMAGE_RESIZE_DIMS=32,1400
export NET_INPUT_DIMS=32,1400
export RAW_SCALE=1.0
export MEAN=0.0,0.0,0.0  # in RGB,
export INPUT_SCALE=1.0
export BGRAY=1
fi

if [ $NET = "efficientnet-lite_b0" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/efficientnet-lite/b0/onnx/efficientnet_lite.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/efficientnet-lite-b0_onnx_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127,127,127
export STD=128,128,128
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export TOLERANCE_INT8=0.95,0.95,0.7
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi


if [ $NET = "fcos" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/fcos/onnx/fcos.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/fcos_calibration_table
export IMAGE_RESIZE_DIMS=800,1216
export NET_INPUT_DIMS=800,1216
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127,127,127
export STD=128,128,128
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output
export TOLERANCE_INT8=0.56,0.39,-0.22
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "alphapose" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx
export IMAGE_PATH=$REGRESSION_PATH/data/pose_256_192.jpg
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
export TOLERANCE_INT8=0.91,0.89,0.49
export TOLERANCE_BF16=0.99,0.99,0.91
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "espcn_3x" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/super_resolution/espcn/onnx/espcn_3x.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/espcn_3x_calibration_table
export IMAGE_RESIZE_DIMS=85,85
export NET_INPUT_DIMS=85,85
export MEAN=0,0,0
export STD=1.0,1.0,1.0
export RAW_SCALE=1
export MODEL_CHANNEL_ORDER="rgb"
export INPUT_SCALE=1.0
export INPUT=input
# cannot do fuse preprocessing, bcuz this model use
# PIL.Image to do resize in ANTIALIAS mode,
# which is not support in opencv
export DO_FUSED_PREPROCESS=0
export TOLERANCE_INT8=0.98,0.97,0.80
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
fi

if [ $NET = "unet" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/segmentation/unet/onnx/unet.onnx
export IMAGE_PATH=$REGRESSION_PATH/data/0.png
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table_0
export NET_INPUT_DIMS=256,256
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=prob
export CALIBRATION_IMAGE_COUNT=30
export TOLERANCE_INT8=0.99,0.98,0.91
export TOLERANCE_BF16=0.99,0.99,0.98
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export EVAL_MODEL_TYPE="isbi"
export BGRAY=1
fi

if [ $NET = "ecanet50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/ecanet/onnx/ecanet50.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export STD=0.229,0.224,0.225
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output_Gemm
export TOLERANCE_INT8=0.96,0.96,0.71
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "res2net50" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/imagenet/res2net/onnx/res2net50_48w_2s.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export MODEL_CHANNEL_ORDER="rgb"
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=1.0
export MEAN=0.485,0.456,0.406
export STD=0.229,0.224,0.225
export INPUT_SCALE=1.0
export INPUT=input
export OUTPUTS=output_Gemm
export TOLERANCE_INT8=0.93,0.93,0.63
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
# export BATCH_SIZE=4
fi

if [ $NET = "segnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/segnet/caffe/segnet_model_driving_webdemo_fix.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/segnet/caffe/segnet_weights_driving_webdemo.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/camvid.png
export COLOURS_LUT=$REGRESSION_PATH/data/camvid12_lut.png
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export INPUT=input
export OUTPUTS=conv1_1_D
export IMAGE_RESIZE_DIMS=360,480
export NET_INPUT_DIMS=360,480
export MEAN=0,0,0
export STD=1.0,1.0,1.0
export RAW_SCALE=255.0
export INPUT_SCALE=1.0
export TOLERANCE_FP32=0.999,0.999,0.977
export TOLERANCE_INT8=0.91,0.90,0.57
export EXCEPTS=upsample2,upsample1,pool1_mask,pool2_mask
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.98,0.87
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "erfnet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_deploy_mergebn.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/erfnet/caffe/erfnet_cityscapes_mergebn.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/erfnet_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/city.png
export COLOURS_LUT=$REGRESSION_PATH/data/city_lut.png
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export CALIBRATION_IMAGE_COUNT=60
export INPUT=data
export OUTPUTS=Deconvolution23_deconv
export TOLERANCE_INT8=0.78,0.77,0.25
export TOLERANCE_BF16=0.99,0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.91
export EXCEPTS=NBD19_add_conv1_3x1/relu,NBD19_add_conv1_1x3/relu,NBD19_add_conv2_3x1/relu,NBD19_add_conv2_1x3
# export BATCH_SIZE=4
fi

if [ $NET = "enet" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/enet/caffe/enet_deploy_final.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/enet/caffe/cityscapes_weights.caffemodel
export IMAGE_PATH=$REGRESSION_PATH/data/city.png
export COLOURS_LUT=$REGRESSION_PATH/data/city_lut.png
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/enet_calibration_table
export NET_INPUT_DIMS=512,1024
export IMAGE_RESIZE_DIMS=512,1024
export RAW_SCALE=255.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export CALIBRATION_IMAGE_COUNT=60
export INPUT=data
export OUTPUTS=deconv6_0_0
export TOLERANCE_INT8=0.69,0.66,0.11
export TOLERANCE_BF16=0.96,0.96,0.74
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
export EXCEPTS=pool1_0_4_mask,pool2_0_4_mask,conv2_7_1_a,prelu2_7_0,prelu2_7_1,prelu3_3_0,conv3_3_1_a,prelu3_3_1,prelu4_0_4,upsample4_0_4,upsample5_0_4
# export BATCH_SIZE=4
fi

if [ $NET = "gaitset" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/pose/gaitset/onnx/gaitset.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/002-bg-02-018-124.png
export IMAGE_RESIZE_DIMS=64,64
export NET_INPUT_DIMS=64,64
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export RAW_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.70,0.69,-0.175
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export BGRAY=1
fi

if [ $NET = "bisenetv2" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/segmentation/bisenetv2/onnx/bisenetv2.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/munich_000299_000019_leftImg8bit.png
# https://github.com/CoinCheung/BiSeNet/blob/master/tools/demo.py
export NET_INPUT_DIMS=1024,2048
export IMAGE_RESIZE_DIMS=1024,2048
export CALIBRATION_IMAGE_COUNT=30
export MEAN=0.3257,0.3690,0.3223
export INPUT_SCALE=1.0
export STD=0.2112,0.2148,0.2115
export RAW_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.70,0.70,-0.10
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_FP32=0.999,0.999,0.993
export DO_PREPROCESS=0
export BGRAY=0
export EXCEPTS=368_Relu,374_Relu,383_Relu,392_Relu,395_Relu,398_Relu,399_MaxPool,400_Concat,408_BatchNormalization,413_BatchNormalization,415_BatchNormalization,419_Relu,425_Relu,427_BatchNormalization,429_Relu,434_BatchNormalization,439_BatchNormalization,441_BatchNormalization,445_Relu,451_Relu,455_Relu,453_BatchNormalization,460_BatchNormalization,465_BatchNormalization,467_BatchNormalization,471_Relu,477_Relu,479_BatchNormalization,481_Relu,487_Relu,489_BatchNormalization,491_Relu,497_Relu,502_ReduceMean,501_Relu,499_BatchNormalization,503_BatchNormalization,518_BatchNormalization,507_Add,510_Relu,519_Conv,521_BatchNormalization,526_BatchNormalization,524_AveragePool,543_Sigmoid,546_Mul,545_Sigmoid,544_Mul,560_Add,566_Relu,567_Conv,583_Relu,584_Conv,600_Relu,601_Conv,617_Relu,618_Conv,634_Relu,635_Conv,371_Relu,377_Relu,380_Relu,386_Relu,389_Relu,403_Relu,406_Relu,411_Relu,417_BatchNormalization,422_Relu,432_Relu,437_Relu,443_BatchNormalization,448_Relu,458_Relu,463_Relu,469_BatchNormalization,474_Relu,484_Relu,494_Relu,506_Relu,528_BatchNormalization,529_Conv,563_Relu
# just compare last one
fi

# TensorFlow
if [ $NET = "resnet50_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/tensorflow/resnet50
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/resnet50_tensorflow_calibration_table_1000
export IMAGE_RESIZE_DIMS=224,224
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=104.01,116.67,122.68 # in BGR
export STD=1,1,1
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.97,0.97,0.76
export EXCEPTS=StatefulPartitionedCall/resnet50/predictions/Softmax # softmax
export TOLERANCE_BF16=0.99,0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.94
fi

if [ $NET = "mobilenet_v1_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/tensorflow/mobilenet_v1
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenet_v1_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.95,0.95,0.7
export TOLERANCE_BF16=0.99,0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.95
fi

if [ $NET = "mobilenet_v2_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/tensorflow/mobilenetv2
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/mobilenet_v2_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.93,0.93,0.52
export TOLERANCE_BF16=0.98,0.98,0.84
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EXCEPTS=block_15_project_BN
fi

if [ $NET = "vgg16_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/tensorflow/vgg16
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/vgg16_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=256,256
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=103.939,116.779,123.68 # in BGR
export STD=1,1,1
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.98,0.98,0.84
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "densenet121_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/tensorflow/densenet121/
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/densenet121_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=224,224
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5,127.5,127.5 # in BGR
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.85,0.85,0.31
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "inception_v3_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/tensorflow/inceptionv3/
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/inceptionv3_tf_threshold_table_1000
export IMAGE_RESIZE_DIMS=299,299
export NET_INPUT_DIMS=299,299
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5,127.5,127.5 # in BGR
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.85,0.82,0.37
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

# work in progress
if [ $NET = "fcn-8s" ]; then
export MODEL_DEF=$MODEL_PATH/segmentation/fcn-8s/caffe/deploy.prototxt
export MODEL_DAT=$MODEL_PATH/segmentation/fcn-8s/caffe/fcn-8s-pascalcontext.caffemodel
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/fcn-8s_calibration_table
export NET_INPUT_DIMS=500,500
export IMAGE_RESIZE_DIMS=500,500
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.92,0.92,0.44
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "espcn_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=
export CALI_TABLE=
export IMAGE_RESIZE_DIMS=540,960
export NET_INPUT_DIMS=540,960
export SHAPE_HW=$NET_INPUT_DIMS
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=127.5 # in BGR
export STD=127.5
export INPUT_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.86,0.83,0.38
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export BGRAY="true"
fi


if [ $NET = "yolo_v3_416_tf" ]; then
export MODEL_TYPE="tensorflow"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/tensorflow/yolo_v3_416_without_detection
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_416_tf_calibration_table_1000
export IMAGE_RESIZE_DIMS=416,416
export RESIZE_KEEP_ASPECT_RATIO=1
export NET_INPUT_DIMS=416,416
export MODEL_CHANNEL_ORDER="rgb"
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=input_1
export TOLERANCE_INT8=0.8,0.78,0.26
export DO_QUANT_BF16=0
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export YOLO_V3=1
fi

if [ $NET = "yolo_v5" ]; then
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v5/onnx/yolov5s.onnx
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/${NET}_calibration_table
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export NET_INPUT_DIMS=640,640 # h,w
export IMAGE_RESIZE_DIMS=640,640
export CALIBRATION_IMAGE_COUNT=1000
export MEAN=0,0,0
export INPUT_SCALE=1.0
export STD=1,1,1
export RAW_SCALE=1.0
export INPUT=input
export TOLERANCE_INT8=0.90,0.90,0.54
export DO_QUANT_BF16=1
export TOLERANCE_BF16=0.99,0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.98
export TOLERANCE_FP32=0.99,0.99,0.99 #
export DO_PREPROCESS=0
export BGRAY=0
# just compare last one
# export YOLO_V5=1 # not support cal accuracy now
fi

if [ $NET = "yolox_s" ]; then
# onnx: IoU 0.5:0.95 0.363, IoU 0.50 0.541, IoU 0.75 0.389
# int8: IoU 0.5:0.95 0.344, IoU 0.50 0.515, IoU 0.75 0.373
export MODEL_TYPE="onnx"
export MODEL_DEF=$MODEL_PATH/object_detection/yolox/onnx/yolox_s.onnx
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolox_s_calib.txt
export INPUT=input
export MODEL_CHANNEL_ORDER="bgr"
export IMAGE_RESIZE_DIMS=640,640
export NET_INPUT_DIMS=640,640
export RAW_SCALE=255.0
export MEAN=0.,0.,0.
export STD=1.,1.,1.
export INPUT_SCALE=1.0
export EXCEPTS="796_Sigmoid" # 0.873364, 0.873364, 0.347177
export TOLERANCE_BF16=0.98,0.98,0.82
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export TOLERANCE_INT8=0.87,0.87,0.6
# accuracy setting
export EVAL_MODEL_TYPE="coco"
export EVAL_SCRIPT_ONNX="eval_yolox.py"
export EVAL_SCRIPT_INT8="eval_yolox.py"
fi

# TFLite


# TFLite Int8
if [ $NET = "resnet50_tflite_int8" ]; then
export INT8_MODEL=1
export MODEL_TYPE="tflite_int8"
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/tflite_int8/resnet50_quant_int8.tflite
export INT8_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tflite.sh
export IMAGE_RESIZE_DIMS=224,224
export NET_INPUT_DIMS=224,224
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="bgr"
export MEAN=103.939,116.779,123.68 # in BGR
export STD=1,1,1
export INPUT_SCALE=1.0
export INPUT=input
export DO_QUANT_INT8=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_QUANT_BF16=0
fi

if [ $NET = "inception_v3_tflite_int8" ]; then
export INT8_MODEL=1
export MODEL_TYPE="tflite_int8"
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/tflite_int8/inception_v3_int8_quant.tflite
export INT8_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_tflite.sh
export IMAGE_RESIZE_DIMS=299,299
export NET_INPUT_DIMS=299,299
export RAW_SCALE=255
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=127.5,127.5,127.5 # in RGB
export STD=127.5,127.5,127.5
export INPUT_SCALE=1.0
export INPUT=input
export DO_QUANT_INT8=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_QUANT_BF16=0
fi

if [ $NET = "yolo_v3_416_without_detection_tflite_int8" ]; then
export INT8_MODEL=1
export MODEL_TYPE="tflite_int8"
export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/tflite_int8/yolo_v3_416_without_detection_int8_quant.tflite
export IMAGE_PATH=$REGRESSION_PATH/data/dog.jpg
export INT8_INFERENCE_SCRIPT=$REGRESSION_PATH/data/run_tflite_int8/regression_yolo_v3_0_tflite_int8.sh
export IMAGE_RESIZE_DIMS=416,416
export NET_INPUT_DIMS=416,416
export RAW_SCALE=1.0
export MEAN=0,0,0
export INPUT_SCALE=1.0
export INPUT=input_1
export DO_QUANT_INT8=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_QUANT_BF16=0
export YOLO_V3=1
fi
