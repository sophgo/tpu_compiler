
#!/bin/bash
set -e


if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/onnx/resnet50-caffe2-v1-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=gpu_0/data_0
fi

if [ $NET = "squeezenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/onnx/squeezenet1.0-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "vgg19" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/onnx/vgg19-caffe2-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "sub_pixel_cnn_2016" ]; then
export MODEL_DEF=$MODEL_PATH/super_resolution/sub_pixel_cnn_2016/super-resolution-10.onnx
export INPUT_SHAPE=1,1,224,224
export INPUT_NAME=input
fi

if [ $NET = "mobilenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/onnx/mobilenetv2-7.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data
fi

if [ $NET = "densenet-121" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/onnx/densenet-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "caffenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/caffenet/onnx/caffenet-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/onnx/googlenet-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "inception_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v1/onnx/inception-v1-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "inception_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v2/onnx/inception-v2-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=data_0
fi

if [ $NET = "zfnet-512" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/zfnet-512/onnx/zfnet512-9.onnx
export INPUT_SHAPE=1,3,224,224
export INPUT_NAME=gpu_0/data_0
fi