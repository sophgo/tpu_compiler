# EfficientNet for BVLC Caffe

This is a conversion tool for [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) from TensorFlow to BVLC Caffe

This fork uses only BVLC standard layers, instead of broadcast_Mull and Swish layers.

## TensorFlow to PyTorch Conversion

### Download tf-model
put pretrain tf-model to this folder

```
pretrained_tensorflow/efficientnet-b0
```

### Convert tf-model to pytorch-model

```
mkdir -p pretrained_pytorch/ckptsaug
pushd convert_tf_to_pt
python3 convert_params_tf_pytorch.py \
 --model_name efficientnet-b0 \
 --tf_checkpoint ../pretrained_tensorflow/ckptsaug/efficientnet-b0/ \
 --output_file ../pretrained_pytorch/ckptsaug/efficientnet-b0.pth
popd
```

### Test PyTorch model

```
pushd convert_tf_pt
python3 test_pytorch.py ckptsaug efficientnet-b0
popd
```

## PyTorch to Caffe

### Convert pytorch-model to caffe .prototxt

```
mkdir -p caffemodel/ckptsaug
pushd convert_tf_pt
python3 pytorch2caffe.py ckptsaug efficientnet-b0
popd
```

### Convert pytorch-model to caffe .caffemodel

```
pushd
python3 pytorch2caffe_model.py ckptsaug efficientnet-b0
popd
```

### Test

```
pushd
python3 test_caffe.py ckptsaug efficientnet-b0
popd
```

## data process RGB

* mean = [0.485, 0.456, 0.406] x 255
* std = [0.229, 0.224, 0.225] x 255
