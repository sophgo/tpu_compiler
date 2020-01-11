# vgg-16


VGG_ILSVRC_16_layers.caffemodel, download from below link

- `http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel`


VGG_ILSVRC_16_layers_deploy.prototxt goes here

```
https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt
```


# vgg-16 mlir accuracy test result(ILSVRC2012 50000 val images)
## Eval imagenet with pytorch dataloader

'load module ', 'vgg16.mlir'
 * Acc@1 70.154 Acc@5 89.502

'load module ', 'vgg16_quant_int8_per_layer.mlir'
 * Acc@1 69.688 Acc@5 89.272

'load module ', 'vgg16_quant_int8_multiplier.mlir'
 * Acc@1 69.920 Acc@5 89.342

'load module ', 'vgg16_quant_bf16.mlir'
 * Acc@1 70.180 Acc@5 89.522



## Eval imagenet with gluoncv dataloader

'load module ', 'vgg16.mlir'

 * Top-1 accuracy: 0.6827, Top-5 accuracy: 0.8836

'load module ', 'vgg16_quant_int8_per_layer.mlir'

 * Top-1 accuracy: 0.6788, Top-5 accuracy: 0.8807

'load module ', 'vgg16_quant_int8_multiplier.mlir'
 * Top-1 accuracy: 0.6804, Top-5 accuracy: 0.8821

'load module ', 'vgg16_quant_bf16.mlir'
 * Top-1 accuracy: 0.6826, Top-5 accuracy: 0.8836