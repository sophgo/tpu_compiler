# vgg-16

## Model

link

- VGG_ILSVRC_16_layers.caffemodel, download from below link

  `http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel`

- VGG_ILSVRC_16_layers_deploy.prototxt goes here

  `https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt`

## Dataset

- imagenet

## Performance Results

## Accuracy Results

- 20200213

  - pytorch (50000)

  | mode             | Top-1 (%) | Top-5 (%) |
  | ---              | ---       | ---       |
  | caffe original   | 70.262    | 89.518    |
  | fp32             | 70.262    | 89.518    |
  | int8 Per-layer   | 69.788    | 89.270    |
  | int8 Per-channel | 70.170    | 89.438    |
  | int8 Multiplier  | 69.956    | 89.336    |
  | fp16             | 70.270    | 89.496    |

- 20191223

  - Eval imagenet with pytorch dataloader

  ```bash
  'load module ', 'vgg16.mlir'
   * Acc@1 70.154 Acc@5 89.502

  'load module ', 'vgg16_quant_int8_per_layer.mlir'
   * Acc@1 69.688 Acc@5 89.272

  'load module ', 'vgg16_quant_int8_multiplier.mlir'
   * Acc@1 69.920 Acc@5 89.342

  'load module ', 'vgg16_quant_bf16.mlir'
   * Acc@1 70.180 Acc@5 89.522
  ```

  - Eval imagenet with gluoncv dataloader

  ```bash
  'load module ', 'vgg16.mlir'

   * Top-1 accuracy: 0.6827, Top-5 accuracy: 0.8836

  'load module ', 'vgg16_quant_int8_per_layer.mlir'

   * Top-1 accuracy: 0.6788, Top-5 accuracy: 0.8807

  'load module ', 'vgg16_quant_int8_multiplier.mlir'
   * Top-1 accuracy: 0.6804, Top-5 accuracy: 0.8821

  'load module ', 'vgg16_quant_bf16.mlir'
   * Top-1 accuracy: 0.6826, Top-5 accuracy: 0.8836
  ```
