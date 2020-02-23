# mobilenet V2

link

- `https://github.com/shicai/MobileNet-Caffe`

```
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt
```

## Dataset

- imagenet

## Performance Results

## Accuracy Results

- 20200208

  - pytorch (50000)

    ```bash
    | mode             | Top-1 (%) | Top-5 (%) |
    | ---              | ---       | ---       |
    | caffe original   | 71.434    | 90.258    |
    | fp32             | 71.434    | 90.258    |
    | int8 Per-layer   | 42.594    | 68.172    |
    | int8 Per-channel | 69.386    | 88.840    |
    | int8 Multiplier  | 69.276    | 88.858    |
    | fp16             | 71.498    | 90.272    |
    ```
