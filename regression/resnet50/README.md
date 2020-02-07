# resnet50

resnet-50/101/152, download from onedrive link

- `https://github.com/KaimingHe/deep-residual-networks#models`
- `https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777`

deploy.txt goes here

```
wget -nc https://github.com/KaimingHe/deep-residual-networks/raw/master/prototxt/ResNet-50-deploy.prototxt
wget -nc https://github.com/KaimingHe/deep-residual-networks/raw/master/prototxt/ResNet-101-deploy.prototxt
wget -nc https://github.com/KaimingHe/deep-residual-networks/raw/master/prototxt/ResNet-152-deploy.prototxt
```


## Dataset

- imagenet

## Performance Results

## Accuracy Results

- 20200113

pytorch (50000)
| mode             | Top-1 (%) | Top-5 (%) |
| ---              | ---       | ---       |
| caffe original   | 74.798    | 92.008    |
| fp32             | 74.798    | 92.008    |
| int8 Per-layer   | 74.308    | 91.648    |
| int8 Per-channel | 74.620    | 91.896    |
| int8 Multiplier  | 74.638    | 91.900    |
| fp16             | 74.760    | 91.980    |
