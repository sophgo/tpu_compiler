# ShuffleNet v2

## trained model
https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases

## shuffle net code
https://github.com/farmingyard/ShuffleNet

## Accuracy Results

- 20200506

  with caffe from master branch, and with a threshold table (1000 image calibration)

  - pytorch (50000)

    ```bash
    | mode             | Top-1 (%) | Top-5 (%) |
    | ---              | ---       | ---       |
    | caffe original   | 55.954    | 79.344    |
    | fp32             | 55.954    | 79.344    |
    | int8 Multiplier  | 53.224    | 77.170    |