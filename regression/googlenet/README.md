# Googlenet

`https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet`

## Accuracy Results

- 20200420

  with caffe from master branch, and with a threshold table (1000 image calibration)

  - pytorch (50000)

    ```bash
    | mode             | Top-1 (%) | Top-5 (%) |
    | ---              | ---       | ---       |
    | caffe original   | 68.072    | 88.652    |
    | fp32             | 68.072    | 88.652    |
    | int8 Multiplier  | 67.682    | 88.450    |
