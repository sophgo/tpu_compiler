# retinaface_mnet25
- This is retinaface using backbone of mobilenet 0.25
- Please refer to $REGRESSION_PATH/retinaface_res50/README.md

## Performance Results
- cv1835 (DDR3)
    - retinaface mobilenet 320x320  1.69 ms, 591.03 fps

## Accuracy Results
- widerface 的 easy/medium/hard 分別如下
- FP32
    - retinaface_mobilenet 600x600   0.817/0.757/0.479
    - retinaface_mobilenet 320x320   0.709/0.585/0.257

- INT8
    - retinaface_mobilenet 320x320  MLIR   kld   0.480/0.303/0.127
    - retinaface_mobilenet 320x320  bmtap2 tune  0.677/0.553/0.242
