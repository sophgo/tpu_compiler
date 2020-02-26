# Retinaface-mobilenet

## Eval Dataset
- widerface

## Performance Results
- cv1835 (DDR3)
    - retinaface mobilenet 320x320  1.69 ms, 591.03 fps
    - retinaface resnet50 320x320   35.19 ms, 28.41 fps
    - retinaface resnet50 600x600   112.7802 ms, 8.86 fps

## Accuracy Results
- fp32
    - easy_val is 0.70951
    - medium_val is 0.58483
    - hard_val is 0.25748
- int8
    - easy_val is 0.48025
    - medium_val is 0.30337
    - hard_val is 0.12671