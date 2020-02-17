# inception V4

link
-  https://github.com/soeaver/caffe-model/tree/master/cls

download deploy_inception-v4.prototxt & inception-v4.caffemodel from
  https://drive.google.com/drive/folders/0B9mkjlmP0d7zUEJ3aEJ2b3J0RFU

mean_value: [128.0, 128.0, 128.0]
std: [128.0, 128.0, 128.0]  => 0.0078125

## Dataset

- imagenet

## Performance Results

## Accuracy Results

caffe:
  Test: [49400/50000]     Time  0.972 ( 0.925)    Loss 6.8985e+00 (6.2163e+00)    Acc@1   0.00 ( 79.27)   Acc@5   0.00 ( 94.45)
  Test: [49600/50000]     Time  0.975 ( 0.925)    Loss 6.0441e+00 (6.2165e+00)    Acc@1 100.00 ( 79.24)   Acc@5 100.00 ( 94.45)
  Test: [49800/50000]     Time  1.068 ( 0.925)    Loss 6.0851e+00 (6.2164e+00)    Acc@1 100.00 ( 79.25)   Acc@5 100.00 ( 94.46)
  Test: [50000/50000]     Time  0.785 ( 0.925)    Loss 6.9059e+00 (6.2164e+00)    Acc@1   0.00 ( 79.25)   Acc@5 100.00 ( 94.46)
   * Acc@1 79.248 Acc@5 94.464

inception_v4.mlir:
  Test: [49850/50000]     Time  3.236 ( 3.260)    Loss 6.4412e+00 (6.2046e+00)    Acc@1 100.00 ( 79.18)   Acc@5 100.00 ( 94.45)
  Test: [49900/50000]     Time  3.228 ( 3.260)    Loss 5.9444e+00 (6.2046e+00)    Acc@1 100.00 ( 79.18)   Acc@5 100.00 ( 94.44)
  Test: [49950/50000]     Time  3.192 ( 3.260)    Loss 5.9256e+00 (6.2046e+00)    Acc@1 100.00 ( 79.19)   Acc@5 100.00 ( 94.44)
  Test: [50000/50000]     Time  3.164 ( 3.260)    Loss 6.0808e+00 (6.2046e+00)    Acc@1 100.00 ( 79.19)   Acc@5 100.00 ( 94.44)
   * Acc@1 79.192 Acc@5 94.442

inception_v4_quant_int8_per_layer.mlir:
  Test: [37500/50000]     Time 23.656 (25.736)    Loss 6.0512e+00 (6.2726e+00)    Acc@1 100.00 ( 77.23)   Acc@5 100.00 ( 93.46)
  Test: [37550/50000]     Time 24.515 (25.734)    Loss 6.8972e+00 (6.2725e+00)    Acc@1   0.00 ( 77.24)   Acc@5   0.00 ( 93.46)
  Test: [37600/50000]     Time 25.341 (25.807)    Loss 5.9994e+00 (6.2726e+00)    Acc@1 100.00 ( 77.23)   Acc@5 100.00 ( 93.45)
  Test: [37650/50000]     Time 25.532 (25.806)    Loss 6.2398e+00 (6.2726e+00)    Acc@1 100.00 ( 77.23)   Acc@5 100.00 ( 93.45)

inception_v4_quant_int8_per_channel.mlir:
  TBD:

inception_v4_quant_int8_multiplier.mlir:
  Test: [32100/50000]     Time 23.969 (27.200)    Loss 6.3060e+00 (6.2693e+00)    Acc@1 100.00 ( 77.86)   Acc@5 100.00 ( 93.82)
  Test: [32150/50000]     Time 27.640 (27.200)    Loss 6.0249e+00 (6.2692e+00)    Acc@1 100.00 ( 77.87)   Acc@5 100.00 ( 93.82)
  Test: [32200/50000]     Time 27.549 (27.199)    Loss 6.2833e+00 (6.2692e+00)    Acc@1 100.00 ( 77.88)   Acc@5 100.00 ( 93.83)
  Test: [32250/50000]     Time 26.485 (27.198)    Loss 6.6590e+00 (6.2691e+00)    Acc@1 100.00 ( 77.89)   Acc@5 100.00 ( 93.83)

inception_v4_quant_bf16.mlir:
  TBD:
