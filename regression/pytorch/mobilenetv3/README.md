# Mobilenetv3

# Model Path

  you can refer from [here](https://pypi.org/project/geffnet/) for more details or just run `gen.py`

# Convert to Onnx

  you just use following commands:
  ```
  python3 ./gen.py
  ```

  and default output name is `mobilenetv3_rw.onnx`

#Accuracy
###onnx
the [original one](https://pypi.org/project/geffnet/) `mobilenetv3_rw` is 
Prec@1 (Err): 75.634 (24.366)
Prec@5 (Err): 92.708 (7.292)

### fp32
Test: [49900/50000]     Time  0.184 ( 0.155)    Loss 5.2522e-01 (1.0822e+00)    Acc@1 100.00 ( $4.40)   Acc@5 100.00 ( 92.39)
Test: [49950/50000]     Time  0.144 ( 0.155)    Loss 3.6687e+00 (1.0823e+00)    Acc@1   0.00 ( $4.39)   Acc@5 100.00 ( 92.39)
Test: [50000/50000]     Time  0.159 ( 0.155)    Loss 3.6980e-01 (1.0822e+00)    Acc@1 100.00 ( $4.39)   Acc@5 100.00 ( 92.39)
 * Acc@1 74.392 Acc@5 92.394

Test: [ 1000/50000]     Time  0.146 ( 0.152)    Loss 3.2913e-01 (1.0927e+00)    Acc@1 100.00 ( 74.90)   Acc@5 100.00 ( 92.40)
 * Acc@1 74.900 Acc@5 92.400

#Accuracy int8, mix-precision
plz refer \regression_7_int8_mix_precision.sh
### config 1

    --tpu-quant \
    --quant-int8-mix-bf16-layers "313_BatchNormalization" \
    --quant-int8-mix-bf16-layers "315_Add" \
    --quant-int8-mix-bf16-layers "316_Clip" \
    --quant-int8-mix-bf16-layers "319_Mul" \
    --quant-int8-mix-bf16-layers "322_Relu" \
    --quant-int8-mix-bf16-layers "353_Add" \
    --quant-int8-mix-bf16-layers "354_Clip" \
    --quant-int8-mix-bf16-layers "372_Clip" \
    --quant-int8-mix-bf16-layers "391_Clip" \
    --quant-int8-mix-bf16-layers "402_Clip" \
    --quant-int8-mix-bf16-layers "410_Clip" \
    --quant-int8-mix-bf16-layers "420_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "431_Mul" \
    --quant-int8-mix-bf16-layers "439_Clip" \
    --quant-int8-mix-bf16-layers "447_Clip" \
    --quant-int8-mix-bf16-layers "458_Clip" \
    --quant-int8-mix-bf16-layers "466_Clip" \
    --quant-int8-mix-bf16-layers "477_Clip" \
    --quant-int8-mix-bf16-layers "485_Clip" \
    --quant-int8-mix-bf16-layers "493_Clip" \
    --quant-int8-mix-bf16-layers "500_Clip" \
    --quant-int8-mix-bf16-layers "510_Clip" \
    --quant-int8-mix-bf16-layers "518_Clip" \
    --quant-int8-mix-bf16-layers "526_Clip" \
    --quant-int8-mix-bf16-layers "533_Clip" \
    --quant-int8-mix-bf16-layers "544_Clip" \
    --quant-int8-mix-bf16-layers "552_Clip" \
    --quant-int8-mix-bf16-layers "560_Clip" \
    --quant-int8-mix-bf16-layers "567_Clip" \
    --quant-int8-mix-bf16-layers "577_Clip" \
    --quant-int8-mix-bf16-layers "585_Clip" \
    --quant-int8-mix-bf16-layers "593_Clip" \
    --quant-int8-mix-bf16-layers "600_Clip" \
    --quant-int8-mix-bf16-layers "611_Clip" \
    --quant-int8-mix-bf16-layers "619_Clip" \
    --quant-int8-mix-bf16-layers "627_Clip" \
    --quant-int8-mix-bf16-layers "634_Clip" \
    --quant-int8-mix-bf16-layers "645_Clip" \
    --quant-int8-mix-bf16-layers "653_Clip" \
    --quant-int8-mix-bf16-layers "521_Mul" \
    --quant-int8-mix-bf16-layers "433_BatchNormalization" \
    --quant-int8-mix-bf16-layers "536_Mul" \
    --quant-int8-mix-bf16-layers "538_BatchNormalization" \
    --quant-int8-mix-bf16-layers "588_Mul" \
    --quant-int8-mix-bf16-layers "529_Mul" \

Test: [  950/50000]     Time  0.522 ( 0.620)    Loss 4.5458e-01 (1.3331e+00)    Acc@1 100.00 ( 69.58)   Acc@5 100.00 ( 90.11)
Test: [ 1000/50000]     Time  0.658 ( 0.619)    Loss 3.5897e+00 (1.3338e+00)    Acc@1   0.00 ( 69.50)   Acc@5 100.00 ( 90.20)
 * Acc@1 69.500 Acc@5 90.200
tensor(69.5000)

### config 2
    --quant-int8-mix-bf16-layers "316_Clip" \
    --quant-int8-mix-bf16-layers "354_Clip" \
    --quant-int8-mix-bf16-layers "372_Clip" \
    --quant-int8-mix-bf16-layers "391_Clip" \
    --quant-int8-mix-bf16-layers "402_Clip" \
    --quant-int8-mix-bf16-layers "410_Clip" \
    --quant-int8-mix-bf16-layers "420_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "439_Clip" \
    --quant-int8-mix-bf16-layers "447_Clip" \
    --quant-int8-mix-bf16-layers "458_Clip" \
    --quant-int8-mix-bf16-layers "466_Clip" \
    --quant-int8-mix-bf16-layers "477_Clip" \
    --quant-int8-mix-bf16-layers "485_Clip" \
    --quant-int8-mix-bf16-layers "493_Clip" \
    --quant-int8-mix-bf16-layers "500_Clip" \
    --quant-int8-mix-bf16-layers "510_Clip" \
    --quant-int8-mix-bf16-layers "518_Clip" \
    --quant-int8-mix-bf16-layers "526_Clip" \
    --quant-int8-mix-bf16-layers "533_Clip" \
    --quant-int8-mix-bf16-layers "544_Clip" \
    --quant-int8-mix-bf16-layers "552_Clip" \
    --quant-int8-mix-bf16-layers "560_Clip" \
    --quant-int8-mix-bf16-layers "567_Clip" \
    --quant-int8-mix-bf16-layers "577_Clip" \
    --quant-int8-mix-bf16-layers "585_Clip" \
    --quant-int8-mix-bf16-layers "593_Clip" \
    --quant-int8-mix-bf16-layers "600_Clip" \
    --quant-int8-mix-bf16-layers "611_Clip" \
    --quant-int8-mix-bf16-layers "619_Clip" \
    --quant-int8-mix-bf16-layers "627_Clip" \
    --quant-int8-mix-bf16-layers "634_Clip" \
    --quant-int8-mix-bf16-layers "645_Clip" \
    --quant-int8-mix-bf16-layers "653_Clip" \
    --quant-int8-mix-bf16-layers "313_BatchNormalization" \
    --quant-int8-mix-bf16-layers "315_Add" \
    --quant-int8-mix-bf16-layers "319_Mul" \
    --quant-int8-mix-bf16-layers "322_Relu" ${comment#54 passed} \
    --quant-int8-mix-bf16-layers "353_Add"  ${comment# 153 passed} \

Test: [  950/50000]     Time  0.627 ( 0.696)    Loss 2.3699e-01 (1.6018e+00)    Acc@1 100.00 ( 63.16)   Acc@5 100.00 ( 86.42)
Test: [ 1000/50000]     Time  0.545 ( 0.691)    Loss 2.2609e+00 (1.5965e+00)    Acc@1   0.00 ( 63.40)   Acc@5 100.00 ( 86.60)
 * Acc@1 63.400 Acc@5 86.600

Test: [50000/50000]     Time  0.535 ( 0.660)    Loss 2.3589e+00 (1.5537e+00)    Acc@1   0.00 ( 6
4.92)   Acc@5 100.00 ( 87.10)
 * Acc@1 64.924 Acc@5 87.104 
