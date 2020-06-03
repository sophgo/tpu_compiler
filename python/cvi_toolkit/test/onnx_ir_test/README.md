# ONNX IR Test


## How to use
* add new case in test_onnx.py, referring to existing cases
* new case should be batch 1
* test command:
```python test_onnx.py```

if test cmdbuf, set self.cvi_model_test = True


## fp32

All pass


## cmdbuf test
succss:
* Add
* AveragePool
* GlobalMaxPool
* LeakyRelu
* LRN
* Max
* Min
* Neg
* PRelu
* Slice
* Sub
* Sum

fail:
* Relu: will fused in int8
* Reciprocal: mlir-opt failed

## issue
* if onnx model have no weight, quant to int8 will fail
* LRN: not support negtive input
* PRelu: input[1] must be weight; only support per channel