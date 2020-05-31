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
* Neg
* Slice
* Sub
* Sum

fail:
* Relu: will fused in int8
* Max: low-to-tg failed
* Min: low-to-tg failed
* PRelu: mlir-opt failed
* Reciprocal: mlir-opt failed

## issue
* LRN not support negtive input
* if onnx model have no weight, quant to int8 will fail