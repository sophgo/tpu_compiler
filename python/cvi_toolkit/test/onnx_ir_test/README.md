## How

* add new case in test_onnx.py, referring to existing cases
* new case should be batch 1
* test command: python test_onnx.py

## cmdbuf test
succss:
* Neg
* Sub
* Sum

fail:
* LeakyRelu: int8 interpreter success, but cmdbuf fail
* Max: add backend later
* Min: add backend later

## issue
* if onnx model have no weight, quant to int8 will fail