# SSD-300  

For SSD-300 model you can download caffemodel and prototxt in NAS server(http://10.34.33.5:5000/)

- `/ai/model_zoo/object_detection/ssd300_coco/caffe/fp32/2016.11.30.01`

# You need to add below modefication for run SSD-300 regression

1. change mlir-tpu-interpreter.cpp L178

  std::vector<float> input(1*3*224*224); --> std::vector<float> input(1*3*300*300);
2. remove detection_out and all priobox layers(not TPU supported)

# History 
## 1/3/2020 
Finish support SSD300 network FP32 inference (with output detection out post process)