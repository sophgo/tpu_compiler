# SSD-300  

For SSD-300 model you can download caffemodel and prototxt here.

- `https://drive.google.com/open?id=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc`

For coco dataset(2017) you can download in NAS server(http://10.34.33.5:5000/)
- `/ai/dataset_zoo/coco/2017/annotations`

# You need to add below modefication for run SSD-300 regression

1. change mlir-tpu-interpreter.cpp L178

  std::vector<float> input(1*3*224*224); --> std::vector<float> input(1*3*300*300);
2. Currently dynamic output tensor is not supported. 

   1) need to mark L1545-L1948 in lib/Dialect/StandardOps/ops.cpp

   2) For ssd300 network detection ouput layer, set (keep_top_k)X1X1X7 shape now where 'keep_top_k' value is the same as keep_top_k parameter from DetectionOutput layer(need to use 300 now). 
3. Set keep_top_k = 300 in detection_out layer to caffe prototxt.


# History 
## 1/3/2020 
Finish support SSD300 network FP32 inference (with output detection out post process)

## 1/16/2020

Finish SSD300 fp32 support.

Accuracy status:

- caffe fp32 coco 2017 dataset accuracy test result:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.103
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.578

- mlir fp32 accuracy test result:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.103
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.578

## 1/22/2020
Finish SSD300 INT8 interpreter support.

Known issue: 

1. Softmax threshold will affect accuracy . 
Tmp Solution: set softmax op threshold as 1. 