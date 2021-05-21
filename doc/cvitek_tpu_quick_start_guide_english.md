![image](./assets/logo_0.png)

# CV183x/CV182x TPU Quick start guide




>Document version: 1.5.0
>
>Release date: 2021-01-29



© 2020 CVITEK Co., Ltd. All rights reserved.

No part of this document may be reproduced or transmiited in any form or by any means without prior written consent of CVITEK Co., Ltd.

<div STYLE="page-break-after: always;"></div>

## Revision record

| Version | Data       | Modifier            | Modify description                                           |
| ------- | ---------- | ------------------- | ------------------------------------------------------------ |
| V0.0.1  | 2019/12/11 | Lei Wang            | Initial version                                              |
| V0.1.0  | 2020/04/18 | Lei Wang            | Add instructions for use                                     |
| V0.1.1  | 2020/04/20 | Lei Wang            | Update test command                                          |
| V0.1.2  | 2020/04/28 | Lei Wang            | Update model accuracy test command                           |
| V0.2.0  | 2020/04/30 | Lei Wang            | Revision                                                     |
| V0.2.1  | 2020/05/15 | Lei Wang            | Update some commands according to V0.9 SDK                   |
| V0.2.2  | 2020/06/01 | Lei Wang            | Add end-to-end inference performance test commands           |
| V0.3.0  | 2020/06/25 | Lei Wang            | Compile the TensorFlow 2.x model according to the V1.0 SDK revision |
| V0.3.1  | 2020/06/29 | Lei Wang            | Caffe transplantation using python importer                  |
| V0.3.2  | 2020/07/17 | Quan Li             | Added Chapter 9 Using TPU for pre-processing                 |
| V0.3.3  | 2020/07/19 | Lei Wang            | Revised according to V1.1 SDK                                |
| V0.3.4  | 2020/07/20 | Lei Wang            | Revision                                                     |
| V0.3.5  | 2020/07/29 | Lei Wang            | Increase porting and compiling TensorFlow 2.x model          |
| V0.3.6  | 2020/08/08 | Lei Wang            | Increase precision optimization and mixed quantization guidelines |
| V0.3.7  | 2020/08/12 | Lei Wang            | Update the pre-processing process using TPU                  |
| V0.3.8  | 2020/09/06 | Lei Wang            | Revised according to V1.2 SDK                                |
| V0.3.9  | 2020/09/22 | Quan Li             | Increase the transplantation of tflite model                 |
| V0.3.10 | 2020/09/30 | Sam Zheng           | Update and transplant the tflite model                       |
| V0.3.11 | 2020/10/26 | Quan Xiao/Charle Hu | Revised according to V1.3 SDK                                |
| V1.4.0  | 2020/12/07 | Quan Xiao/Charle Hu | Revised according to V1.4 SDK                                |
| V1.5.0  | 2021/01/29 | Quan Xiao/Charle Hu | Revised according to V1.5 SDK                                |

<div STYLE="page-break-after: always;"></div>

# Terms and Conditions

The document and all information contained herein remain the CVITEK Co., Ltd’s (“CVITEK”）confidential information, and should not disclose to any third party or use it in any way without CVITEK’s prior written consent. User shall be liable for any damage and loss caused by unauthority use and disclosure.

CVITEK reserves the right to make changes to information contained in this document at any time and without notice.

All information contained herein is provided in “AS IS” basis, without warranties of any kind, expressed or implied, including without limitation mercantability, non-infringement and fitness for a particular purpose. In no event shall CVITEK be liable for any third party’s software provided herein, User shall only seek remedy against such third party. CVITEK especially disclaims that CVITEK shall have no liable for CVITEK’s work result based on Customer’s specification or published shandard.

<div STYLE="page-break-after: always;"></div>

##  Contents

* content
{:toc}



<div STYLE="page-break-after: always;"></div>

## 1 Overview

#### 1.1 Read instructions

This document contains the following chapters, please refer to the relevant chapters as needed.

* Run sample

  No need to compile, run the sample programs and models provided with the release on EVB, including:

  * Execute the samples program
  * Test the correctness and performance of cvimodel

* Development environment configuration

  Use the docker provided by CVITEK to configure the environment required for compilation and development

* Compile the samples program

  Introduce how to cross-compile the sample application and call the runtime API to complete the inference task. Specifically includes 4 samples:

  * Sample-1 : classifier (mobilenet_v2)

  * Sample-2 : detector (yolo_v3)

  * Sample-3 : alphapose (yolo_v3 + fastpose)

  * Sample-4 : insightface (retinaface + arcface)

* Compile and generate cvimodel

  Introduce how to generate all cvimodels for samples and tests through scripts

* Compile and transplant the caffe model

  Introduce how to transplant a new caffe model, take `mobilenet_v2` as an example

* Compile and transplant the pytorch model

  Introduce how to transplant a new pytorch model, taking `resnet18` as an example

* Compile and transplant tensorflow 2.x model

  Introduce how to transplant a new tensorflow 2.x model, take `mobilenet_v2` as an example

* Compile and transplant tensorflow 1.x model

  Introduce how to transplant a new tensorflow 1.x model, take `mobilenet_v1_0.25_224` as an example

* Use TPU for preprocessing

  Introduce how to add pre-processing description in the cvimodel model, and use TPU for pre-processing at runtime



#### 1.2 Release content

CVITEK Release includes the following components:

| Files                                          | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| cvitek_mlir_ubuntu-18.04.tar.gz                | cvitek NN tool chain software                                |
| cvitek_tpu_sdk_[cv182x/cv183x].tar.gz          | cvitek Runtime SDK, including cross-compiled header files and library files |
| cvitek_tpu_samples.tar.gz                      | sample source code                                           |
| cvimodel_samples_[cv182x/cv183x].tar.gz        | The cvimodel model file used by the sample program           |
| cvimodel_regression_bs1_[cv182x/cv183x].tar.gz | Model cvimodel files and corresponding input and output data files |
| cvimodel_regression_bs4_[cv182x/cv183x].tar.gz | Model cvimodel files and corresponding input and output data files |
| docker_cvitek_dev.tar                          | CVITEK develops Docker image file                            |
| models.tar.gz                                  | Caffe/onnx original model file package for testing (support github download) |
| dataset.tar.gz                                 | Dataset package for testing (downloadable on github, refer to REAMDE preparation) |



#### 1.3 Models 和 Dataset

The original framework model file and dataset used for testing can be obtained from the following link, and refer to the description of README.md to prepare accordingly.

* <https://github.com/cvitek-mlir/models>

* <https://github.com/cvitek-mlir/dataset>

<div STYLE="page-break-after: always;"></div>

## 2 Run test

No need to compile, run the sample precompiled program and model provided by release in EVB.

The following documents are required for this chapter:

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvimodel_samples_[cv182x/cv183x].tar.gz
* cvimodel_regression_bs1_[cv182x/cv183x].tar.gz
* cvimodel_regression_bs4_[cv182x/cv183x].tar.gz



#### 2.1 Run the sample program

Load the required file to the EVB file system according to the chip type, and execute it on the linux console on the evb. Take cv183x as an example:

Unzip the model file used by samples (delivered in cvimodel format), unzip TPU_SDK, and enter the samples directory to Execute the test. The process is as follows:

``` evb_shell
# envs
tar zxf cvimodel_samples_cv183x.tar.gz
export MODEL_PATH=$PWD/cvimodel_samples
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_ROOT=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh

# get cvimodel info
cd samples
./bin/cvi_sample_model_info $MODEL_PATH/mobilenet_v2.cvimodel

####################################
# sample-1 : classifier
###################################
./bin/cvi_sample_classifier_mobilenet_v2 \
    $MODEL_PATH/mobilenet_v2.cvimodel \
    ./data/cat.jpg \
    ./data/synset_words.txt
# TOP_K[5]:
#    0.356300, idx 285, n02124075 Egyptian cat
#    0.062108, idx 287, n02127052 lynx, catamount
#    0.046420, idx 331, n02326432 hare
#    0.006048, idx 852, n04409515 tennis ball
#    0.000788, idx 876, n04493381 tub, vat

############################################
# sample-2 : detector
############################################
./bin/cvi_sample_detector_yolo_v3 \
    $MODEL_PATH/yolo_v3_416_with_detection.cvimodel \
    ./data/dog.jpg \
    yolo_v3_out.jpg

############################################
# sample-3 : alphapose
############################################
./bin/cvi_sample_alphapose \
     $MODEL_PATH/yolo_v3_416_with_detection.cvimodel \
     $MODEL_PATH/alphapose.cvimodel \
     ./data/pose_demo_2.jpg \
     alphapose_out.jpg

############################################
# sample-4 : insightface
############################################
./bin/cvi_sample_fd_fr \
     $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
     $MODEL_PATH/arcface_res50.cvimodel \
     ./data/obama1.jpg ./data/obama2.jpg
# Similarity: 0.735814
./bin/cvi_sample_fd_fr \
     $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
     $MODEL_PATH/arcface_res50.cvimodel \
     ./data/obama1.jpg ./data/trump1.jpg
# Similarity: -0.0169034
```

At the same time, the script is provided as a reference, and the execution effect is the same as that of direct operation, as follows:

``` evb_shell
./run_classifier.sh
./run_detector.sh
./run_alphapose.sh
./run_insightface.sh
```

There are also scripts that use preprocessing as a reference, as follows:

``` evb_shell
./run_classifier_fused_preprocess.sh
./run_detector_fused_preprocess.sh
./run_alphapose_fused_preprocess.sh
./run_insightface_fused_preprocess.sh
```



#### 2.2 Test cvimodel

Execute the script regression_models.sh in EVB. The script calls model_runner for each network to Execute inference calculations, compares whether the output data is correct, and prints running time information at the same time.

* Inference performance test based on PMU data

  The regression model file is divided into two parts, bs=1 and bs=4, and tests are performed respectively to test the correctness and operating efficiency of all networks. Take the cv183x platform as an example:

  ``` evb_shell
  cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..
  export TPU_ROOT=$PWD/cvitek_tpu_sdk

  # For batch_size = 1
  tar zxf cvimodel_regression_bs1_cv183x.tar.gz
  MODEL_PATH=$PWD/cvimodel_regression_bs1 $TPU_ROOT/regression_models.sh

  # For batch_size = 4
  tar zxf cvimodel_regression_bs4_cv183x.tar.gz
  MODEL_PATH=$PWD/cvimodel_regression_bs4 $TPU_ROOT/regression_models.sh batch

  # Run one model (eg. Resnet50 run once)
  MODEL_PATH=$PWD/cvimodel_regression_bs1 $TPU_ROOT/regression_models.sh resnet50 1
  ```



* End-to-end performance test based on system clock

  Include the end-to-end network inference time including data input, post-processing and data export time. Take the cv183x platform as an example:

  ``` evb_shell
  cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..
  export TPU_ROOT=$PWD/cvitek_tpu_sdk
  export PATH=$TPU_ROOT/samples/bin:$PATH

  tar zxf cvimodel_regression_bs1_cv183x.tar.gz
  MODEL_PATH=$PWD/cvimodel_regression_bs1 $TPU_ROOT/regression_models_e2e.sh
  ```



#### 2.3 List of networks currently supported for testing

The networks supported by cv183x are as follows:

| Classification                                               | Detection                                                    | Misc                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| resnet50       [BS=1,4] <br />resnet18       [BS=1,4]<br />mobilenet_v1     [BS=1,4]<br />mobilenet_v2     [BS=1,4]<br />squeezenet_v1.1    [BS=1,4]<br />shufflenet_v2     [BS=1,4]<br />googlenet       [BS=1,4]<br />inception_v3     [BS=1,4]<br />inception_v4     [BS=1,4]<br />vgg16         [BS=1,4]<br />densenet_121     [BS=1,4]<br />densenet_201     [BS=1,4]<br />senet_res50      [BS=1,4]<br />resnext50       [BS=1,4]<br />res2net50       [BS=1,4]<br />ecanet50       [BS=1,4]<br />efficientnet_b0    [BS=1,4]<br />efficientnet_lite_b0 [BS=1,4]<br />nasnet_mobile     [BS=1,4] | retinaface_mnet25 [BS=1,4]<br />retinaface_res50   [BS=1]<br />ssd300        [BS=1,4]<br />mobilenet_ssd [BS=1,4]<br />yolo_v1_448      [BS=1]<br />yolo_v2_416      [BS=1]<br />yolo_v2_1080     [BS=1]<br />yolo_v3_416      [BS=1,4]<br />yolo_v3_608      [BS=1]<br />yolo_v3_tiny     [BS=1]<br />yolo_v3_spp      [BS=1]<br />yolo_v4        [BS=1] | arcface_res50 [BS=1,4]<br />alphapose       [BS=1,4]<br />espcn_3x       [BS=1,4]<br />unet          [BS=1,4]<br />erfnet         [BS=1] |

cv182x支持的网络如下：

| Classification                                               | Detection                                                    | Misc                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| resnet50       [BS=1,4] <br />resnet18       [BS=1,4]<br />mobilenet_v1     [BS=1,4]<br />mobilenet_v2     [BS=1,4]<br />squeezenet_v1.1    [BS=1,4]<br />shufflenet_v2     [BS=1,4]<br />googlenet       [BS=1,4]<br />inception_v3     [BS=1]<br />densenet_121     [BS=1,4]<br />densenet_201     [BS=1]<br />senet_res50      [BS=1]<br />resnext50       [BS=1,4]<br />efficientnet_lite_b0 [BS=1,4]<br />nasnet_mobile     [BS=1] | retinaface_mnet25 [BS=1,4]<br />retinaface_res50   [BS=1]<br />mobilenet_ssd [BS=1,4]<br />yolo_v1_448      [BS=1]<br />yolo_v2_416      [BS=1]<br />yolo_v3_416      [BS=1,4]<br />yolo_v3_608      [BS=1]<br />yolo_v3_tiny     [BS=1]<br /> | arcface_res50 [BS=1,4]<br />alphapose       [BS=1,4]<br /> |



**Note:** BS means batch, [BS=1] means that the board currently has at least batch 1, and [BS=1,4] means that the board supports at least batch 1 and batch 4.

<div STYLE="page-break-after: always;"></div>

## 3 Development environment configuration

Obtain from docker hub (recommended):

```
docker pull cvitek/cvitek_dev:1.4-ubuntu-18.04
```

Or load the image file:

```
docker load -i docker_cvitek_dev_1.4-ubuntu-18.04.tar
```



If it is the first time to use docker, execute the following commands to install and configure (Ubuntu system)

```
sudo apt install docker.io
systemctl start docker
systemctl enable docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker (use before reboot)
```



After obtaining the docker image, execute the following command to run docker:

```
# It is assumed that models and datasets are located in the ~/data/models and ~/data/dataset directories respectively, please adjust # accordingly if they are different.
docker run -itd -v $PWD:/work \
   -v ~/data/models:/work/models \
   -v ~/data/dataset:/work/dataset \
   --name cvitek cvitek/cvitek_dev:1.4-ubuntu-18.04
docker exec -it cvitek bash
```

<div STYLE="page-break-after: always;"></div>

## 4 Compile the samples program

Please select the corresponding TPU sdk to cross-compile the samples code according to the chip type, load it to the evb and run the test.

The following documents are required for this section:

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvitek_tpu_samples.tar.gz


#### cv183x platform 64-bit

TPU sdk preparation:

``` shell
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

Compile the samples and install to the install_samples directory:

``` shell
tar zxf cvitek_tpu_samples.tar.gz
cd cvitek_tpu_samples
mkdir build_soc
cd build_soc
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
    -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-aarch64-linux.cmake \
    -DTPU_SDK_PATH=$TPU_SDK_PATH \
    -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
    -DCMAKE_INSTALL_PREFIX=../install_samples \
    ..
cmake --build . --target install
```

#### cv182x platform 32-bit

TPU sdk preparation:

``` shell
tar zxf cvitek_tpu_sdk_cv182x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

Update the 32-bit system library (only once):

``` shell
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386
```

Compile the samples and install to the install_samples directory:

``` shell
tar zxf cvitek_tpu_samples.tar.gz
cd cvitek_tpu_samples
mkdir build_soc
cd build_soc
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
    -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-linux-gnueabihf.cmake \
    -DTPU_SDK_PATH=$TPU_SDK_PATH \
    -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
    -DCMAKE_INSTALL_PREFIX=../install_samples \
    ..
cmake --build . --target install
```

<div STYLE="page-break-after: always;"></div>

## 5 Compile and generate cvimodel for testing

The following documents are required for this section:

* cvitek_mlir_ubuntu-18.04.tar.gz
* models.tar.gz
* dataset.tar.gz

#### 5.1 Call the script to generate cvimodel

Prepare TPU simulation development environment:

```
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```

 Use the following script commands to quickly generate all cvimodel files for testing:

```
generate_all_cvimodels.sh
```

 Generate the `regression_out/cvimodel_release `directory, the model contained in `cvimodel_samples` is a subset of the `cvimodel_release` generated here.



#### 5.2 Run regression tests to generate cvimodel and input and output data

Use the following regression test commands to compare and verify the results of each step of model transplantation. At the same time, you can also choose to perform accuracy testing.

At the end of the regression test, in addition to generating all test cvimodel files, the input and output test data of each model is also generated at the same time, which is used to load the EVB for model testing.



Prepare TPU simulation development environment:

```
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```



Use the following command to start the regression test. The test network is classified into basic and extra to adjust the test time, and the user can also edit the `run_regression.sh` self-adjustment list.

```
run_regression.sh    # basic  models only  Or
run_regression.sh -e  # with  extra models
```



The generated `cvimodel_regression` content is consistent with the `cvimodel_regression.tar.gz` content in the `release`, and can be loaded into the EVB for consistency and performance testing.

The user can also execute regression test on one of the networks separately. The command is as follows (take `resnet50` as an example). For the list of supported networks, see section 2.2.

```
regression_generic.sh resnet50
```



#### 5.3 Test model accuracy

After executing `run_regression.sh`, we can use the script to test the accuracy of the mlir model and compare it with the accuracy of the original model. The command is:

```
source cvitek_mlir/cvitek_envs.sh
cd regression_out
# accuracy_generic.sh ${NET} ${COUNT}
# Eg. Imagenet
accuracy_generic.sh mobilenet_v2 50000 2>&1 | tee mnet_v2_50000.txt
accuracy_generic.sh resnet50 50000 2>&1 | tee res50_50000.txt
# Eg. coco
accuracy_generic.sh yolo_v3_416 5000 2>&1 | tee yolo_v3_416_5000.txt
accuracy_generic.sh yolo_v3_320 5000 2>&1 | tee yolo_v3_320_5000.txt
```

**Note: ** Need to prepare imagenet or coco data set, see section 1.3.

<div STYLE="page-break-after: always;"></div>

## 6 Compile and transplant the caffe model

This chapter uses `mobilenet_v2` as an example to introduce how to compile and migrate a caffe model to the CV183x TPU platform; if you need to switch to the cv182x platform, you can specify it through the command line parameter `--chipname cv182x`.

 The following documents are required for this chapter:

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz

#### Step 0: Load the cvitek_mlir environment

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### Step 1: Obtain the caffe model

Download the model from <https://github.com/shicai/MobileNet-Caffe> and save it in the `models_mobilenet_v2` directory:

``` shell
mkdir models_mobilenet_v2 && cd models_mobilenet_v2
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt
```

Create a working directory workspace:

``` shell
mkdir workspace && cd workspace
```

#### Step 2: Model conversion

Use `model_transform.py` to convert the model into a mlir file, which has preprocessing parameters as follows:

| **parameter** | **Description**                  |
| ------------------- | ------------------------------------ |
| image_resize_dims   | Indicates the image resize size, such as 256,256 |
| keep_aspect_ratio   | Whether to maintain the aspect ratio when resizing |
| net_input_dims      | Indicates the size of the model input, such as 224,224 |
| model_channel_order | Channel order, default bgr; can be specified as rgb |
| raw_scale           | Operation: *raw_scale/255.0, the default is 255.0 |
| mean                | Operation:-mean, default is 0.0,0.0,0.0 |
| std                 | Operation: /std, the default is 1.0,1.0,1.0 |
| input_scale         | Operation: * input_scale, the default is 1.0 |

The preprocessing process is expressed in the following formula (x represents input):
$$
y = \frac{x \times \frac{raw\_scale}{255.0} - mean}{std} \times input\_scale
$$

Convert from caffe model to mlir:

``` shell
model_transform.py \
  --model_type caffe \
  --model_name mobilenet_v2 \
  --model_def ../mobilenet_v2_deploy.prototxt \
  --model_data ../mobilenet_v2.caffemodel \
  --image ./cat.jpg \
  --image_resize_dims 256,256 \
  --keep_aspect_ratio false \
  --net_input_dims 224,224 \
  --raw_scale 255.0 \
  --mean 103.94,115.78,123.68 \
  --std 1.0,1.0,1.0 \
  --input_scale 0.017 \
  --model_channel_order "bgr" \
  --gray false \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --excepts prob \
  --mlir mobilenet_v2_fp32.mlir
```

Get the `mobilenet_v2_fp32.mlir` file.

The conversion process includes:

- Inference of the original caffe model, and save the results of each layer as a numpy npz file

- Import the original caffe model, convert the original model into an MLIR fp32 model
  - Perform the inference of the MLIR fp32 model, and save the output of each layer to the numpy npz file
  - Compare the inference results of the caffe model with the inference results of MLIR fp32 to ensure that the converted MLIR fp32 model is correct
  - Optimize the MLIR fp32 model as the input of the subsequent process

**Note:** The preprocessing parameters entered above are only stored in mlir in the form of information, and subsequently converted to cvimodel, they are also only stored in the form of information. The image preprocessing process needs to be processed externally and then passed to the model for calculation. If you need to pre-process the pictures inside the model, please refer to Chapter 12: Using TPU for pre-processing.

#### Step 3: Calibration

Before Calibration, you need to prepare a calibration picture set. The number of pictures is about 100~1000 according to the situation.
Execute calibration:

``` shell
run_calibration.py \
  mobilenet_v2_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mobilenet_v2_calibration_table
```

 Get `mobilenet_v2_calibration_table`.

#### Step 4: Model quantification and generate cvimodel

``` shell
model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.95,0.94,0.69 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mobilenet_v2.cvimodel
```

The above command contains the following steps:

- Generate the MLIR int8 model, run the inference of the MLIR quantitative model, and compare the results with the MLIR fp32 model
- Generate cvimodel, call the simulator to run the inference results, and compare the results with the MLIR quantitative model

**Note:** `--tolerance` represents the error tolerance of the similarity between the MLIR int8 quantitative model and the MLIR fp32 model inference result, `--correctnetss` represents the error tolerance of the similarity between the result of the simulator and the result of the MLIR int8 model,`--chip` can choose `cv183x` and `cv182x`, use `cv183x` by default

<div STYLE="page-break-after: always;"></div>

## 7 Compile and transplant the pytorch model

This chapter uses `resnet18` as an example to introduce how to compile and migrate a pytorch model to the CV183x TPU platform.

The following documents are required for this chapter:

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz



#### Step 0: Load the cvitek_mlir environment

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### Step 1: Obtain the pytorch model and convert it to onnx

Use the resnet18 model provided by torchvision<https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>

Use the following python script to download the pytorch model, and output the pytorch model to onnx format, and save it in the `model_resnet18` directory:

``` shell
mkdir model_resnet18
cd model_resnet18
```

Execute the python command:

``` python
# python
import torch
import  torchvision.models as models
# Use an existing model  from Torchvision, note it
# will download this if  not already on your computer (might take time)
model = models.resnet18(pretrained=True)
batch_size = 1
# Create some sample  input in the shape this model expects
dummy_input = torch.randn(batch_size, 3, 224, 224)
input_names = ['input']
output_names = ['output']
# Use the exporter from  torch to convert to onnx
torch.onnx.export(model,
    dummy_input,
    'resnet18.onnx',
    export_params=True,
    opset_version=10,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={'input'  : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}})
```

Get `resnet18.onnx`。

#### Step 2: Model conversion

Create a working directory and get a test picture. This example uses `cat.jpg` contained in `cvitek_mlir`

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

Before inference, we need to understand the preprocessing parameters of this model. The preprocessing of resnet18 is described in the link <https://pytorch.org/hub/pytorch_vision_resnet>:

> preprocess = transforms.Compose([
>
> ​    transforms.Resize(256),
>
> ​    transforms.CenterCrop(224),
>
> ​    transforms.ToTensor(),
>
> ​    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
>
> ])

Use `model_transform.py` to convert the onnx model into a mlir file:

``` shell
model_transform.py \
  --model_type onnx \
  --model_name resnet18 \
  --model_def ../resnet18.onnx \
  --image ./cat.jpg \
  --image_resize_dims 256,256 \
  --keep_aspect_ratio false \
  --net_input_dims 224,224 \
  --raw_scale 1.0 \
  --mean 0.406,0.456,0.485 \
  --std 0.255,0.244,0.229 \
  --input_scale 1.0 \
  --model_channel_order "rgb" \
  --gray false \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --mlir resnet18_fp32.mlir
```

Get the `resnet18_fp32.mlir` file.

The conversion process includes:

- Inference of the original onnx model, and save the results of each layer as a numpy npz file

- Import the original onnx model, convert the original model to MLIR fp32 model
  - Perform the inference of the MLIR fp32 model, and save the output of each layer to the numpy npz file
  - Compare the inference result of the onnx model with the inference result of MLIR fp32 to ensure that the converted MLIR fp32 model is correct
  - Optimize the MLIR fp32 model as the input of the subsequent process

**Note:** The preprocessing parameters entered above are only stored in mlir in the form of information, and subsequently converted to cvimodel, they are also only stored in the form of information. The image preprocessing process needs to be processed externally and then passed to the model for calculation. If you need to pre-process the pictures inside the model, please refer to Chapter 12: Using TPU for pre-processing.

#### Step 3: Calibration

Before Calibration, you need to prepare a calibration picture set. The number of pictures is about 100~1000 according to the situation.
Execute calibration:

``` shell
run_calibration.py \
  resnet18_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o resnet18_calibration_table
```

  Get `resnet18_calibration_table`。

#### Step 4: Model quantification and generate cvimodel

``` shell
model_deploy.py \
  --model_name resnet18 \
  --mlir resnet18_fp32.mlir \
  --calibration_table resnet18_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.99,0.99,0.87 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel resnet18.cvimodel
```

The above command also contains the following steps:

- Generate the MLIR int8 quantitative model, run the inference of the MLIR int8 quantitative model, and compare with the results of the MLIR fp32 model
- Generate cvimodel, call the simulator to run the inference result, compare the result with the MLIR int8 quantized model result

**Note:** `--tolerance` indicates the error tolerance of the similarity between the MLIR int8 quantized model and the MLIR fp32 model inference result, `--correctnetss` indicates the error tolerance of the similarity between the result of the simulator and the result of the MLIR int8 model, `--chip` can choose `cv183x` and `cv182x` to use `cv183x` by default

<div STYLE="page-break-after: always;"></div>

## 8 Compile and transplant tensorflow 2.x model

The TPU tool chain uses a direct import method for the Tensorflow 2.x model.

This chapter takes `mobilenet_v2` as an example to introduce how to compile and migrate a tensorflow 2.x model to run on the CV183x TPU platform.

The following documents are required for this chapter:

* cvitek_mlir.tar.gz
* dataset.tar.gz



#### Step 0: Load the cvitek_mlir environment

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### Step 1: Get the tensorflow model

Use the mobilenet_v2 model provided by tensorflow, <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2>

Use the following python script to download and save the model:

``` shell
mkdir model_mobilenet_v2_tf
cd model_mobilenet_v2_tf
```

Execute the python command:

``` python
# python
import tensorflow as tf
import numpy as np
import os
model =  tf.keras.applications.MobileNetV2()
model.save('mobilenet_v2',  save_format='tf')
```

The obtained model is saved in the `mobilenet_v2` directory, and the directory structure is as follows:

``` shell
tree mobilenet_v2
# mobilenet_v2/
# ├── assets
# ├── saved_model.pb
# └── variables
#    ├──  variables.data-00000-of-00001
#    └── variables.index
# 2 directories, 3 files
```

Create a workspace:

``` shell
mkdir workspace && cd workspace
```

#### Step 2: Model conversion

Get a test picture, this example uses `cat.jpg` contained in `cvitek_mlir`:

``` shell
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

Convert tensorflow model to mlir file

``` shell
model_transform.py \
  --model_type tensorflow \
  --model_name mobilenet_v2_tf \
  --model_def ../mobilenet_v2 \
  --image ./cat.jpg \
  --image_resize_dims 256,256 \
  --keep_aspect_ratio false \
  --net_input_dims 224,224 \
  --raw_scale 255.0 \
  --mean 127.5,127.5,127.5 \
  --std 127.5,127.5,127.5 \
  --input_scale 1.0 \
  --model_channel_order "rgb" \
  --gray false \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --mlir mobilenet_v2_tf_fp32.mlir
```

Get the `mobilenet_v2_tf_fp32.mlir` file.

The conversion process includes:

- Inference of the original tensorflow model, and save the results of each layer as a numpy npz file

- Import of the original tensorflow model, convert the original model into an MLIR fp32 model
  - 
  - Execute the inference of the MLIR fp32 model, and save the output of each layer to the numpy npz file
  - Compare the inference results of the tensorflow model with the inference results of MLIR fp32 to ensure that the converted MLIR fp32 model is correct
  - Optimize the MLIR fp32 model as the input of the subsequent process

**Note:** The preprocessing parameters entered above are only stored in mlir in the form of information, and subsequently converted to cvimodel, they are also only stored in the form of information. The image preprocessing process needs to be processed externally and then passed to the model for calculation. If you need to pre-process the pictures inside the model, please refer to Chapter 12: Using TPU for pre-processing.

#### Step 3: Calibration

Before Calibration, you need to prepare a calibration picture set. The number of pictures is about 100~1000 according to the situation.
Execute calibration：

``` shell
run_calibration.py \
  mobilenet_v2_tf_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mobilenet_v2_tf_calibration_table
```

  Get `mobilenet_v2_tf_calibration_table`。

#### Step 4: Model quantification and generate cvimodel

``` shell
model_deploy.py \
  --model_name mobilenet_v2_tf \
  --mlir mobilenet_v2_tf_fp32.mlir \
  --calibration_table mobilenet_v2_tf_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.95,0.94,0.64 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mobilenet_v2_tf.cvimodel
```

The above command also contains the following steps:

- Generate MLIR int8 quantized model, run the inference of MLIR int8 quantized model, and compare with the result of MLIR fp32 model
- Generate cvimodel, call the simulator to run the inference result, compare the result with the MLIR int8 quantized model result`

**Note:** `--tolerance` indicates the error tolerance of the similarity between the MLIR int8 quantized model and the MLIR fp32 model inference result, `--correctnetss` indicates the error tolerance of the similarity between the result of the simulator and the result of the MLIR int8 model, `--chip` can choose `cv183x` and `cv182x` to use `cv183x` by default

<div STYLE="page-break-after: always;"></div>

## 9 Compile and transplant tensorflow 1.x model

The TPU tool chain converts the Tensorflow 1.x model to the onnx model.

This chapter takes `mobilenet_v1_0.25` as an example to introduce how to compile and migrate a tensorflow 1.x model to run on the CV183x TPU platform.

 The following documents are required for this chapter:

* cvitek_mlir.tar.gz
* dataset.tar.gz



#### Step 0: Load the cvitek_mlir environment

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### Step 1: Obtain the tensorflow model and convert it to the onnx model

Use the `mobilenet_v1_0.25_224` model provided by tensorflow, see:

<https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md>

Download link:

<http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz>

First open `mobilenet_v1_0.25_224_eval.pbtxt`, find the output node name is `MobilenetV1/Predictions/Reshape_1`,
Use the following command to convert to onnx model:

``` shell
mkdir model_mnet_25 && cd model_mnet_25
wget -nc \
http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz
tar zxf mobilenet_v1_0.25_224.tgz
pip install tf2onnx

python3 -m tf2onnx.convert --graphdef mobilenet_v1_0.25_224_frozen.pb --output mnet_25.onnx --inputs input:0 --outputs MobilenetV1/Predictions/Reshape_1:0
```

Get `mnet_25.onnx`.

However, because the tensorflow model uses NHWC as the input by default, after being converted to the onnx model, it is still input in NHWC format and connected to a transpose node. Before compiling, we first convert the input format and remove the transpose node. Use the following python script to proceed:

``` python
import onnx

model = onnx.load('mnet_25.onnx')

print(model.graph.input[0].type.tensor_type.shape.dim)

model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224

print(model.graph.input[0].type.tensor_type.shape.dim)

input_name = model.graph.input[0].name

del model.graph.node[0]

model.graph.node[0].input[0] = input_name

onnx.save(model, 'mnet_25_new.onnx')
```

Get `mnet_25_new.onnx`。

#### Step 2: Model conversion

Create a workspace and get a test picture. This example uses `cat.jpg`contained in `cvitek_mlir`

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

The preprocessing parameters are as follows:

> RAW_SCALE=255
>
> MODEL_CHANNEL_ORDER="rgb"
>
> MEAN=127.5,127.5,127.5 # in RGB
>
> STD=127.5,127.5,127.5
>
> INPUT_SCALE=1.0

Convert onnx model to mlir file

``` shell
model_transform.py \
  --model_type onnx \
  --model_name mnet_25 \
  --model_def ../mnet_25_new.onnx \
  --image ./cat.jpg \
  --image_resize_dims 256,256 \
  --keep_aspect_ratio false \
  --net_input_dims 224,224 \
  --raw_scale 255.0 \
  --mean 127.5,127.5,127.5 \
  --std 127.5,127.5,127.5 \
  --input_scale 1.0 \
  --model_channel_order "rgb" \
  --gray false \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --mlir mnet_25_fp32.mlir
```

Get `mnet_25_fp32.mlir`.

The conversion process includes:

- Inference of the original tensorflow model, and save the results of each layer as a numpy npz file

- Import of the original tensorflow model, convert the original model into an MLIR fp32 model
  - Execute the inference of the MLIR fp32 model, and save the output of each layer to the numpy npz file
  - Compare the inference results of the caffe model with the inference results of MLIR fp32 to ensure that the converted MLIR fp32 model is correct
  - Optimize the MLIR fp32 model as the input of the subsequent process

**Note:** The preprocessing parameters entered above are only stored in mlir in the form of information, and subsequently converted to cvimodel, they are also only stored in the form of information. The image preprocessing process needs to be processed externally and then passed to the model for calculation. If you need to pre-process the pictures inside the model, please refer to Chapter 12: Using TPU for pre-processing.

#### Step 4: Calibration

Before Calibration, you need to prepare a calibration picture set. The number of pictures is about 100~1000 according to the situation.
Execute calibration：

``` shell
run_calibration.py \
  mnet_25_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mnet_25_calibration_table
```

  Get `mnet_25_calibration_table`。

#### Step 5: Model quantification and generate cvimodel

``` shell
model_deploy.py \
  --model_name mnet_25 \
  --mlir mnet_25_fp32.mlir \
  --calibration_table mnet_25_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.93,0.90,0.62 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mnet_25.cvimodel
```

The above command also contains the following steps:

- Generate the MLIR int8 quantitative model, run the inference of the MLIR int8 quantitative model, and compare with the results of the MLIR fp32 model
- Generate cvimodel, call the simulator to run the inference result, compare the result with the MLIR int8 quantized model result

**Note:** `--tolerance` indicates the error tolerance of the similarity between the MLIR int8 quantized model and the MLIR fp32 model inference result, `--correctnetss` indicates the error tolerance of the similarity between the result of the simulator and the result of the MLIR int8 model, `--chip` can choose `cv183x` and `cv182x` to use `cv183x` by default

The quantized mlir model file `mnet_25_quantized.mlir` will be generated in the previous step. You can use the pymlir python interface to test the accuracy:

``` shell
# FP32
eval_classifier.py \
    --mlir_file=mnet_25_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --label_file=$REGRESSION_PATH/data/synset_words.txt \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 127.5,127.5,127.5 \
    --std 127.5,127.5,127.5 \
    --data_format "nchw" \
    --model_channel_order "rgb" \
    --model_type mlir \
    --count=50000

# INT8
eval_classifier.py \
    --mlir_file=mnet_25_quantized.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --label_file=$REGRESSION_PATH/data/synset_words.txt \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 127.5,127.5,127.5 \
    --std 127.5,127.5,127.5 \
    --data_format "nchw" \
    --model_channel_order "rgb" \
    --model_type mlir \
    --count=50000
```

<div STYLE="page-break-after: always;"></div>

## 10 Compile and transplant the tflite model

This chapter takes `resnet50` as an example to introduce how to compile and migrate a tflite model to run on the CV183x TPU platform.

The following documents are required for this chapter:

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz



#### Step 0: Load the cvitek_mlir environment

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### Step 1: Obtain the tensorflow model and convert it to a tflite model

(If you use the prepared tflite model directly, this step can be omitted)

Create a model directory:

``` shell
mkdir resnet50_tflite && cd resnet50_tflite
```

Use the following python script to download the tensorflow model and convert it to a tflite model:

``` python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

os.environ['CUDA_VISIBLE_DEVICES'] = ''

data_path = os.environ['DATASET_PATH'] + '/imagenet/img_val_extracted/val/'

def representative_data_gen():
    class_path = os.path.join(data_path)
    all_class = os.listdir(class_path)
    for i in all_class[:100]: # data numbers
        imgs_name = os.listdir('{}/{}'.format(class_path, i))
        for img_name in imgs_name:
            image_path = os.path.join('{}/{}'.format(class_path, i), img_name)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            print(img_name)
            yield [x]
            # take one image from each class
            break

def main():
    # tf._logging.set_verbosity(tf._logging.INFO)
    model = tf.keras.applications.ResNet50(
        weights='imagenet', input_shape=(224, 224, 3))
    model.save('resnet50_model', save_format='tf')
    converter = tf.lite.TFLiteConverter.from_saved_model('resnet50_model')
    tflite_model = converter.convert()
    open('resnet50.tflite', 'wb').write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()
    open('resnet50_int8_quant.tflite', 'wb').write(tflite_model)

if __name__ == '__main__':
    main()
```

Get the corresponding `resnet50_int8_quant.tflite` model.

#### Step 2: Perform tflite inference (Optional)

Create a working directory and obtain a test picture. This example uses cat.jpg contained in cvitek_mlir, and uses the following script to generate input data for interpter:

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
# 进入python后台
```

Execute the python command as follows:

``` python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np

img = image.load_img('cat.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = np.ascontiguousarray(np.transpose(x, (0,3,1,2)))
np.savez('resnet50_in_fp32.npz', input=x)
```

Run tflite inference:

``` shell
cvi_model_inference.py \
    --input_file cat.jpg \
    --output_file resnet50_out_ref.npz \
    --model_def ../resnet50_int8_quant.tflite \
    --net_input_dims 224,224 \
    --image_resize_dims 256,256 \
    --model_type=tflite \
    --raw_scale 255.0 \
    --mean 103.939,116.779,123.68 \
    --data_format nhwc \
    --input_scale 1.0
```

Get `resnet50_out_ref.npz`.

#### Step 3: Convert to mlir and Execute front-end optimization

Execute conversion and front-end optimization:

``` shell
cvi_model_convert.py  \
    --model_path ../resnet50_int8_quant.tflite \
    --model_name resnet50 \
    --model_type tflite_int8  \
    --batch_size 1 \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 103.939,116.779,123.68 \
    --input_scale 1.0 \
    --mlir_file_path resnet50_int8.mlir

tpuc-opt resnet50_int8.mlir \
    --assign-chip-name \
    --chipname cv183x \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv \
    -o resnet50_int8_opt.mlir
```

Get `resnet50_int8_opt.mlir`.

Run tpuc-interpreter to reason about mlir and get the layer-by-layer data:

``` shell
# inference with mlir and input data, dump all tensor
tpuc-interpreter resnet50_int8_opt.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8.npz \
    --dump-all-tensor=resnet50_tensor_all_int8.npz
```

Get `resnet50_out_int8.npz`.

#### Step 4: Generate cvimodel

The input of this model is an int8 model, no calibraion is required.

``` shell
mlir_to_cvimodel.sh -i resnet50_int8_opt.mlir -o resnet50.cvimodel

model_runner \
    --input resnet50_in_fp32.npz \
    --model resnet50.cvimodel \
    --output out.npz

# check output data
cvi_npz_tool.py dump out.npz
cvi_npz_tool.py dump out.npz Identity_int8 5
# Show Top-K 5
# (277, 0.5969119)
# (278, 0.23570427)
# (356, 0.07378012)
# (263, 0.036752112)
# (287, 0.023094626)
```

Test accuracy:

``` shell
# INT8
eval_classifier.py \
    --mlir_file=resnet50_int8_opt.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims 224,224 \
    --model_type=mlir \
    --raw_scale 255.0 \
    --mean 103.939,116.779,123.68 \
    --input_scale 1 \
    --count=50000
```

<div STYLE="page-break-after: always;"></div>

## 11  Guidelines for precision optimization and mixed quantization

CV183X TPU supports two quantization methods, INT8 and BF16. In the model compilation stage, the tool chain supports searching to find the ops that are most sensitive to the accuracy of the model, and supports the replacement of a specified number of ops with BF16, thereby improving the accuracy of the entire network.

This chapter takes `mobilenet_v1_0.25` as an example to introduce how to use automatic search and mixed precision methods for this model to improve the accuracy of the model.

The following documents are required for this chapter:

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz

#### Step 0: Obtain the tensorflow model and convert it to the onnx model

This is the same as Chapter 9.

Use the `mobilenet_v1_0.25_224` model provided by tensorflow, see: <https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md>

Download link: <http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz>

First open `mobilenet_v1_0.25_224_eval.pbtxt`, find the output node name is `MobilenetV1/Predictions/Reshape_1`,
Use the following command to convert to onnx model:

``` shell
source cvitek_mlir/cvitek_envs.sh
pip install tf2onnx
mkdir model_mnet_25 && cd model_mnet_25
wget -nc http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz
tar xzf mobilenet_v1_0.25_224.tgz
python3 -m tf2onnx.convert --graphdef mobilenet_v1_0.25_224_frozen.pb \
    --output mnet_25.onnx --inputs input:0 \
    --outputs MobilenetV1/Predictions/Reshape_1:0
```

Get `mnet_25.onnx`.

However, because the tensorflow model uses NHWC as the input by default, after being converted to the onnx model, it is still input in NHWC format and connected to a transpose node. Before compiling, we first convert the input format and remove the transpose node. Use the following python script to proceed:

``` python
import onnx

model = onnx.load('mnet_25.onnx')

print(model.graph.input[0].type.tensor_type.shape.dim)

model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224

print(model.graph.input[0].type.tensor_type.shape.dim)

input_name = model.graph.input[0].name

del model.graph.node[0]

model.graph.node[0].input[0] = input_name

onnx.save(model, 'mnet_25_new.onnx')
```

Get `mnet_25_new.onnx`.

#### Step 1: Model conversion

Get a test picture, this example uses cat.jpg contained in cvitek_mlir:

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

The preprocessing parameters are as follows:

> RAW_SCALE=255
>
> MODEL_CHANNEL_ORDER="rgb"
>
> MEAN=127.5,127.5,127.5 # in RGB
>
> STD=127.5,127.5,127.5
>
> INPUT_SCALE=1.0

Convert to mlir file:

``` shell
model_transform.py \
  --model_type onnx \
  --model_name mnet_25 \
  --model_def ../mnet_25_new.onnx \
  --image ./cat.jpg \
  --image_resize_dims 256,256 \
  --keep_aspect_ratio false \
  --net_input_dims 224,224 \
  --raw_scale 255.0 \
  --mean 127.5,127.5,127.5 \
  --std 127.5,127.5,127.5 \
  --input_scale 1.0 \
  --model_channel_order "rgb" \
  --gray false \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --mlir mnet_25_fp32.mlir
```

Get `mnet_25_fp32.mlir`.


#### Step 2: Test the accuracy of the FP32 model (Optional)

Use the pymlir python interface to test the accuracy:

``` shell
# FP32
eval_classifier.py \
    --mlir_file=mnet_25_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --label_file=$REGRESSION_PATH/data/synset_words.txt \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 127.5,127.5,127.5 \
    --std 127.5,127.5,127.5 \
    --data_format "nchw" \
    --model_channel_order "rgb" \
    --model_type mlir \
    --count=50000

# ......
# Test: [49900/50000]     Time  0.042 ( 0.056)    Loss 5.9356e+00 (6.5670e+00)    Acc@1 100.00 ( 49.15)       Acc@5 100.00 ( 73.46)
# Test: [49950/50000]     Time  0.041 ( 0.056)    Loss 6.8937e+00 (6.5669e+00)    Acc@1   0.00 ( 49.16)       Acc@5   0.00 ( 73.47)
# Test: [50000/50000]     Time  0.043 ( 0.056)    Loss 6.8896e+00 (6.5669e+00)    Acc@1   0.00 ( 49.16)       Acc@5   0.00 ( 73.47)
# * Acc@1 49.164 Acc@5 73.468
# tensor(49.1640)
```

According to the test, the accuracy of the FP32 model is Top-1 49.2% Top-5 73.5%.

#### Step 4: Execute INT8 quantization

Do calibration：

``` shell
run_calibration.py \
    mnet_25_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --input_num=1000 \
    --calibration_table mnet_25_calibration_table
```

Get`mnet_25_calibration_table`.

Execute INT8 quantization and compare layer by layer:

``` shell
model_deploy.py \
  --model_name mnet_25 \
  --mlir mnet_25_fp32.mlir \
  --calibration_table mnet_25_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.93,0.90,0.62 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mnet_25.cvimodel.npz
```

#### Step 5: Test the accuracy of the INT8 model (Optional)

The quantized mlir model file `mnet_25_quantized.mlir` will be generated in the previous step. You can use the pymlir python interface to test the accuracy:

``` shell
# INT8
eval_classifier.py \
    --mlir_file=mnet_25_quantized.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --label_file=$REGRESSION_PATH/data/synset_words.txt \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 127.5,127.5,127.5 \
    --std 127.5,127.5,127.5 \
    --data_format "nchw" \
    --model_channel_order "rgb" \
    --model_type mlir \
    --count=50000

# ......
# Test: [49900/50000]     Time  0.088 ( 0.078)    Loss 5.9236e+00 (6.6264e+00)    Acc@1 100.00 ( 43.19)   Acc@5 100.00 ( 68.34)
# Test: [49950/50000]     Time  0.041 ( 0.078)    Loss 6.9057e+00 (6.6264e+00)    Acc@1   0.00 ( 43.18)   Acc@5   0.00 ( 68.33)
# Test: [50000/50000]     Time  0.081 ( 0.078)    Loss 6.9052e+00 (6.6264e+00)    Acc@1   0.00 ( 43.18)   Acc@5   0.00 ( 68.32)
# * Acc@1 43.176 Acc@5 68.318
```

The test shows that the accuracy of the INT8 model is Top-1 43.2% Top-5 68.3%, which is somewhat lower than the accuracy of the FP32 model (Top-1 49.2% Top-5 73.5%).

#### Step 6: Execute mix quantization search and execute mix quantization

Search the mix quantization table. This model has 59 layers. How many layers are selected for replacement can be adjusted according to the need for accuracy and the accuracy results of the test. The number of data sets for searching can also be adjusted as needed.

 Here is an example of replacing 6 layers (`--max_bf16_layers=6`), and the test data set for searching is 100 sheets:

``` shell
run_mix_precision.py \
    mnet_25_fp32.mlir \
    --model_name mnet25 \
    --dataset ${DATESET_PATH} \
    --calibration_table mnet_25_calibration_table \
    --input_num=20 \
    --max_bf16_layers=6 \
    -o mnet_25_mix_precision_bf16_table
```

Get `mnet_25_mix_precision_bf16_table`, the content is as follows:

``` shell
cat mnet_25_mix_precision_bf16_table
# MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:0_Clip
# MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0_Clip
# MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0_relu6_reluClip
```

Execute mix quantification：

``` shell
model_deploy.py \
  --model_name mnet_25 \
  --mlir mnet_25_fp32.mlir \
  --calibration_table mnet_25_calibration_table \
  --mix_precision_table mnet_25_mix_precision_bf16_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.94,0.93,0.67 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mnet_25_mix_precision.cvimodel
```

#### Step 7: Test the accuracy of the mixed quantitative model (Optional)

The quantized mlir model file `mnet_25_quantized.mlir` will be generated in the previous step. You can use the pymlir python interface to test the accuracy:

``` shell
# MIXED, 6 layers
eval_classifier.py \
    --mlir_file=mnet_25_quantized.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --label_file=$REGRESSION_PATH/data/synset_words.txt \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --raw_scale 255.0 \
    --mean 127.5,127.5,127.5 \
    --std 127.5,127.5,127.5 \
    --data_format "nchw" \
    --model_channel_order "rgb" \
    --model_type mlir \
    --count=50000
# Test: [49900/50000]     Time  0.052 ( 0.064)    Loss 6.8954e+00 (6.5889e+00)    Acc@1   0.00 ( 47.40)   Acc@5   0.00 ( 72.28)
# Test: [49950/50000]     Time  0.044 ( 0.064)    Loss 6.2587e+00 (6.5890e+00)    Acc@1 100.00 ( 47.39)   Acc@5 100.00 ( 72.29)
# Test: [50000/50000]     Time  0.053 ( 0.064)    Loss 5.9630e+00 (6.5890e+00)    Acc@1 100.00 ( 47.40)   Acc@5 100.00 ( 72.29)
#  * Acc@1 47.400 Acc@5 72.292
```

According to the test, the accuracy of the mix quantitative T8 model is Top-1 47.4% Top-5 72.3%.

To compare the effect, we adjust the `number_bf16` parameters to 10 and 15, repeat the above test (specific commands are omitted), the results are respectively.

``` shell
# MIXED, 10 layers

# Test: [49950/50000]     Time  0.041 ( 0.044)    Loss 6.8057e+00 (6.5866e+00)    Acc@1   0.00 ( 47.64)   Acc@5 100.00 ( 72.28)
# Test: [50000/50000]     Time  0.040 ( 0.044)    Loss 6.2855e+00 (6.5865e+00)    Acc@1 100.00 ( 47.65)   Acc@5 100.00 ( 72.29)
#  * Acc@1 47.648 Acc@5 72.294

# MIXED, 15 layers

# Test: [49950/50000]     Time  0.049 ( 0.044)    Loss 5.9641e+00 (6.5852e+00)    Acc@1 100.00 ( 47.78)   Acc@5 100.00 ( 72.51)
# Test: [50000/50000]     Time  0.043 ( 0.044)    Loss 6.3955e+00 (6.5852e+00)    Acc@1 100.00 ( 47.78)   Acc@5 100.00 ( 72.52)
#  * Acc@1 47.782 Acc@5 72.518
```

Full bf16 quantified measurement:

``` shell
model_deploy.py \
  --model_name mnet_25 \
  --mlir mnet_25_fp32.mlir \
  --all_bf16 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.99,0.99,0.86 \
  --correctness 0.99,0.99,0.94 \
  --cvimodel mnet_25_all_bf16_precision.cvimodel

# BF16
# Test: [49950/50000]     Time  0.031 ( 0.036)    Loss 6.4377e+00 (6.5711e+00)    Acc@1 100.00 ( 48.49)   Acc@5 100.00 ( 73.06)
# Test: [50000/50000]     Time  0.033 ( 0.036)    Loss 6.8726e+00 (6.5711e+00)    Acc@1   0.00 ( 48.50)   Acc@5 100.00 ( 73.06)
# * Acc@1 48.498 Acc@5 73.064
```

Compare the results of 6 quantization methods (mix quantization includes 6-layer, 10-layer and 15-layer versions):

| **Quant Type**    | **Top-1** | **Top-5** |
| ----------------- | --------- | --------- |
| INT8              | 43.2%     | 68.3%     |
| MIXED (6 layers)  | 47.4%     | 72.3%     |
| MIXED (10 layers) | 47.6%     | 72.3%     |
| MIXED (15 layers) | 47.8%     | 72.5%     |
| BF16              | 48.5%     | 73.1%     |
| FP32              | 49.2%     | 73.5%     |

<div STYLE="page-break-after: always;"></div>

## 12 Use TPU for preprocessing

CV183X provides two hardware resources to accelerate the pre-processing of neural network models.

* Use VPSS: VPSS is a video post-processing module provided by CV18xx, and expands the pre-processing function of the neural network, so that the video processing pipeline outputs pre-processed image data, which can be directly used as neural network input data.

* Use TPU: TPU can also be used to support common pre-processing calculations, including raw_scale, mean, input_scale, channel swap, split, and quantization. Developers can pass the corresponding pre-processing parameters through the compilation options during the model compilation stage, and the compiler directly inserts the corresponding pre-processing operators before the model is transported. The generated cvimodel can directly use the pre-processed image as input and reason with the model The process uses TPU to process pre-processing operations.

Customers can flexibly choose which engine to use for preprocessing based on system optimization needs. Please refer to "CV18xx Media Software Development Reference" for the detailed usage method of preprocessing using VPSS. This document does not introduce it. This chapter introduces the specific steps of using TPU for pre-processing. This chapter takes Caffe model compilation as an example. Follow the steps in Chapter 6 to make slight modifications to generate a cvimodel that supports pre-processing. Take `mobilenet_v2` as an example.

#### Step 0-3: Same as the corresponding steps in the Caffe chapter

Assuming that the user and following the steps described in Chapter 6, after performing the model conversion and generating the calibraiton table.

#### Step 4: Model quantification and generate cvimodel with TPU preprocessing

First, load the cvitek_mlir environment:

``` shell
source cvitek_mlir/cvitek_envs.sh
cd models_mobilenet_v2/workspace
```

Execute the following commands:

``` shell
model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.96,0.96,0.71 \
  --fuse_preprocess \
  --pixel_format BGR_PACKED \
  --aligned_input false \
  --excepts data \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mobilenet_v2_fused_preprocess.cvimodel
```

Get the cvimodel with pre-processing.

Among them, `pixel_format` is used to specify the input data format, there are several formats:

| pixel_format  | Description                                            |
| ------------- | ------------------------------------------------------ |
| RGB_PLANAR    | rgb order, arranged according to nchw                  |
| RGB_PACKED    | rgb order, arranged according to nhwc                  |
| BGR_PLANAR    | bgr order, arranged according to nchw                  |
| BGR_PACKED    | bgr order, arranged according to nhwc                  |
| GRAYSCALE     | There is only one gray channel, press nchw to place it |
| YUV420_PLANAR | yuv420 format, arranged according to nchw              |

Among them, `aligned_input` is used to indicate whether the data is aligned. If the data comes from VPSS, there will be data alignment requirements, for example, w is aligned according to 32 bytes.

The above process includes the following steps:

- Generate MLIR int8 model with pre-processing, and input without pre-processing mobilenet_v2_resized_only_in_fp32.npz
- Execute comparison between MLIR int8 inference and MLIR fp32 inference results to verify the correctness of MLIR int8 with pre-processing model
- Generate a cvimodel with pre-processing, and call the simulator to execute inference, and compare the result with the inference result of the MLIR int8 model with pre-processing
