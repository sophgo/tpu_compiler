![image](./assets/logo_0.png)

# CV183x/CV182x TPU 快速入门指南




>文档版本: 1.5.0
>
>发布日期: 2021-01-29



© 2020 北京晶视智能科技有限公司

本文件所含信息归<u>北京晶视智能科技有限公司</u>所有。

未经授权，严禁全部或部分复制或披露该等信息。

<div STYLE="page-break-after: always;"></div>

## 修订记录

| 版本    | 日期       | 修改人      | 修改描述                                       |
| ------- | ---------- | ----------- | ---------------------------------------------- |
| V0.0.1  | 2019/12/11 | 王雷        | 初始版本                                       |
| V0.1.0  | 2020/04/18 | 王雷        | 增加使用说明                                   |
| V0.1.1  | 2020/04/20 | 王雷        | 更新测试命令                                   |
| V0.1.2  | 2020/04/28 | 王雷        | 更新模型精度测试命令                           |
| V0.2.0  | 2020/04/30 | 王雷        | 修订                                           |
| V0.2.1  | 2020/05/15 | 王雷        | 根据V0.9 SDK更新部分命令                       |
| V0.2.2  | 2020/06/01 | 王雷        | 增加端到端推理性能测试命令                     |
| V0.3.0  | 2020/06/25 | 王雷        | 根据V1.0 SDK修订增加移植编译TensorFlow 2.x模型 |
| V0.3.1  | 2020/06/29 | 王雷        | 采用python importer进行caffe移植               |
| V0.3.2  | 2020/07/17 | 李全        | 增加第9章使用TPU进行前处理                     |
| V0.3.3  | 2020/07/19 | 王雷        | 根据V1.1 SDK修订                               |
| V0.3.4  | 2020/07/20 | 王雷        | 修订                                           |
| V0.3.5  | 2020/07/29 | 王雷        | 增加移植编译TensorFlow 2.x模型                 |
| V0.3.6  | 2020/08/08 | 王雷        | 增加精度优化和混合量化指南                     |
| V0.3.7  | 2020/08/12 | 王雷        | 更新使用TPU进行前处理流程                      |
| V0.3.8  | 2020/09/06 | 王雷        | 根据V1.2 SDK修订                               |
| V0.3.9  | 2020/09/22 | 李全        | 增加移植tflite模型                             |
| V0.3.10 | 2020/09/30 | 郑伟圣      | 更新移植tflite模型                             |
| V0.3.11 | 2020/10/26 | 肖泉/胡鹏超 | 根据V1.3 SDK修订                               |
| V1.4.0  | 2020/12/07 | 肖泉/胡鹏超 | 根据V1.4 SDK修订                               |
| V1.5.0  | 2021/01/29 | 肖泉/胡鹏超 | 根据V1.5 SDK修订                               |

<div STYLE="page-break-after: always;"></div>

## 法律声明

本数据手册包含北京晶视智能科技有限公司（下称"晶视智能"）的保密信息。未经授权，禁止使用或披露本数据手册中包含的信息。如您未经授权披露全部或部分保密信息，导致晶视智能遭受任何损失或损害，您应对因之产生的损失/损害承担责任。

本文件内信息如有更改，恕不另行通知。晶视智能不对使用或依赖本文件所含信息承担任何责任。

本数据手册和本文件所含的所有信息均按"原样"提供，无任何明示、暗示、法定或其他形式的保证。晶视智能特别声明未做任何适销性、非侵权性和特定用途适用性的默示保证，亦对本数据手册所使用、包含或提供的任何第三方的软件不提供任何保证；用户同意仅向该第三方寻求与此相关的任何保证索赔。此外，晶视智能亦不对任何其根据用户规格或符合特定标准或公开讨论而制作的可交付成果承担责任。

<div STYLE="page-break-after: always;"></div>

##  目录

* content
{:toc}



<div STYLE="page-break-after: always;"></div>

## 1 概述

#### 1.1 阅读说明

本文档包含下述章节，请根据需要参阅相关章节。

* 运行测试

  不需编译，在EVB运行随release提供的sample程序和模型，包括：

  * 执行samples程序
  * 对测试cvimodel进行正确性和性能测试

* 开发环境配置

  使用CVITEK提供的docker，配置编译开发所需的环境

* 编译samples程序

  介绍如何交叉编译sample应用程序，调用runtime API完成推理任务。具体包括4个samples：

  * Sample-1 : classifier (mobilenet_v2)

  * Sample-2 : detector (yolo_v3)

  * Sample-3 : alphapose (yolo_v3 + fastpose)

  * Sample-4 : insightface (retinaface + arcface)

* 编译生成cvimodel

  介绍如何通过脚本生成所有sample用和测试用的cvimodel

* 编译移植caffe模型

  介绍如何移植一个新的caffe模型，以`mobilenet_v2`为例

* 编译移植pytorch模型

  介绍如何移植一个新的pytorch模型，以`resnet18`为例

* 编译移植tensorflow 2.x模型

  介绍如何移植一个新的tensorflow 2.x模型，以`mobilenet_v2`为例

* 编译移植tensorflow 1.x模型

  介绍如何移植一个新的tensorflow 1.x模型，以`mobilenet_v1_0.25_224`为例

* 使用TPU进行前处理

  介绍如何在cvimodel模型中增加前处理描述，并在运行时使用TPU进行前处理



#### 1.2 Release 内容

CVITEK Release包含如下组成部分：

| 文件                                                    | 描述                                             |
| ------------------------------------------------------- | ------------------------------------------------ |
| cvitek_mlir_ubuntu-18.04.tar.gz                         | cvitek NN工具链软件                              |
| cvitek_tpu_sdk_[cv182x/cv183x].tar.gz                 | cvitek Runtime SDK，包括交叉编译头文件和库文件   |
| cvitek_tpu_samples.tar.gz                               | sample程序源代码                                 |
| cvimodel_samples_[cv182x/cv183x].tar.gz               | sample程序使用的cvimodel模型文件                 |
| cvimodel_regression_bs1_[cv182x/cv183x].tar.gz | 模型测试cvimodel文件和相应输入输出数据文件       |
| cvimodel_regression_bs4_[cv182x/cv183x].tar.gz | 模型测试cvimodel文件和相应输入输出数据文件       |
| docker_cvitek_dev.tar                                   | CVITEK开发Docker镜像文件                         |
| models.tar.gz                                           | 测试用caffe/onnx原始模型文件包（支持github下载） |
| dataset.tar.gz                                          | 测试用dataset包（可github下载，参考REAMDE准备）  |



#### 1.3 Models 和 Dataset

测试用的原始框架模型文件和dataset可以由下列链接取得，并参考README.md描述进行相应准备。

* <https://github.com/cvitek-mlir/models>

* <https://github.com/cvitek-mlir/dataset>

<div STYLE="page-break-after: always;"></div>

## 2 运行测试

不需编译，在EVB运行release提供的sample预编译程序和模型。

本章需要如下文件：

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvimodel_samples_[cv182x/cv183x].tar.gz
* cvimodel_regression_bs1_[cv182x/cv183x].tar.gz
* cvimodel_regression_bs4_[cv182x/cv183x].tar.gz



#### 2.1 运行sample程序

将根据chip类型选择所需文件加载至EVB的文件系统，于evb上的linux console执行，以cv183x为例：

 解压samples使用的model文件（以cvimodel格式交付），并解压TPU_SDK，并进入samples目录，执行测试，过程如下：

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

同时提供脚本作为参考，执行效果与直接运行相同，如下：

``` evb_shell
./run_classifier.sh
./run_detector.sh
./run_alphapose.sh
./run_insightface.sh
```

也有使用preprocess（预处理）的脚本作为参考，如下：

``` evb_shell
./run_classifier_fused_preprocess.sh
./run_detector_fused_preprocess.sh
./run_alphapose_fused_preprocess.sh
./run_insightface_fused_preprocess.sh
```



#### 2.2 测试cvimodel

在EVB执行脚本regression_models.sh，该脚本对每个网络调用model_runner进行推理运算，比对输出数据是否正确，同时打印运行时间信息。

* 基于PMU数据的Inference性能测试

  Regression模型文件分成bs=1和bs=4两部分，分别执行测试，对所有网络进行正确性和运行效率测试。以cv183x平台为例：

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



* 基于系统时钟的端到端性能测试

  计入数据输入，后处理和数据导出时间在内的端到端网络推理时间，以cv183x平台为例：

  ``` evb_shell
  cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..
  export TPU_ROOT=$PWD/cvitek_tpu_sdk
  export PATH=$TPU_ROOT/samples/bin:$PATH

  tar zxf cvimodel_regression_bs1_cv183x.tar.gz
  MODEL_PATH=$PWD/cvimodel_regression_bs1 $TPU_ROOT/regression_models_e2e.sh
  ```



#### 2.3 当前支持测试的网络列表

cv183x支持的网络如下：

| Classification                                               | Detection                                                    | Misc                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| resnet50       [BS=1,4] <br />resnet18       [BS=1,4]<br />mobilenet_v1     [BS=1,4]<br />mobilenet_v2     [BS=1,4]<br />squeezenet_v1.1    [BS=1,4]<br />shufflenet_v2     [BS=1,4]<br />googlenet       [BS=1,4]<br />inception_v3     [BS=1,4]<br />inception_v4     [BS=1,4]<br />vgg16         [BS=1,4]<br />densenet_121     [BS=1,4]<br />densenet_201     [BS=1,4]<br />senet_res50      [BS=1,4]<br />resnext50       [BS=1,4]<br />res2net50       [BS=1,4]<br />ecanet50       [BS=1,4]<br />efficientnet_b0    [BS=1,4]<br />efficientnet_lite_b0 [BS=1,4]<br />nasnet_mobile     [BS=1,4] | retinaface_mnet25 [BS=1,4]<br />retinaface_res50   [BS=1]<br />ssd300        [BS=1,4]<br />mobilenet_ssd [BS=1,4]<br />yolo_v1_448      [BS=1]<br />yolo_v2_416      [BS=1]<br />yolo_v2_1080     [BS=1]<br />yolo_v3_416      [BS=1,4]<br />yolo_v3_608      [BS=1]<br />yolo_v3_tiny     [BS=1]<br />yolo_v3_spp      [BS=1]<br />yolo_v4        [BS=1] | arcface_res50 [BS=1,4]<br />alphapose       [BS=1,4]<br />espcn_3x       [BS=1,4]<br />unet          [BS=1,4]<br />erfnet         [BS=1] |

cv182x支持的网络如下：

| Classification                                               | Detection                                                    | Misc                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| resnet50       [BS=1,4] <br />resnet18       [BS=1,4]<br />mobilenet_v1     [BS=1,4]<br />mobilenet_v2     [BS=1,4]<br />squeezenet_v1.1    [BS=1,4]<br />shufflenet_v2     [BS=1,4]<br />googlenet       [BS=1,4]<br />inception_v3     [BS=1]<br />densenet_121     [BS=1,4]<br />densenet_201     [BS=1]<br />senet_res50      [BS=1]<br />resnext50       [BS=1,4]<br />efficientnet_lite_b0 [BS=1,4]<br />nasnet_mobile     [BS=1] | retinaface_mnet25 [BS=1,4]<br />retinaface_res50   [BS=1]<br />mobilenet_ssd [BS=1,4]<br />yolo_v1_448      [BS=1]<br />yolo_v2_416      [BS=1]<br />yolo_v3_416      [BS=1,4]<br />yolo_v3_608      [BS=1]<br />yolo_v3_tiny     [BS=1]<br /> | arcface_res50 [BS=1,4]<br />alphapose       [BS=1,4]<br /> |



**注：** BS表示batch，[BS=1]表示板子目前至少batch 1，[BS=1,4]表示板子至少支持batch 1和batch 4。

<div STYLE="page-break-after: always;"></div>

## 3 开发环境配置

从docker hub获取（推荐）:

```
docker pull cvitek/cvitek_dev:1.4-ubuntu-18.04
```

或者加载镜像文件：

```
docker load -i docker_cvitek_dev_1.4-ubuntu-18.04.tar
```



如果是首次使用docker，可执行下述命令进行安装和配置（Ubuntu系统）

```
sudo apt install docker.io
systemctl start docker
systemctl enable docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker (use before reboot)
```



取得docker image后，执行下述命令运行docker：

```
# 这里假设models和dataset分别位于~/data/models和~/data/dataset目录，如有不同请相应调整。
docker run -itd -v $PWD:/work \
   -v ~/data/models:/work/models \
   -v ~/data/dataset:/work/dataset \
   --name cvitek cvitek/cvitek_dev:1.4-ubuntu-18.04
docker exec -it cvitek bash
```

<div STYLE="page-break-after: always;"></div>

## 4 编译samples程序

请根据chip类型选择使用对应的TPU sdk对samples code做交叉编译，加载至evb上并运行测试。

本节需要如下文件：

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvitek_tpu_samples.tar.gz


下面以cv183x平台为例. TPU sdk准备：

``` shell
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
```

编译samples，安装至install_samples目录：

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

<div STYLE="page-break-after: always;"></div>

## 5 编译生成测试用cvimodel

本节需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* models.tar.gz
* dataset.tar.gz

#### 5.1 调用脚本生成cvimodel

准备TPU仿真开发环境：

```
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```

 使用下述脚本命令，快速生成所有测试用的cvimodel文件：

```
generate_all_cvimodels.sh
```

 生成regression_out/cvimodel_release目录，cvimodel_samples所包含的模型是此处生成cvimodel_release的子集。



#### 5.2 运行回归测试生成cvimodel和输入输出数据

使用下述回归测试命令，对模型移植的各步骤结果进行比对和验证。同时也可以选择进行精度测试。

 回归测试结束，除生成所有测试cvimodel文件外，还同时生成各个模型的输入输出测试数据，用于加载至EVB进行模型测试。



准备TPU仿真开发环境：

```
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```



使用下述命令启动回归测试。测试网络分级为basic和extra以调节测试时间，用户也可以编辑run_regression.sh自行调节列表。

```
run_regression.sh    # basic  models only  Or
run_regression.sh -e  # with  extra models
```



生成的cvimodel_regression内容与release中cvimodel_regression.tar.gz内容一致，可以加载至EVB进行一致性和性能测试。

 用户也可以单独对其中一个网络进行回归测试，命令如下（以resnet50为例），所支持的网络列表参见2.2节。

```
regression_generic.sh resnet50
```



#### 5.3 测试模型精度

在执行run_regression.sh后，我们可以利用脚本对mlir模型进行精度测试，并与原始模型精度进行比较。命令为：

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

**注：** 需要准备imagenet或者coco的数据集，参见1.3节。

<div STYLE="page-break-after: always;"></div>

## 6 编译移植caffe模型

本章以mobilenet_v2为例，介绍如何编译迁移一个caffe模型至CV183x TPU平台运行; 如果需要切换到cv182x平台，可以通过命令行参数--chipname cv182x来指定。

 本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz

#### 步骤 0：加载cvitek_mlir环境

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取caffe模型

从<https://github.com/shicai/MobileNet-Caffe>下载模型，并保存在`models_mobilenet_v2`目录：

``` shell
mkdir models_mobilenet_v2 && cd models_mobilenet_v2
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt
```

创建工作目录workspace：

``` shell
mkdir workspace && cd workspace
```

#### 步骤 2：模型转换

使用`model_transform.py`将模型转换成mlir文件，其中有预处理参数如下：

| **参数名**          | **说明**                             |
| ------------------- | ------------------------------------ |
| image_resize_dims   | 表明图片resize大小，比如256,256      |
| keep_aspect_ratio   | 在Resize时是否保持长宽比 |
| net_input_dims      | 表明模型输入的大小，比如224,224      |
| model_channel_order | channel顺序，默认bgr；可以指定为rgb  |
| raw_scale           | 操作：* raw_scale/255.0，默认为255.0 |
| mean                | 操作：- mean，默认为0.0,0.0,0.0      |
| std                 | 操作：/std，默认为1.0,1.0,1.0        |
| input_scale         | 操作：* input_scale，默认为1.0       |

预处理过程用公式表达如下（x代表输入)：
$$
y = \frac{x \times \frac{raw\_scale}{255.0} - mean}{std} \times input\_scale
$$

由caffe模型转换为mlir：

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

得到`mobilenet_v2_fp32.mlir`文件.

其转换过程包括:

- 原始caffe模型的推理, 并将各层结果保存为numpy的npz文件

- 原始caffe模型的导入, 将原始模型转换成MLIR fp32模型
  - 执行MLIR fp32模型的推理, 将各层输出保存到numpy 的npz文件中
  - 将caffe模型的推理的结果与MLIR fp32的推理结果对比, 确保转换的MLIR fp32模型正确
  - 将MLIR fp32模型做优化，作为后续流程的输入

**注：** 上述填入的预处理参数仅仅以信息的形式存放在mlir中，后续转换成cvimodel，也仅以信息的方式存放。对图片的预处理过程需要再外部处理，再传给模型运算。如果需要模型内部对图片进行预处理，请参考12章节：使用TPU做前处理。

#### 步骤 3：Calibration

Calibration前需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。
执行calibration：

``` shell
run_calibration.py \
  mobilenet_v2_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mobilenet_v2_calibration_table
```

  得到`mobilenet_v2_calibration_table`。

#### 步骤 4：模型量化并生成cvimodel

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

以上命令包含以下几步:

- 生成MLIR int8模型, 运行MLIR量化模型的推理, 并与MLIR fp32模型的结果做比较
- 生成cvimodel, 并调用仿真器运行推理结果, 将结果与MLIR 量化模型做比较

**注：** --tolerance 表示 MLIR int8 量化模型与 MLIR fp32模型推理结果相似度的误差容忍度， --correctnetss 表示仿真器运行的结果与MLIR int8模型的结果相似度的误差容忍度, --chip 可以选择cv183x和cv182x 默认使用cv183x

<div STYLE="page-break-after: always;"></div>

## 7 编译移植pytorch模型

本章以resnet18为例，介绍如何编译迁移一个pytorch模型至CV183x TPU平台运行。

 本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz



#### 步骤 0：加载cvitek_mlir环境

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取pytorch模型并转换为onnx

使用torchvision提供的resnet18模型<https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>

使用下列python脚本下载pytorch模型，并将pytorch模型输出为onnx格式，保存在`model_resnet18`目录：

``` shell
mkdir model_resnet18
cd model_resnet18
```

执行python命令：

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

得到`resnet18.onnx`。

#### 步骤 2：模型转换

创建工作目录，取得一张测试用图片，本示例使用cvitek_mlir包含的cat.jpg

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

推理前，我们需要了解这个模型的预处理参数，resnet18的预处理如链接描述<https://pytorch.org/hub/pytorch_vision_resnet>：

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

使用`model_transform.py`将onnx模型转换成mlir文件：

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

得到resnet18_fp32.mlir文件.

其转换过程包括:

- 原始onnx模型的推理, 并将各层结果保存为numpy的npz文件

- 原始onnx模型的导入, 将原始模型转换成MLIR fp32模型
  - 执行MLIR fp32模型的推理, 将各层输出保存到numpy 的npz文件中
  - 将onnx模型的推理的结果与MLIR fp32的推理结果对比, 确保转换的MLIR fp32模型正确
  - 将MLIR fp32模型做优化，作为后续流程的输入

**注：** 上述填入的预处理参数仅仅以信息的形式存放在mlir中，后续转换成cvimodel，也仅以信息的方式存放。对图片的预处理过程需要再外部处理，再传给模型运算。如果需要模型内部对图片进行预处理，请参考12章节：使用TPU做前处理。

#### 步骤 3：Calibration

Calibration前需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。
执行calibration：

``` shell
run_calibration.py \
  resnet18_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o resnet18_calibration_table
```

  得到`resnet18_calibration_table`。

#### 步骤 4：模型量化并生成cvimodel

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

以上命令同时包含以下几步:

- 生成MLIR int8量化模型, 运行MLIR int8量化模型的推理, 并与MLIR fp32模型的结果做比较
- 生成cvimodel, 并调用仿真器运行推理结果, 将结果与MLIR int8量化模型结果做比较

**注：** --tolerance 表示 MLIR int8 量化模型与 MLIR fp32模型推理结果相似度的误差容忍度， --correctnetss 表示仿真器运行的结果与MLIR int8模型的结果相似度的误差容忍度， --chip 可以选择cv183x和cv182x 默认使用cv183x

<div STYLE="page-break-after: always;"></div>

## 8 编译移植tensorflow 2.x模型

TPU工具链对Tensorflow 2.x模型采用直接import方式进行。

本章以`mobilenet_v2`为例，介绍如何编译迁移一个tensorflow 2.x模型至CV183x TPU平台运行。

本章需要如下文件：

* cvitek_mlir.tar.gz
* dataset.tar.gz



#### 步骤 0：加载cvitek_mlir环境

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取tensorflow模型

使用tensorflow提供的mobilenet_v2模型，<https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2>

使用下列python脚本下载并保存模型：

``` shell
mkdir model_mobilenet_v2_tf
cd model_mobilenet_v2_tf
```

执行python命令：

``` python
# python
import tensorflow as tf
import numpy as np
import os
model =  tf.keras.applications.MobileNetV2()
model.save('mobilenet_v2',  save_format='tf')
```

得到的模型保存在mobilenet_v2目录，目录结构如下：

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

创建workspace：

``` shell
mkdir workspace && cd workspace
```

#### 步骤 2：模型转换

取得一张测试用图片，本示例使用cvitek_mlir包含的cat.jpg：

``` shell
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

将tensorflow模型转成mlir文件

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

得到`mobilenet_v2_tf_fp32.mlir`文件。

其转换过程包括:

- 原始tensorflow模型的推理, 并将各层的结果保存为numpy的npz文件

- 原始tensorflow模型的导入, 将原始模型转换成MLIR fp32模型
  - 执行MLIR fp32模型的推理, 将各层输出保存到numpy 的npz文件中
  - 将tensorflow模型的推理的结果与MLIR fp32的推理结果对比, 确保转换的MLIR fp32模型正确
  - 将MLIR fp32模型做优化，作为后续流程的输入

**注：** 上述填入的预处理参数仅仅以信息的形式存放在mlir中，后续转换成cvimodel，也仅以信息的方式存放。对图片的预处理过程需要再外部处理，再传给模型运算。如果需要模型内部对图片进行预处理，请参考12章节：使用TPU做前处理。

#### 步骤 3：Calibration

Calibration前需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。
执行calibration：

``` shell
run_calibration.py \
  mobilenet_v2_tf_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mobilenet_v2_tf_calibration_table
```

  得到`mobilenet_v2_tf_calibration_table`。

#### 步骤 4：模型量化并生成cvimodel

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

以上命令同时包含以下几步:

- 生成MLIR int8量化模型，运行MLIR int8量化模型的推理, 并与MLIR fp32模型的结果做比较
- 生成cvimodel, 并调用仿真器运行推理结果, 将结果与MLIR int8量化模型结果做比较

**注：** --tolerance 表示 MLIR int8 量化模型与 MLIR fp32模型推理结果相似度的误差容忍度， --correctnetss 表示仿真器运行的结果与MLIR int8模型的结果相似度的误差容忍度， --chip 可以选择cv183x和cv182x 默认使用cv183x

<div STYLE="page-break-after: always;"></div>

## 9 编译移植tensorflow 1.x模型

TPU工具链对Tensorflow 1.x模型采用转为onnx模型方式进行。

 本章以`mobilenet_v1_0.25`为例，介绍如何编译迁移一个tensorflow 1.x模型至CV183x TPU平台运行。

 本章需要如下文件：

* cvitek_mlir.tar.gz
* dataset.tar.gz



#### 步骤 0：加载cvitek_mlir环境

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取tensorflow模型，并转换为onnx模型

使用tensorflow提供的`mobilenet_v1_0.25_224`模型，参见：

<https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md>

下载链接：

<http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz>

首先打开`mobilenet_v1_0.25_224_eval.pbtxt`，找到输出节点名称为`MobilenetV1/Predictions/Reshape_1`，
使用下列命令转换为onnx模型：

``` shell
mkdir model_mnet_25 && cd model_mnet_25
wget -nc \
http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz
tar zxf mobilenet_v1_0.25_224.tgz
pip install tf2onnx

python3 -m tf2onnx.convert --graphdef mobilenet_v1_0.25_224_frozen.pb --output mnet_25.onnx --inputs input:0 --outputs MobilenetV1/Predictions/Reshape_1:0
```

得到mnet_25.onnx。

但是由于tensorflow模型默认采用NHWC作为输入，转为onnx模型后，仍然是NHWC格式输入，并连接一个transpose节点。在编译前，我们先转换输入格式，并去除这个transpose节点。采用如下python脚本进行：

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

得到`mnet_25_new.onnx`。

#### 步骤 2：模型转换

创建workspace，取得一张测试用图片，本示例使用cvitek_mlir包含的cat.jpg

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

预处理参数如下：

> RAW_SCALE=255
>
> MODEL_CHANNEL_ORDER="rgb"
>
> MEAN=127.5,127.5,127.5 # in RGB
>
> STD=127.5,127.5,127.5
>
> INPUT_SCALE=1.0

将onnx模型转成mlir文件

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

得到`mnet_25_fp32.mlir`文件.

其转换过程包括:

- 原始tensorflow模型的推理, 并将各层结果保存为numpy的npz文件

- 原始tensorflow模型的导入, 将原始模型转换成MLIR fp32模型
  - 执行MLIR fp32模型的推理, 将各层输出保存到numpy 的npz文件中
  - 将caffe模型的推理的结果与MLIR fp32的推理结果对比, 确保转换的MLIR fp32模型正确
  - 将MLIR fp32模型做优化，作为后续流程的输入

**注：** 上述填入的预处理参数仅仅以信息的形式存放在mlir中，后续转换成cvimodel，也仅以信息的方式存放。对图片的预处理过程需要再外部处理，再传给模型运算。如果需要模型内部对图片进行预处理，请参考12章节：使用TPU做前处理。

#### 步骤 4：Calibration

Calibration前需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。
执行calibration：

``` shell
run_calibration.py \
  mnet_25_fp32.mlir \
  --dataset=$DATASET_PATH/imagenet/img_val_extracted \
  --input_num=1000 \
  --histogram_bin_num=20480 \
  -o mnet_25_calibration_table
```

  得到`mnet_25_calibration_table`。

#### 步骤 5：模型量化并生成cvimodel

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

以上命令同时包含以下几步:

- 生成MLIR int8量化模型, 运行MLIR int8量化模型的推理, 并与MLIR fp32模型的结果做比较
- 生成cvimodel, 并调用仿真器运行推理结果, 将结果与MLIR int8量化模型结果做比较

**注：** --tolerance 表示 MLIR int8 量化模型与 MLIR fp32模型推理结果相似度的误差容忍度， --correctnetss 表示仿真器运行的结果与MLIR int8模型的结果相似度的误差容忍度， --chip 可以选择cv183x和cv182x 默认使用cv183x



上一步会产生量化mlir模型文件mnet_25_quantized.mlir, 可以使用pymlir python接口进行测试精度：

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

## 10 编译移植tflite模型

本章以resnet50为例，介绍如何编译迁移一个tflite模型至CV183x TPU平台运行。

本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz



#### 步骤 0：加载cvitek_mlir环境

``` shell
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取tensorflow模型，并转换为tflite模型

(如直接使用准备好的tflite模型, 这一步可以省略)

创建模型目录：

``` shell
mkdir resnet50_tflite && cd resnet50_tflite
```

使用以下python脚本下载tensorflow模型并转换为tflite模型：

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

得到对应的`resnet50_int8_quant.tflite`模型。

#### 步骤 2：执行tflite推理（Optional）

创建工作目录，并取得一张测试用图片，本示例使用cvitek_mlir包含的cat.jpg, 并用以下脚本生成输入数据给interpter使用：

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
# 进入python后台
```

执行python命令如下：

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

运行tflite推理：

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

得到`resnet50_out_ref.npz`。

#### 步骤 3：转换为mlir，进行前端优化

执行转换和前端优化：

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

得到`resnet50_int8_opt.mlir`文件。

运行tpuc-interpreter对mlir进行推理，得到逐层数据：

``` shell
# inference with mlir and input data, dump all tensor
tpuc-interpreter resnet50_int8_opt.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8.npz \
    --dump-all-tensor=resnet50_tensor_all_int8.npz
```

得到resnet50_out_int8.npz。

#### 步骤 4：生成cvimodel

此模型输入为int8模型, 不须做calibraion。

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

测试精度：

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

## 11  精度优化和混合量化使用指南

CV183X TPU支持INT8和BF16两种量化方法。在模型编译阶段，工具链支持以搜索的方式找到对模型精度最敏感的op，并支持将指定数量的op替换为BF16，从而提高整个网络的精度。

本章以`mobilenet_v1_0.25`为例，介绍如何对这个模型采用自动搜索和混合精度的方式提高模型精度。

本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* dataset.tar.gz

#### 步骤 0：获取tensorflow模型，并转换为onnx模型

此处与第9章相同。

使用tensorflow提供的`mobilenet_v1_0.25_224`模型，参见：<https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md>

下载链接：<http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz>

首先打开`mobilenet_v1_0.25_224_eval.pbtxt`，找到输出节点名称为`MobilenetV1/Predictions/Reshape_1`,
使用下列命令转换为onnx模型：

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

得到mnet_25.onnx。

但是由于tensorflow模型默认采用NHWC作为输入，转为onnx模型后，仍然是NHWC格式输入，并连接一个transpose节点。在编译前，我们先转换输入格式，并去除这个transpose节点。采用如下python脚本进行：

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

得到`mnet_25_new.onnx`。

#### 步骤 1：模型转换

取得一张测试用图片，本示例使用cvitek_mlir包含的cat.jpg：

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
```

预处理参数如下：

> RAW_SCALE=255
>
> MODEL_CHANNEL_ORDER="rgb"
>
> MEAN=127.5,127.5,127.5 # in RGB
>
> STD=127.5,127.5,127.5
>
> INPUT_SCALE=1.0

转换为mlir文件:

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

得到`mnet_25_fp32.mlir`文件。


#### 步骤 2：测试FP32模型精度（Optional）

使用pymlir python接口测试精度：

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

测试得到FP32模型精度为Top-1 49.2% Top-5 73.5%。

#### 步骤 4：进行INT8量化

进行calibration：

``` shell
run_calibration.py \
    mnet_25_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --input_num=1000 \
    --calibration_table mnet_25_calibration_table
```

得到`mnet_25_calibration_table`。

进行INT8量化，并进行逐层比较：

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

#### 步骤 5：测试INT8模型精度（Optional)

上一步会产生量化mlir模型文件mnet_25_quantized.mlir, 可以使用pymlir python接口进行测试精度：

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

测试得到INT8模型精度为Top-1 43.2% Top-5 68.3%，比FP32模型精度（Top-1 49.2% Top-5 73.5%）有一定幅度下降。

#### 步骤 6：进行混合量化搜索，并进行混合量化

搜索混合量化表。此模型共有59层，选择多少层进行替换，可以根据对精度的需要，以及测试的精度结果来进行调整。搜索用的数据集数量也可以根据需要调整。

 此处以替换其中6层为例（`--max_bf16_layers=6`），搜索用的测试数据集为100张：

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

得到`mnet_25_mix_precision_bf16_table`，内容如下：

``` shell
cat mnet_25_mix_precision_bf16_table
# MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6:0_relu6_reluClip
# MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:0_Clip
# MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0_Clip
# MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0_relu6_reluClip
```

进行混合量化：

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

#### 步骤 7：测试混合量化模型精度 (Optional)

上一步会产生量化mlir模型文件mnet_25_quantized.mlir, 可以使用pymlir python接口进行测试精度：

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

测试得到混合量化T8模型精度为Top-1 47.4% Top-5 72.3%。

为比较效果，我们调整number_bf16参数分别为10和15，重复上面测试（具体命令略），结果分别为。

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

全bf16量化的测量：

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

比较6种量化方式的结果（混合量化包含6层，10层和15层三个版本）：

| **Quant Type**    | **Top-1** | **Top-5** |
| ----------------- | --------- | --------- |
| INT8              | 43.2%     | 68.3%     |
| MIXED (6 layers)  | 47.4%     | 72.3%     |
| MIXED (10 layers) | 47.6%     | 72.3%     |
| MIXED (15 layers) | 47.8%     | 72.5%     |
| BF16              | 48.5%     | 73.1%     |
| FP32              | 49.2%     | 73.5%     |

<div STYLE="page-break-after: always;"></div>

## 12 使用TPU做前处理

CV183X提供两种硬件资源进行神经网络模型的前处理加速。

* 使用VPSS：VPSS是CV18xx提供的视频后处理模块，并针对神经网络的前处理功能进行了扩展，使得视频处理流水线输出预处理后的图像数据，可以直接用做神经网络输入数据。

* 使用TPU：TPU也可以用于支持常见的前处理计算，包括raw_scale，mean，input_scale，channel swap，split，以及quantization。开发者可以在模型编译阶段，通过编译选项传递相应预处理参数，由编译器直接在模型运输前插入相应前处理算子，生成的cvimodel即可以直接以预处理前的图像作为输入，随模型推理过程使用TPU处理前处理运算。

客户可以基于系统优化需要，灵活选择使用哪个引擎进行预处理。使用VPSS进行预处理的详细使用方法请参阅《CV18xx 媒体软件开发参考》，本文档不做介绍。本章介绍使用TPU做前处理的具体步骤。本章以Caffe模型编译为例，按照第6章的步骤稍做修改，生成支持前处理的cvimodel。以`mobilenet_v2`为例。

#### 步骤 0-3：与Caffe章节相应步骤相同

假设用户以及按照第6章所述步骤，执行完模型转换并生成calibraiton table后。

#### 步骤 4：模型量化并生成含TPU预处理的cvimodel

首先，加载cvitek_mlir环境：

``` shell
source cvitek_mlir/cvitek_envs.sh
cd models_mobilenet_v2/workspace
```

执行以下命令：

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

就可以得到带前处理的cvimodel.

其中`pixel_format`用于指定输入的数据格式，有这几种格式：

| pixel_format  | 说明                         |
| ------------- | ---------------------------- |
| RGB_PLANAR    | rgb顺序，按照nchw摆放        |
| RGB_PACKED    | rgb顺序，按照nhwc摆放        |
| BGR_PLANAR    | bgr顺序，按照nchw摆放        |
| BGR_PACKED    | bgr顺序，按照nhwc摆放        |
| GRAYSCALE     | 仅有一个灰色通道，按nchw摆放 |
| YUV420_PLANAR | yuv420格式，按照nchw摆放     |

其中`aligned_input`用于表示是否数据存在对齐，如果数据来源于VPSS，则会有数据对齐要求，比如w按照32字节对齐。

以上过程包含以下几步:

- 生成带前处理的MLIR int8模型, 以及不包含前处理的输入 mobilenet_v2_resized_only_in_fp32.npz
- 执行MLIR int8推理 与 MLIR fp32 推理结果的比较, 验证MLIR int8 带前处理模型的正确性
- 生成带前处理的 cvimodel, 以及调用仿真器执行推理, 将结果与 MLIR int8 带前处理的模型的推理结果做比较