![image](./assets/logo_0.png)

# CV183x/CV182x TPU 快速入门指南




>文档版本: 1.5.3
>
>发布日期: 2021-11-01



© 2021 北京晶视智能科技有限公司

本文件所含信息归<u>北京晶视智能科技有限公司</u>所有。

未经授权，严禁全部或部分复制或披露该等信息。

<div STYLE="page-break-after: always;"></div>

## 修订记录

| 版本    | 日期       | 修改描述                                       |
| ------- | ---------- |--------------------------------------------- |
| V0.0.1  | 2019/12/11 | 初始版本                                      |
| V0.1.0  | 2020/04/18 | 增加使用说明                                  |
| V0.1.1  | 2020/04/20 | 更新测试命令                                  |
| V0.1.2  | 2020/04/28 | 更新模型精度测试命令                           |
| V0.2.0  | 2020/04/30 | 修订                                         |
| V0.2.1  | 2020/05/15 | 根据V0.9 SDK更新部分命令                      |
| V0.2.2  | 2020/06/01 | 增加端到端推理性能测试命令                     |
| V0.3.0  | 2020/06/25 | 根据V1.0 SDK修订增加移植编译TensorFlow 2.x模型 |
| V0.3.1  | 2020/06/29 | 采用python importer进行caffe移植              |
| V0.3.2  | 2020/07/17 | 增加第9章使用TPU进行前处理                     |
| V0.3.3  | 2020/07/19 | 根据V1.1 SDK修订                              |
| V0.3.4  | 2020/07/20 | 修订                                          |
| V0.3.5  | 2020/07/29 | 增加移植编译TensorFlow 2.x模型                 |
| V0.3.6  | 2020/08/08 | 增加精度优化和混合量化指南                      |
| V0.3.7  | 2020/08/12 | 更新使用TPU进行前处理流程                       |
| V0.3.8  | 2020/09/06 | 根据V1.2 SDK修订                               |
| V0.3.9  | 2020/09/22 | 增加移植tflite模型                             |
| V0.3.10 | 2020/09/30 | 更新移植tflite模型                             |
| V0.3.11 | 2020/10/26 | 根据V1.3 SDK修订                               |
| V1.4.0  | 2020/12/07 | 根据V1.4 SDK修订                               |
| V1.5.0  | 2021/01/29 | 根据V1.5 SDK修订                               |
| V1.5.1 | 2021/08/01  | 根据V1.5.1 SDK修订                             |
| V1.5.2 | 2021/09/20  | 根据V1.5.2 SDK修订                             |
| V1.5.3 | 2021/11/01  | 根据V1.5.3 SDK修订                             |

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

#### 1) 阅读说明

本文档包含下述章节，请根据需要参阅相关章节。

* 开发环境配置

  使用CVITEK提供的docker，配置编译开发所需的环境

* 编译移植pytorch模型

  以pytorch为例，介绍如何将原始模型转换成onnx模型，实现bf16和int8的转换为cvimodel，再到开发板中验证结果

* 编译移植caffe模型

  介绍如何移植一个新的caffe模型，实现bf16和int8的转换为cvimodel，再到开发板中验证结果

* 精度优化和混合量化

  介绍BF16量化和混合量化，提高精度

* 使用TPU进行前处理

  介绍如何在cvimodel模型中增加前处理描述，并在运行时使用TPU进行前处理，将未经过前处理的输入到开发板带前处理的模型验证

* 合并cvimodel模型

  介绍将同一个模型的不同batch合并到一起，以及接口如何调用

* 编译samples程序

  介绍如何交叉编译sample应用程序，调用runtime API完成推理任务。具体包括3个samples：

  * Sample-1 : classifier (mobilenet_v2)

  * Sample-2 : classifier fused preprocess (mobilenet_v2)

  * Sample-3 : classifier multiple batch (mobilenet_v2)

#### 2) Release 内容

CVITEK Release包含如下组成部分：

| 文件                                    | 描述                                           |
| --------------------------------------- | ---------------------------------------------- |
| cvitek_mlir_ubuntu-18.04.tar.gz         | cvitek NN工具链软件                            |
| cvitek_tpu_sdk_[cv182x/cv183x].tar.gz   | cvitek Runtime SDK，包括交叉编译头文件和库文件 |
| cvitek_tpu_samples.tar.gz               | sample程序源代码                               |
| cvimodel_samples_[cv182x/cv183x].tar.gz | sample程序使用的cvimodel模型文件               |
| docker_cvitek_dev.tar                   | CVITEK开发Docker镜像文件                       |

#### 3) 当前支持测试的网络列表

cv183x支持的网络如下：

| Classification                | Detection                  | Misc                     |
| ----------------------------- | -------------------------- | ------------------------ |
| resnet50       [BS=1,4]       | retinaface_mnet25 [BS=1,4] | arcface_res50 [BS=1,4]   |
| resnet18       [BS=1,4]       | retinaface_res50   [BS=1]  | alphapose       [BS=1,4] |
| mobilenet_v1     [BS=1,4]     | ssd300        [BS=1,4]     | espcn_3x       [BS=1,4]  |
| mobilenet_v2     [BS=1,4]     | mobilenet_ssd [BS=1,4]     | unet          [BS=1,4]   |
| squeezenet_v1.1    [BS=1,4]   | yolo_v1_448      [BS=1]    | erfnet         [BS=1]    |
| shufflenet_v2     [BS=1,4]    | yolo_v2_416      [BS=1]    |                          |
| googlenet       [BS=1,4]      | yolo_v2_1080     [BS=1]    |                          |
| inception_v3     [BS=1,4]     | yolo_v3_416      [BS=1,4]  |                          |
| inception_v4     [BS=1,4]     | yolo_v3_608      [BS=1]    |                          |
| vgg16         [BS=1,4]        | yolo_v3_tiny     [BS=1]    |                          |
| densenet_121     [BS=1,4]     | yolo_v3_spp      [BS=1]    |                          |
| densenet_201     [BS=1,4]     | yolo_v4        [BS=1]      |                          |
| senet_res50      [BS=1,4]     |                            |                          |
| resnext50       [BS=1,4]      |                            |                          |
| res2net50       [BS=1,4]      |                            |                          |
| ecanet50       [BS=1,4]       |                            |                          |
| efficientnet_b0    [BS=1,4]   |                            |                          |
| efficientnet_lite_b0 [BS=1,4] |                            |                          |
| nasnet_mobile     [BS=1,4]    |                            |                          |

cv182x支持的网络如下：

| Classification                | Detection                  | Misc                     |
| ----------------------------- | -------------------------- | ------------------------ |
| resnet50       [BS=1,4]       | retinaface_mnet25 [BS=1,4] | arcface_res50 [BS=1,4]   |
| resnet18       [BS=1,4]       | retinaface_res50   [BS=1]  | alphapose       [BS=1,4] |
| mobilenet_v1     [BS=1,4]     | mobilenet_ssd [BS=1,4]     |                          |
| mobilenet_v2     [BS=1,4]     | yolo_v1_448      [BS=1]    |                          |
| squeezenet_v1.1    [BS=1,4]   | yolo_v2_416      [BS=1]    |                          |
| shufflenet_v2     [BS=1,4]    | yolo_v3_416      [BS=1,4]  |                          |
| googlenet       [BS=1,4]      | yolo_v3_608      [BS=1]    |                          |
| inception_v3     [BS=1]       | yolo_v3_tiny     [BS=1]    |                          |
| densenet_121     [BS=1,4]     |                            |                          |
| densenet_201     [BS=1]       |                            |                          |
| senet_res50      [BS=1]       |                            |                          |
| resnext50       [BS=1,4]      |                            |                          |
| efficientnet_lite_b0 [BS=1,4] |                            |                          |
| nasnet_mobile     [BS=1]      |                            |                          |

**注：** BS表示batch，[BS=1]表示板子目前至少batch 1，[BS=1,4]表示板子至少支持batch 1和batch 4。

<div STYLE="page-break-after: always;"></div>

## 2 开发环境配置

加载镜像文件：

```shell
docker load -i docker_cvitek_dev_1.7-ubuntu-18.04.tar
```

或者从docker hub获取:

```shell
docker pull cvitek/cvitek_dev:1.7-ubuntu-18.04
```

如果是首次使用docker，可执行下述命令进行安装和配置（Ubuntu系统）

```shell
sudo apt install docker.io
systemctl start docker
systemctl enable docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker (use before reboot)
```

取得docker image后，执行下述命令运行docker：

``` shell
docker run -itd -v $PWD:/work --name cvitek cvitek/cvitek_dev:1.5-ubuntu-18.04
docker exec -it cvitek bash
```

如果需要挂载模型和数据集到docker中，可以如下操作：（可选）
``` shell
# 这里假设models和dataset分别位于~/data/models和~/data/dataset目录，如有不同请相应调整。
docker run -itd -v $PWD:/work \
   -v ~/data/models:/work/models \
   -v ~/data/dataset:/work/dataset \
   --name cvitek cvitek/cvitek_dev:1.5-ubuntu-18.04
docker exec -it cvitek bash
```

<div STYLE="page-break-after: always;"></div>

## 3 编译移植pytorch模型

本章以resnet18为例，介绍如何编译迁移一个pytorch模型至CV183x TPU平台运行。

 本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz

除caffe外的框架，如tensorflow/pytorch均可以参考本章节步骤，先转换成onnx，再转换成cvimodel。

如何将模型转换成onnx，可以参考onnx官网: <https://github.com/onnx/tutorials>

#### 步骤 0：加载cvitek_mlir环境

``` shell
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
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

#### 步骤 2：onnx模型转换为fp32 mlir形式

创建工作目录workspace，拷贝测试图片cat.jpg，和数据集100张图片（来自ILSVRC2012）：

``` shell
mkdir workspace && cd workspacecp
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
cp -rf $MLIR_PATH/tpuc/regression/data/images .
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

使用`model_transform.py`将onnx模型转换成mlir文件，其中可支持预处理参数如下:
| **参数名**           | **说明**                                      |
| ------------------- | ---------------------------------------------  |
| net_input_dims      | 模型输入的大小，比如224,224 |
| image_resize_dims   | 改变图片宽高的大小，默认与网络输入维度一样  |
| keep_aspect_ratio   | 在Resize时是否保持长宽比，默认为false；设置true时会对不足部分补0 |
| raw_scale           | 图像的每个像素与255的比值，默认为255.0 |
| mean                | 图像每个通道的均值，默认为0.0,0.0,0.0  |
| std                 | 图像每个通道的标准值，默认为1.0,1.0,1.0  |
| input_scale         | 图像的每个像素比值，默认为1.0 |
| model_channel_order | 模型的通道顺序(bgr/rgb/rgba)，默认为bgr |
| gray                | 支持灰度格式，默认为Flase |

预处理过程用公式表达如下（x代表输入)：
$$
y = \frac{x \times \frac{raw\_scale}{255.0} - mean}{std} \times input\_scale
$$

由onnx模型转换为mlir,执行以下shell：

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

**注：** 上述填入的预处理参数仅仅以信息的形式存放在mlir中，后续转换成cvimodel，也仅以信息的方式存放。对图片的预处理过程需要再外部处理，再传给模型运算。如果需要模型内部对图片进行预处理，请参考12章节：使用TPU做前处理。

#### 步骤 3：生成全bf16量化cvimodel

``` shell
model_deploy.py \
  --model_name resnet18 \
  --mlir resnet18_fp32.mlir \
  --quantize BF16 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.99,0.99,0.86 \
  --correctness 0.99,0.99,0.93 \
  --cvimodel resnet18_bf16.cvimodel
```
#### 步骤 4：生成全int8量化cvimodel

先做Calibration，需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。

这里用100张图片举例，执行calibration,执行如下shell：

``` shell
run_calibration.py \
  resnet18_fp32.mlir \
  --dataset=./images \
  --input_num=100 \
  -o resnet18_calibration_table
```

得到`resnet18_calibration_table`。

导入calibration_table，生成cvimodel：

``` shell
model_deploy.py \
  --model_name resnet18 \
  --mlir resnet18_fp32.mlir \
  --calibration_table resnet18_calibration_table \
  --quantize INT8 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.98,0.98,0.84 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel resnet18_int8.cvimodel
```

以上命令包含以下几步:

- 生成MLIR int8模型, 运行MLIR量化模型的推理, 并与MLIR fp32模型的结果做比较
- 生成cvimodel, 并调用仿真器运行推理结果, 将结果与MLIR 量化模型做比较

model_deploy.py的相关参数说明如下：

| **参数名**               | **说明**                                                        |
| -------------------     | ----------------------------------------------------------------|
| model_name              | 模型名称                                                         |
| mlir                    | mlir文件                                                        |
| calibration_table       | 指定calibration文件路径                                          |
| quantize                | 指定默认量化方式，BF16/MIX_BF16/INT8                       |
| tolerance               | 表示 MLIR 量化模型与 MLIR fp32模型推理结果相似度的误差容忍度          |
| excepts                 | 指定需要排除比较的层的名称，默认为-                        |
| correctnetss            | 表示仿真器运行的结果与MLIR int8模型的结果相似度的误差容忍度          |
| chip                    | 支持平台，可以为cv183x或cv182x                                    |
| inputs_type             | 指定输入类型(AUTO/FP32/INT8/BF16)，如果是AUTO，当第一层是INT8时用INT8，BF16时用FP32 |
| outputs_type            | 指定输出类型(AUTO/FP32/INT8/BF16)，如果是AUTO，当最后层是INT8时用INT8，BF16时用FP32  |
| model_version           | 支持选择模型的版本，默认为latest                                    |
| custom_op_plugin        | 支持用户自定义op的动态库文件                                        |
| image                   | 用于测试的输入文件，可以是图片、npz、npy；如果有多个输入，用,隔开       |
| cvimodel                | 表示输出的cvimodel文件名                                         |
| debug                   | 调试选项，保存所有的临时文件进行调试                                |



#### 步骤 5：开发板中测试bf16和int8模型

配置开发板的TPU sdk环境：


``` shell
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

测试仿真环境与真实硬件的输出结果，需要步骤三或步骤四生成的调试文件：

* xxx_quantized_tensors_sim.npz 仿真环境中网络推理过程的tensor文件，作为与真实硬件输出结果参考对比
* xxx__in_fp32.npz              模型的输入tensor文件，测试不同类型的模型，input_npz文件需要不一样
* xxx_[int8/bf16].cvimodel      输出int8或者bf16的cvimodel文件

在开发板中执行以下shell，测试量化为int8模型的仿真环境和真实硬件输出结果进行比较

``` shell
model_runner \
--input resnet18_in_fp32.npz \
--model resnet18_int8.cvimodel \
--output out.npz \
--dump-all-tensors \
--reference resnet18_quantized_tensors_sim.npz
```

<div STYLE="page-break-after: always;"></div>

## 4 编译移植caffe模型

本章以mobilenet_v2为例，介绍如何编译迁移一个caffe模型至TPU平台运行

 本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz

#### 步骤 0：加载cvitek_mlir环境

``` shell
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```

#### 步骤 1：获取caffe模型

从<https://github.com/shicai/MobileNet-Caffe>下载模型，并保存在`model_mobilenet_v2`目录：

``` shell
mkdir model_mobilenet_v2 && cd model_mobilenet_v2
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
wget -nc https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt
```

创建工作目录workspace，拷贝测试图片cat.jpg，和数据集100张图片（来自ILSVRC2012）：

``` shell
mkdir workspace && cd workspace
cp $MLIR_PATH/tpuc/regression/data/cat.jpg .
cp -rf $MLIR_PATH/tpuc/regression/data/images .
```

#### 步骤 2：caffe模型转换为fp32 mlir形式

使用`model_transform.py`将模型转换成mlir文件，其中可支持预处理参数如下：

由caffe模型转换为mlir，执行以下shell：

``` shell
model_transform.py \
  --net_input_dims 224,224 \
  --keep_aspect_ratio false \
  --raw_scale 255.0 \
  --mean 103.94,115.78,123.68 \
  --std 1.0,1.0,1.0 \
  --input_scale 0.017 \
  --model_channel_order "bgr" \
  --gray false \
  --image_resize_dims 256,256 \
  --image ./cat.jpg \
  --model_def ../mobilenet_v2_deploy.prototxt \
  --model_data ../mobilenet_v2.caffemodel \
  --model_name mobilenet_v2 \
  --model_type caffe \
  --batch_size 1 \
  --tolerance 0.99,0.99,0.99 \
  --excepts prob \
  --mlir mobilenet_v2_fp32.mlir
```

得到`mobilenet_v2_fp32.mlir`文件.

#### 步骤 3：生成全bf16量化cvimodel

``` shell
model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32.mlir \
  --quantize BF16 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.94,0.94,0.61 \
  --correctness 0.99,0.99,0.96 \
  --cvimodel mobilenet_v2_bf16.cvimodel
```


#### 步骤 4：生成全int8量化cvimodel
先做Calibration。Calibration前需要先准备校正图片集,图片的数量根据情况准备100~1000张左右。
这里用100张图片举例，执行calibration：

``` shell
run_calibration.py \
  mobilenet_v2_fp32.mlir \
  --dataset=./images \
  --input_num=100 \
  -o mobilenet_v2_calibration_table
```

得到`mobilenet_v2_calibration_table`。

导入calibration_table，生成cvimodel：

``` shell
model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.94,0.94,0.61 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mobilenet_v2_int8.cvimodel
```

#### 步骤 5：开发板中测试bf16和int8模型

配置开发板的TPU sdk环境：


``` shell
# 此处以183x举例
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

测试仿真环境与真实硬件的输出结果，需要步骤三或步骤四生成的调试文件：

* xxx_quantized_tensors_sim.npz 仿真环境中网络推理过程的tensor文件，作为与真实硬件输出结果参考对比
* xxx__in_fp32.npz              模型的输入tensor文件，测试不同类型的模型，input_npz文件需要不一样
* xxx_[int8/bf16].cvimodel      输出int8或者bf16的cvimodel文件

在开发板中执行以下shell，测试量化为int8模型的仿真环境和真实硬件输出结果进行比较：

``` shell
model_runner \
--input mobilenet_v2_in_fp32.npz \
--model mobilenet_v2_int8.cvimodel \
--output out.npz \
--dump-all-tensors \
--reference mobilenet_v2_quantized_tensors_sim.npz
```

<div STYLE="page-break-after: always;"></div>

## 5  精度优化和混合量化

TPU支持INT8和BF16两种量化方法。在模型编译阶段，工具链支持以搜索的方式找到对模型精度最敏感的op，并支持将指定数量的op替换为BF16，从而提高整个网络的精度。

本章以`mobilenet_v1_0.25`为例，介绍如何对这个模型采用自动搜索和混合精度的方式提高模型精度。

本章需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz

#### 步骤 0：获取tensorflow模型，并转换为onnx模型

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

#### 步骤 1：模型转换为fp32 mlir形式

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

###### 测试FP32模型精度（可选）

数据集来自ILSVRC2012，下载地址： <https://github.com/cvitek-mlir/dataset>
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

#### 步骤 2：进行INT8量化

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
  --cvimodel mnet_25.cvimodel
```

###### 测试INT8模型精度（可选)

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

#### 步骤 3：进行BF16量化

``` shell
model_deploy.py \
  --model_name mnet_25 \
  --mlir mnet_25_fp32.mlir \
  --quantize BF16 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.99,0.99,0.86 \
  --correctness 0.99,0.99,0.93 \
  --cvimodel mnet_25_all_bf16_precision.cvimodel
```

###### 测试BF16模型精度 （可选)

上一步会产生量化mlir模型文件mnet_25_quantized.mlir, 可以使用pymlir python接口进行测试精度

``` shell
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

# Test: [49950/50000]     Time  0.031 ( 0.036)    Loss 6.4377e+00 (6.5711e+00)    Acc@1 100.00 ( 48.49)   Acc@5 100.00 ( 73.06)
# Test: [50000/50000]     Time  0.033 ( 0.036)    Loss 6.8726e+00 (6.5711e+00)    Acc@1   0.00 ( 48.50)   Acc@5 100.00 ( 73.06)
# * Acc@1 48.498 Acc@5 73.064
```



#### 步骤 4：进行混合量化搜索，并进行混合量化

搜索混合量化表。此模型共有59层，选择多少层进行替换，可以根据对精度的需要，以及测试的精度结果来进行调整。搜索用的数据集数量也可以根据需要调整。

 此处以替换其中6层为例（`--max_bf16_layers=6`），搜索用的测试数据集为100张：

``` shell
run_mix_precision.py \
    mnet_25_fp32.mlir \
    --dataset ${DATASET_PATH} \
    --input_num=20 \
    --calibration_table mnet_25_calibration_table \
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
  --quantize INT8 \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.94,0.93,0.67 \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mnet_25_mix_precision.cvimodel
```

###### 测试混合量化模型精度 (可选)

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

###### 各种量化精度测试对比 (可选)

比较6种量化方式的结果（混合量化包含6层，10层和15层三个版本）：

| **Quant Type**    | **Top-1** | **Top-5** |
| ----------------- | --------- | --------- |
| INT8              | 43.2%     | 68.3%     |
| MIXED (6 layers)  | 47.4%     | 72.3%     |
| MIXED (10 layers) | 47.6%     | 72.3%     |
| MIXED (15 layers) | 47.8%     | 72.5%     |
| BF16              | 48.5%     | 73.1%     |
| FP32              | 49.2%     | 73.5%     |

#### 混精度规则说明

**规则一：** quantize参数可以指定默认的量化方式

目前支持三种量化方式，BF16、INT8、MIX_BF16。其中MIX_BF16方式，精度和性能都介于BF16与INT8之间，ION内存占用与INT8相同。

**规则二：** mix_precision_table指定特定量化方式，优先级高于quantize

mix_precision_table文件格式如下：

```shell
# layer_name quantize_type
# 如果没有跟quantize_type，则认为是BF16
120_Add BF16
121_Conv INT8
122_Conv
```

**规则三：** 当存在INT8的量化时，需要传入calibration_table

当quantize指定为INT8时，或者mix_precision_table中存在layer被指定为INT8时，则需要传入calibration参数。

<div STYLE="page-break-after: always;"></div>

## 6 使用TPU做前处理

CV183X提供两种硬件资源进行神经网络模型的前处理加速。

* 使用VPSS：VPSS是CV18xx提供的视频后处理模块，并针对神经网络的前处理功能进行了扩展，使得视频处理流水线输出预处理后的图像数据，可以直接用做神经网络输入数据。

* 使用TPU：TPU也可以用于支持常见的前处理计算，包括raw_scale，mean，input_scale，channel swap，split，以及quantization。开编发者可以在模型编译阶段，通过译选项传递相应预处理参数，由编译器直接在模型运输前插入相应前处理算子，生成的cvimodel即可以直接以预处理前的图像作为输入，随模型推理过程使用TPU处理前处理运算。

客户可以基于系统优化需要，灵活选择使用哪个引擎进行预处理。使用VPSS进行预处理的详细使用方法请参阅《CV18xx 媒体软件开发参考》，本文档不做介绍。本章介绍使用TPU做前处理的具体步骤。本章以Caffe模型编译为例，按照第6章的步骤稍做修改，生成支持前处理的cvimodel。以`mobilenet_v2`为例。

#### 步骤 0-4：与Caffe章节相应步骤相同

假设用户以及按照第4章步骤0到步骤2执行完模型转换后, 并且跳过步骤3全量化为bf16模型，到步骤4生成全INT8量化模型

#### 步骤 5：模型量化并生成含TPU预处理的cvimodel

首先，加载cvitek_mlir环境：

``` shell
source cvitek_mlir/cvitek_envs.sh
cd model_mobilenet_v2/workspace
```

执行以下命令：

``` shell
#与预处理有关的参数：fuse_preprocess/pixel_format/aligned_input
model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.94,0.94,0.61 \
  --fuse_preprocess \
  --pixel_format BGR_PACKED \
  --aligned_input false \
  --excepts data \
  --correctness 0.99,0.99,0.99 \
  --cvimodel mobilenet_v2_fused_preprocess.cvimodel
```

就可以得到带前处理的cvimodel.

其中`pixel_format`用于指定外部输入的数据格式，有这几种格式：

| pixel_format  | 说明                         |
| ------------- | ---------------------------- |
| RGB_PLANAR    | rgb顺序，按照nchw摆放        |
| RGB_PACKED    | rgb顺序，按照nhwc摆放        |
| BGR_PLANAR    | bgr顺序，按照nchw摆放        |
| BGR_PACKED    | bgr顺序，按照nhwc摆放        |
| GRAYSCALE     | 仅有一个灰色通道，按nchw摆放 |
| YUV420_PLANAR | yuv420 planner格式，来自vpss的输入 |
| YUV_NV21 | yuv420的NV21格式，来自vpss的输入 |
| YUV_NV12 | yuv420的NV12格式，来自vpss的输入 |
| RGBA_PLANAR | rgba格式，按照nchw摆放     |

其中`aligned_input`用于表示是否数据存在对齐，如果数据来源于VPSS，则会有数据对齐要求，比如w按照32字节对齐。

以上过程包含以下几步:

- 生成带前处理的MLIR int8模型, 以及不包含前处理的输入 mobilenet_v2_resized_only_in_fp32.npz
- 执行MLIR int8推理 与 MLIR fp32 推理结果的比较, 验证MLIR int8 带前处理模型的正确性
- 生成带前处理的 cvimodel, 以及调用仿真器执行推理, 将结果与 MLIR int8 带前处理的模型的推理结果做比较

#### 步骤 6：开发板中测试带前处理的模型

配置开发板的TPU sdk环境


``` shell
# 183x为例
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

测试仿真环境与真实硬件的输出结果，需要之前步骤生成的调试文件：

* xxx_quantized_tensors_sim.npz 仿真环境中网络推理过程的tensor文件，作为与真实硬件输出结果参考对比
* xxx_[int8/bf16].cvimodel      输出int8或者bf16的cvimodel文件
* xxx_in_fp32_resize_only.npz   未经过前处理的输入数据文件

在开发板中执行以下shell，测试量化为int8模型的仿真环境和真实硬件输出结果进行比较

``` shell
model_runner \
--input mobilenet_v2_in_fp32_resize_only.npz \
--model mobilenet_v2_fused_preprocess.cvimodel \
--output out.npz --dump-all-tensors \
--reference mobilenet_v2_quantized_tensors_sim.npz
```

以上过程包含将为经过未前处理的输入文件放到带前处理的cvi_model模型中推理，并且将输出的结果和在仿真环境输出的结果进行比较，保证误差在可控范围内。


<div STYLE="page-break-after: always;"></div>

## 7 合并cvimodel模型文件
对于同一个模型，可以依据输入的batch size以及分辨率(不同的h和w)分别生成独立的cvimodel文件。不过为了节省外存和运存，可以选择将这些相关的cvimodel文件合并为一个cvimodel文件，共享其权重部分。具体步骤如下：
#### 步骤 0：生成batch 1的cvimodel
请参考前述章节，新建workspace，通过model_transform.py将mobilenet_v2的caffemodel转换为mlir fp32模型:

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
  --mlir mobilenet_v2_fp32_bs1.mlir
```

使用第4章节生成的`mobilenet_v2_calibration_table`；如果没有，则通过run_calibration.py工具对mobilenet_v2_fp32.mlir进行量化校验获得calibration table文件.

然后将模型量化并生成cvimodel：

``` shell
 # 加上--merge_weight 参数
 model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32_bs1.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.95,0.94,0.69 \
  --correctness 0.99,0.99,0.99 \
  --merge_weight  \
  --cvimodel mobilenet_v2_bs1.cvimodel
```

#### 步骤 1：生成batch 4的cvimodel
同步骤1，在同一个workspace中生成batch为4的mlir fp32文件:

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
  --batch_size 4 \
  --tolerance 0.99,0.99,0.99 \
  --excepts prob \
  --mlir mobilenet_v2_fp32_bs4.mlir
```
使用`mobilenet_v2_calibration_table`文件将模型量化并生成cvimodel：

``` shell
 # 打开--merge_weight选项
 model_deploy.py \
  --model_name mobilenet_v2 \
  --mlir mobilenet_v2_fp32_bs4.mlir \
  --calibration_table mobilenet_v2_calibration_table \
  --chip cv183x \
  --image cat.jpg \
  --tolerance 0.95,0.94,0.69 \
  --correctness 0.99,0.99,0.99 \
  --merge_weight \
  --cvimodel mobilenet_v2_bs4.cvimodel
```
#### 步骤 2: 合并batch 1和batch 4的cvimodel
使用cvimodel_tool合并两个cvimodel文件:
``` shell
cvimodel_tool \
  -a merge \
  -i mobilenet_v2_bs1.cvimodel \
     mobilenet_v2_bs4.cvimodel \
  -o mobilenet_v2_bs1_bs4.cvimodel
```
#### 步骤 3：runtime接口调用cvimodel
在运行时可以通过命令：
``` shell
cvimodel_tool -a dump -i mobilenet_v2_bs1_bs4.cvimodel
```
查看bs1和bs4指令的program id，在运行时可以透过如下方式去运行不同的batch指令：
``` c++
CVI_MODEL_HANDEL bs1_handle;
CVI_RC ret = CVI_NN_RegisterModel("mobilenet_v2_bs1_bs4.cvimodel", &bs1_handle);
assert(ret == CVI_RC_SUCCESS);
CVI_NN_SetConfig(bs1_handle, OPTION_PROGRAM_INDEX, 0);
CVI_NN_GetInputOutputTensors(bs_handle, ...);
....


CVI_MODEL_HANDLE bs4_handle;
// 复用已加载的模型
CVI_RC ret = CVI_NN_CloneModel(bs1_handle, &bs4_handle);
assert(ret == CVI_RC_SUCCESS);
// 选择bs4的指令
CVI_NN_SetConfig(bs4_handle, OPTION_PROGRAM_INDEX, 1);
CVI_NN_GetInputOutputTensors(bs_handle, ...);
...

// 最后销毁bs1_handle, bs4_handel
CVI_NN_CleanupModel(bs1_handle);
CVI_NN_CleanupModel(bs4_handle);

```

#### 综述：合并过程

使用上面的方面，不论是相同模型还是不同模型，均可以进行合并。
合并的原理是：模型生成过程中，会叠加前面模型的weight（如果相同则共用）。
主要步骤在于：

1. 用`model_deploy.py`生成模型时，加上`--merge_weight`参数
2. 要合并的模型的生成目录必须是同一个，且在合并模型前不要清理任何中间文件
3. 用 `cvimodel_tool -a merge`将多个cvimodel合并


<div STYLE="page-break-after: always;"></div>

## 8 运行runtime samples

#### 1) EVB运行Samples程序

在EVB运行release提供的sample预编译程序。

需要如下文件：

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvimodel_samples_[cv182x/cv183x].tar.gz

将根据chip类型选择所需文件加载至EVB的文件系统，于evb上的linux console执行，以cv183x为例：

解压samples使用的model文件（以cvimodel格式交付），并解压TPU_SDK，并进入samples目录，执行测试，过程如下：

``` shell
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
./bin/cvi_sample_classifier \
    $MODEL_PATH/mobilenet_v2.cvimodel \
    ./data/cat.jpg \
    ./data/synset_words.txt

# TOP_K[5]:
#  0.361328, idx 285, n02124075 Egyptian cat
#  0.062500, idx 287, n02127052 lynx, catamount
#  0.045898, idx 331, n02326432 hare
#  0.006012, idx 852, n04409515 tennis ball
#  0.001854, idx 332, n02328150 Angora, Angora rabbit


############################################
# sample-2 : classifier fused preprocess
############################################
./bin/cvi_sample_classifier_fused_preprocess \
    $MODEL_PATH/mobilenet_v2_fused_preprocess.cvimodel \
    ./data/cat.jpg \
    ./data/synset_words.txt

# TOP_K[5]:
#  0.361328, idx 285, n02124075 Egyptian cat
#  0.062500, idx 287, n02127052 lynx, catamount
#  0.045898, idx 331, n02326432 hare
#  0.006012, idx 852, n04409515 tennis ball
#  0.001854, idx 332, n02328150 Angora, Angora rabbit

############################################
# sample-3 : classifier multiple batch
############################################
./bin/cvi_sample_classifier_multi_batch \
    $MODEL_PATH/mobilenet_v2_bs1_bs4.cvimodel \
    ./data/cat.jpg \
    ./data/synset_words.txt

# TOP_K[5]:
#  0.361328, idx 285, n02124075 Egyptian cat
#  0.062500, idx 287, n02127052 lynx, catamount
#  0.045898, idx 331, n02326432 hare
#  0.006012, idx 852, n04409515 tennis ball
#  0.001854, idx 332, n02328150 Angora, Angora rabbit

```

同时提供脚本作为参考，执行效果与直接运行相同，如下：

``` shell
./run_classifier.sh
./run_classifier_fused_preprocess.sh
./run_classifier_multi_batch.sh
```

**在cvitek_tpu_sdk/samples/samples_extra目录下有更多的samples，可供参考：**

```sh
./bin/cvi_sample_classifier_yuv420 \
    $MODEL_PATH/mobilenet_v2_int8_yuv420.cvimodel \
    ./data/cat.jpg \
    ./data/synset_words.txt

./bin/cvi_sample_detector_yolo_v3 \
    $MODEL_PATH/yolo_v3_416_with_detection.cvimodel \
    ./data/dog.jpg \
    yolo_v3_out.jpg

./bin/cvi_sample_alphapose \
    $MODEL_PATH/yolo_v3_416_with_detection.cvimodel \
    $MODEL_PATH/alphapose.cvimodel \
    ./data/pose_demo_2.jpg \
    alphapose_out.jpg

./bin/cvi_sample_alphapose_fused_preprocess \
    $MODEL_PATH/yolo_v3_416_with_detection.cvimodel \
    $MODEL_PATH/alphapose_fused_preprocess.cvimodel \
    ./data/pose_demo_2.jpg \
    alphapose_out.jpg

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama1.jpg \
    ./data/obama2.jpg

# Similarity: 0.747192

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama1.jpg \
    ./data/obama3.jpg

# Similarity: 0.800899

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama2.jpg \
    ./data/obama3.jpg

# Similarity: 0.795205

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama1.jpg \
    ./data/trump1.jpg

# Similarity: -0.013767

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama1.jpg \
    ./data/trump2.jpg

# Similarity: -0.060050

./bin/cvi_sample_fd_fr \
    $MODEL_PATH/retinaface_mnet25_600_with_detection.cvimodel \
    $MODEL_PATH/arcface_res50.cvimodel \
    ./data/obama1.jpg \
    ./data/trump3.jpg

# Similarity: 0.036089
```

#### 2) 交叉编译samples程序

发布包有samples的源代码，按照本节方法在Docker环境下交叉编译samples程序，然后在evb上运行。

本节需要如下文件：

* cvitek_tpu_sdk_[cv182x/cv183x].tar.gz
* cvitek_tpu_samples.tar.gz

**64位平台**

> 如cv183x 64位平台

TPU sdk准备：

``` shell
tar zxf cvitek_tpu_sdk_cv183x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
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

**32位平台**

> 如cv182x平台32位，或cv183x平台32位

TPU sdk准备：

``` shell
tar zxf cvitek_tpu_sdk_cv182x.tar.gz
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk
source ./envs_tpu_sdk.sh
cd ..
```

如果docker版本低于1.7，则需要更新32位系统库（只需一次）：

``` shell
dpkg --add-architecture i386
apt-get update
apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386
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
    -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-linux-gnueabihf.cmake \
    -DTPU_SDK_PATH=$TPU_SDK_PATH \
    -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
    -DCMAKE_INSTALL_PREFIX=../install_samples \
    ..
cmake --build . --target install
```

#### 3) 编译docker环境下运行的samples程序

需要如下文件：

* cvitek_mlir_ubuntu-18.04.tar.gz
* cvimodel_samples_[cv182x/cv183x].tar.gz
* cvitek_tpu_samples.tar.gz

TPU sdk准备：

``` shell
tar zxf cvitek_mlir_ubuntu-18.04.tar.gz
source cvitek_mlir/cvitek_envs.sh
```

编译samples，安装至install_samples目录：

``` shell
tar zxf cvitek_tpu_samples.tar.gz
cd cvitek_tpu_samples
mkdir build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
    -DTPU_SDK_PATH=$MLIR_PATH/tpuc \
    -DCNPY_PATH=$MLIR_PATH/cnpy \
    -DOPENCV_PATH=$MLIR_PATH/opencv \
    -DCMAKE_INSTALL_PREFIX=../install_samples \
    ..
cmake --build . --target install
```

运行samples程序：

``` shell
# envs
tar zxf cvimodel_samples_cv183x.tar.gz
export MODEL_PATH=$PWD/cvimodel_samples
source cvitek_mlir/cvitek_envs.sh

# get cvimodel info
cd samples
./bin/cvi_sample_model_info $MODEL_PATH/mobilenet_v2.cvimodel
```

**其他samples运行命令参照EVB运行命令**

