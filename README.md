# tpu_compiler

晶视AI芯片的TPU编译器

## 代码下载与提交

* 下载代码

  ```shell
  # gerritmaster.cvitek.com当前ip是10.58.65.11
  $ git clone "ssh://xxxxxx@gerritmaster.cvitek.com:29418/tpu_compiler"
  $ cd tpu_compiler
  $ git submodule update --init
  ```

* 下载模型(可选)

  ``` shell
  # 下载和更新
  $ git clone "ssh://xxxxxx@gerritmaster.cvitek.com:29418/mlir-models" --depth=1
  $ export GIT_SSL_NO_VERIFY=1
  $ apt-get install git-lfs
  $ cd mlir-models
  $ git lfs install --skip-smudge
  $ git lfs pull *
  # 后续更新
  $ export GIT_SSL_NO_VERIFY=1
  $ export GIT_LFS_SKIP_SMUDGE=1
  $ git lfs pull --include imagenet/resnet/caffe/ResNet-50-model.caffemodel
  ```

* 提交代码

  ```shell
  # 先rebase，再提交
  $ git pull -r
  $ git push origin HEAD:refs/for/master
  ```

## 编译代码

* 下载docker和建立容器

  ```shell
  docker pull cvitek/cvitek_dev:1.7-ubuntu-18.04
  cd tpu_compiler
  docker run -it --name work -v $PWD:/work  -v $PWD/../mlir-models:/models cvitek/cvitek_dev:1.7-ubuntu-18.04
  ```

* 编译代码

  ```shell
  source ./envsetup.sh
  ./build.sh
  ```

## 代码验证

* 算子验证

  ``` shell
  mkdir tmp && cd tmp
  # 验证onnx算子
  test_onnx.py
  # 验证pytorch算子
  test_torch.py
  ```

* 模型验证

  ``` shell
  # 在tpu_compiler的同级
  cd regression
  ./run_regression.sh -n resnet50
  ```

## Release给客户

release给客户的有7个文件，都在武汉服务器(http://172.22.242.10)`/data/dailyrelease`目录，一般取最近一次jenkins成功的日期即可。

| 文件                                   | 说明                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| cvimodel_samples_cv18?x.tar.gz         |                                                              |
| cvitek_mlir_ubuntu-18.04.tar.gz        |                                                              |
| cvitek_tpu_samples.tar.gz              |                                                              |
| cvitek_tpu_sdk_cv18?x.tar.gz           |                                                              |
| cvitek_tpu_quick_start_guide.pdf       | **cvitek_tpu_quick_start_guide.md**，导出为pdf，找产品打上客户水印 |
| cvitek_tpu_development_manual.pdf      | **cvitek_tpu_development_manual.md**，同上操作               |
| docker_cvitek_dev_1.7-ubuntu-18.04.tar | 在服务器/data/docker目录                                     |
| version.txt                            | 版本号信息，可以直接添加到cvitek_mlir和cvitek_tpu_sdk文件名上    |

注意：cvitek_tpu_sdk有32位和uclic版本，问一下产品客户是用的哪一种。

#### 临时发布cvitek_mlir

有时候cvitek_mlir可能有个bug，需要当天解决后直接发给客户，操作如下：

1. 清理工程的所有文件，和确保补丁就位

2. 执行以下命令

   ``` shell
   rm -rf install
   rm -rf cvitek_mlir*
   ./build.sh RELEASE
   tpuc-interpreter --version ##查看版本，比如tpu_rel_v1.5.0-692-g907799966:20210701
   mv install cvitek_mlir
   tar -czf cvitek_mlir_ubuntu-18.04_v1.5.0-692-g907799966.tar.gz cvitek_mlir
   ```