# mlir-tpu

## Quick Start

* get code

  ```sh
  $ git clone ssh://10.34.33.3:29418/llvm-project.git
  $ git clone --recurse-submodules ssh://10.34.33.3:29418/mlir.git
  $ mv mlir llvm-project/llvm/projects/
  ```

* update code

  ```sh
  $ cd llvm-project/llvm/projects/mlir
  $ get fetch origin
  $ get rebase origin/tpu
  $ git submodule update
  # NOTE:
  # don't add "--remote", "--remote --merge", or "--remote rebase"
  # update will ensure to checkout to the point that the repo designated
  # IF you have local changes, please do "rebase" manually
  ```

* build from scratch

  ```sh
  $ source llvm-project/llvm/projects/mlir/envsetup.sh
  $ llvm-project/llvm/projects/mlir/build.sh
  ```

* build for update

  ```sh
  $ source llvm-project/llvm/projects/mlir/envsetup.sh
  $ llvm-project/llvm/projects/mlir/build_update.sh
  ```

* build for soc

  ```sh
  $ source llvm-project/llvm/projects/mlir/envsetup.sh
  $ llvm-project/llvm/projects/mlir/build_soc.sh
  # or (this will build all including soc though)
  $ llvm-project/llvm/projects/mlir/build.sh SOC
  ```

* run regression

  ```sh
  $ llvm-project/llvm/projects/mlir/regression/run_regression.sh
  or
  $ llvm-project/llvm/projects/mlir/regression/run_regression.sh resnet50
  ```

* run generic network regression and accuracy

  ```sh
  $ llvm-project/llvm/projects/mlir/regression/generic/regression_generic.sh mobilenet_v2
  $ llvm-project/llvm/projects/mlir/regression/generic/accuracy_generic.sh mobilenet_v2 50000
  ```

## Prerequsit

Tested with Ubuntu 18.04

* TODO: apt install list
* TODO: pip install list

## Build Release

  Build mlir and sdk

  ```sh
  $ MLIR_INSTALL_PATH=$PWD/cvitek_mlir SDK_INSTALL_PATH=$PWD/cvitek_tpu_sdk \
      source ./llvm-project/llvm/projects/mlir/envsetup.sh
  $ BUILD_OPENCV=1 BUILD_SAMPLES=1 build_soc.sh
  $ build.sh RELEASE
  Find `cvitek_mlir` and `cvitek_tpu_sdk` dirs
  $ tar zcf cvitek_tpu_sdk_vx.y.tar.gz cvitek_tpu_sdk
  $ tar zcf cvitek_mlir_vx.y.tar.gz cvitek_mlir
  ```

  Run MLIR regression and generate cvimodel release

  ```sh
  $ tar zxf cvitek_mlir.tar.gz
  $ source cvitek_mlir/cvitek_envs.sh
  $ ./cvitek_mlir/regression/parallel/paralle_test.sh
  Find `cvimodel_regression` in `regression_out/`
  $ mv regression_out/cvimodel_regression
  $ tar zcf cvimodel_regression.tar.gz cvimodel_regression
  Generate cvimodel release
  $ ./cvitek_mlir/regression/generate_all_cvimodels.sh
  Find `cvimodel_release`
  $ tar zcf cvimodel_release.tar.gz cvimodel_release
  ```

  Test SDK on target board

  ```sh
  $ tar zxf cvitek_tpu_sdk.tar.gz
  $ export TPU_ROOT=$PWD/cvitek_tpu_sdk

  $ tar zxf cvimodel_release.tar.gz
  $ export MODEL_PATH=$PWD/cvimodel_release
  $ $TPU_ROOT/samples/bin/cvi_sample_model_info \
      $MODEL_PATH/mobilenet_v2.cvimodel
  $ $TPU_ROOT/samples/run_classifier.sh
  $ $TPU_ROOT/samples/run_detector.sh
  $ $TPU_ROOT/samples/run_alphapose.sh
  $ $TPU_ROOT/samples/run_insightface.sh

  $ tar zxf cvimodel_regression.tar.gz
  $ export PATH=$TPU_ROOT/bin:$PATH
  $ MODEL_PATH=$PWD/cvimodel_regression $TPU_ROOT/regression_models.sh
  ```
