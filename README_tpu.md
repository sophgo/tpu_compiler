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

# Build Release

  ```sh
  $ MLIR_INSTALL_PATH=$PWD/cvitek_mlir SDK_INSTALL_PATH=$PWD/cvitek_tpu_sdk \
      source ./llvm-project/llvm/projects/mlir/envsetup.sh
  $ BUILD_OPENCV=1 BUILD_SAMPLES=1 build_soc.sh
  $ build.sh RELEASE
  ```

  Run regression

  ```sh
  $ source cvitek_mlir/cvitek_envs.sh
  $
  ```
