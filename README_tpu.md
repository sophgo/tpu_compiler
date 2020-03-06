# mlir-tpu

## Quick Start

* get code and update

  ```sh
  $ git clone ssh://10.34.33.3:29418/llvm-project.git
  $ git clone --recurse-submodules ssh://10.34.33.3:29418/llvm-project.git
  $ mv mlir llvm-project/llvm/projects/

  # to update
  $ git submodule update --remote --rebase
  ```


* build from scratch

  ```sh
  $ source llvm-project/llvm/projects/mlir/envsetup.sh
  $ llvm-project/llvm/projects/mlir/build.sh
  ```

* build update

  ```sh
  $ source llvm-project/llvm/projects/mlir/envsetup.sh
  $ llvm-project/llvm/projects/mlir/build_update.sh
  ```

* regression

  ```sh
  $ llvm-project/llvm/projects/mlir/regression/run_regression.sh
  or
  $ llvm-project/llvm/projects/mlir/regression/run_regression.sh resnet50
  ```

* generic network regression and accuracy

  ```sh
  $ llvm-project/llvm/projects/mlir/regression/generic/regression_generic.sh mobilenet_v2
  $ llvm-project/llvm/projects/mlir/regression/generic/accuracy_generic.sh mobilenet_v2 50000
  ```

## Get Code and Build

### 0. Setup path

```
$ source envsetup.sh
```

### 1. Prerequsit

Tested with Ubuntu 18.04

* TODO: apt install list

### 2. Clone llvm-project and mlir

Clone llvm-project
```
$ git clone https://github.com/llvm/llvm-project.git
# checkout the certain point that we start the project
# TODO: update to latest later
$ git checkout -b mlir-tpu 21bc8631
```

Clone mlir into `llvm-project/llvm/projects` directory.

```
$ git clone git@xxx:mlir.git llvm-project/llvm/projects/mlir
$ cd llvm-project/llvm/projects/mlir
$ git checkout -b tpu origin/tpu
```

### 3. Third Party Libraries

Some libraries are tree build, some rely on manually build for now.\
Read third_party/README.md for details.

1. pybind11
2. Caffe (build and install to $TPU_BASE/install_caffe)
3. MKLDNN (unzip and install to $TPU_BASE/install_mlkdnn)
4. CNPY (tree build)

### 4. External Projects

All rely on manually build, except backend, backend is using tree build.\
Read externals/README.md for details.

1. python_tools
1. cvikernel
1. backend
1. cmodel (for testing only)
1. cvibuilder (for testing only)
1. support (for testing only)
1. runtime (for testing only)

### 5. Build mlir-tpu

```
$ cd llvm-project
$ mkdir build
$ cd build

$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH=$TPU_BASE/install_caffe -DMKLDNN_PATH=$TPU_BASE/install_mkldnn -DCVIKERNEL_PATH=$TPU_BASE/install_cvikernel -DCMAKE_INSTALL_PREFIX=$TPU_BASE/install_mlir -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir

# build pybind11 wrapper
$ cmake --build . --target pymlir
```

## Work flow

### 1. translate from caffe model to mlir tpu dialect

translate
```
$ ./bin/mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir
```

### 2. run inference with mlir-tpu-interpreter or python wrapper

run with interpreter
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
```

check
```
$ bin_dump.py out.bin float32 1 1 1 1000 5
$ bin_dump.py $DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
$ bin_compare.py out.bin $DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
```

with python bindings
```
$ python ../llvm/projects/mlir/bindings/python/tools/run_inference.py \
    resnet-50.mlir resnet50_input_1_3_224_224.npy 5
```

### 3. Pre-Quantization optimization

graph level, on float32 domain, with weight transform.

We support following pre-quantization optimzation

* convert-bn-to-scale
* fold-scale
* merge-scale-into-conv

To apply all passes
```
$ ./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    resnet-50.mlir \
    -o resnet-50-opt.mlir
```

check with inference
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-opt.bin
$ python bin_compare.py out.bin out-opt.bin float32 1 1 1 1000 5
```

### 4. calibration

The only information we need from the calibration process is a threshold value
for each neuron tensor (threshold_y). we use `calibration_caffe` result for now.

```
$ ./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt.mlir \
    -o resnet-50-cali.mlir
```

TODO: do calibration with mlir-interpreter pybind

### 5. Post-Calibration optimization

Some optimization need to take place before quantization but after calibration.

```
$ ./bin/mlir-opt \
    --fuse-relu \
    resnet-50-cali.mlir \
    -o resnet-50-opt-post-cali.mlir
```

### 6. quantization

int8 per-channel multiplier quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-int8.mlir
```

check with inference
```
$ ./bin/mlir-tpu-interpreter resnet-50-int8.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-int8.bin
$ bin_compare.py out.bin out-int8.bin float32 1 1 1 1000 5
```

### 7. Post-Quantization optimization

TODO

### 8. generate cmdbuf from tpu dialect

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-int8.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o resnet-50_cmdbuf.bin
```

### 8. run cmdbuf with runtime

quant-int8 per channel with multiplier

```
# run test
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight.bin \
    resnet-50_cmdbuf.bin \
    out_cmodel.bin \
    1000 150528 16460784 1
$ bin_dump.py out_cmodel.bin int8 1 1 1 1000 5

# to dump all neuron
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight.bin \
    resnet-50_cmdbuf.bin \
    out_all.bin \
    16460784 0 16460784 1
$ bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
$ bin_dump.py out_fc1000.bin int8 1 1 1 1000 5
```

compare output

```
$ ./bin/mlir-tpu-interpreter \
    resnet-50-int8.mlir \
    --tensor-in ~/work_cvitek/llvm-project/llvm/projects/mlir/data/test_cat_in_fp32.bin \
    --tensor-out out-int8.bin \
    --dump-all-tensor=tensor_all-int8.npz

$ bin_to_npz.py out_all.bin neuron_map.csv out_all.npz

$ npz_compare.py out_all.npz tensor_all-int8.npz
```
