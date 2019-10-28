# mlir-tpu

## Get Code and Build

### 0. Setup path

```
$ export TPU_BASE=~/work_cvitek
$ export TPU_DATA_PATH=$TPU_BASE/llvm-project/llvm/projects/mlir/data
$ export TPU_MODEL_PATH=/data/models
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
$ git checkout -b mydev 6d5a8c92b
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
1. Caffe (build and install to $TPU_BASE/install_caffe)
1. MKLDNN (unzip and install to $TPU_BASE/install_mlkdnn)
1. CNPY (tree build)

### 4. External Projects

All rely on manually build, except backend, backend is using tree build.\
Read externals/README.md for details.

1. python_tools
1. bmkernel
1. backend
1. cmodel (for testing only)
1. bmbuilder (for testing only)
1. support (for testing only)
1. runtime (for testing only)

setup external lib paths
```
$ export LD_LIBRARY_PATH=$TPU_BASE/install_bmkernel/lib:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$TPU_BASE/install_support/lib:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$TPU_BASE/install_bmbuilder/lib:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$TPU_BASE/install_cmodel/lib:$LD_LIBRARY_PATH
```

### 5. Build mlir-tpu

```
$ cd llvm-project
$ mkdir build
$ cd build

$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH=$TPU_BASE/install_caffe -DMKLDNN_PATH=$TPU_BASE/install_mkldnn -DBMKERNEL_PATH=$TPU_BASE/install_bmkernel -DCMAKE_INSTALL_PREFIX=$TPU_BASE/install_mlir -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir

# build pybind11 wrapper
$ cmake --build . --target pymlir
```

## Work flow

### 1. translate from caffe model to mlir tpu dialect

translate
```
$ ./bin/mlir-translate \
    --caffe-to-mlir $TPU_MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --caffemodel $TPU_MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir
```

check with inference
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
    --tensor-in $TPU_DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
```

* TODO: handle multiple outputs (use npz for inputs and output)

check
```
$ python bin_dump.py out.bin float32 1 1 1 1000 5
$ python bin_dump.py $TPU_DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
$ python bin_compare.py out.bin $TPU_DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
```

### 2. Pre-Quantization optimization

graph level, on float32 domain, with weight transform.

We support following pre-quantization optimzation

* convert-bn-to-scale
* fold-scale
* fuse-scale-into-conv

To apply all passes
```
$ ./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    resnet-50.mlir \
    -o resnet-50-opt.mlir
```

check with inference
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt.mlir \
    --tensor-in $TPU_DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-opt.bin
$ python bin_compare.py out.bin out-opt.bin float32 1 1 1 1000 5
```

### 3. quantization

int8 per-channel multiplier quantization

```
$ ./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $TPU_DATA_PATH/bmnet_resnet50_calibration_table.1x10 \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-opt.mlir \
    -o resnet-50-int8.mlir
```

check with inference
```
$ ./bin/mlir-tpu-interpreter resnet-50-int8.mlir \
    --tensor-in $TPU_DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-int8.bin
$ python bin_compare.py out.bin out-int8.bin float32 1 1 1 1000 5
```

### 5. generate cmdbuf from tpu dialect

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-int8.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o resnet-50_cmdbuf.bin
```

### 6. run cmdbuf with runtime

quant-int8 per channel with multiplier

```
# run test
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $TPU_DATA_PATH/test_cat_in_int8.bin \
    ResNet-50-model.bin \
    resnet-50_cmdbuf.bin \
    out_cmodel.bin \
    1000 150528 25542640 1
$ python ./bin_dump.py out_cmodel.bin int8 1 1 1 1000 5

# to dump all neuron
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $TPU_DATA_PATH/test_cat_in_int8.bin \
    ResNet-50-model.bin \
    resnet-50_cmdbuf.bin \
    out_all.bin \
    25542640 0 25542640 1
$ python ./bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
$ python ./bin_dump.py out_fc1000.bin int8 1 1 1 1000 5
```

compare output

```
$ ./bin/mlir-tpu-interpreter \
    resnet-50-int8.mlir \
    --tensor-in ~/work_cvitek/llvm-project/llvm/projects/mlir/data/test_cat_in_fp32.bin \
    --tensor-out out-int8.bin \
    --dump-all-tensor=tensor_all-int8.npz

$ python ./bin_to_npz.py out_all.bin neuron_map.csv out_all.npz

$ python ./npz_compare.py out_all.npz tensor_all-int8.npz int8
```
