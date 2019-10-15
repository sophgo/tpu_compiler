# mlir-tpu

## Get Code and Build

### 1. Prerequsit

Tested with Ubuntu 18.04

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
1. Caffe
1. MKLDNN
1. CNPY

### 4. External Projects

All rely on manually build.\
Read externals/README.md for details.

1. bmkernel
1. cmodel
1. runtime
1. builder

### 5. Build mlir-tpu

```
$ cd llvm-project
$ mkdir build
$ cd build

$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work_cvitek/install_caffe" -DMKLDNN_PATH="~/work_cvitek/install_mkldnn" -DBMKERNEL_PATH="~/work_cvitek/install_bmkernel" -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_mlir -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir
```

## Work flow

### 1. translate from caffe model to mlir tpu dialect

translate
```
$ ./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffemodel /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir
```

- output together with a weight file in npz format
- weight file name is described in .mlir file memref loadFile op
- each weight tensor save as a npy file inside the npz, with name. eg. conv1_0, conv1_1, etc.

check
```
$ vim resnet-50.mlir
$ python npz_list.py ResNet-50-model.npz
$ python npz_dump.py ResNet-50-model.npz conv1_0
```

### 2. run fp32 inference with mlir-tpu-interpreter

inference
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out.bin
```

* TODO: handle multiple outputs (use npz for inputs and output)

check
```
$ python bin_dump.py out.bin float32 1 1 1 1000 5
$ python bin_compare.py out.bin test_cat_out_fp32.bin float32 1 1 1 1000 5
```

### 3. Pre-Quantization optimization

graph level, on float32 domain, with weight transform

#### 3.1 convert bn to scale

```
$ ./bin/mlir-opt \
    --convert-bn-to-scale \
    resnet-50.mlir \
    -o resnet-50-opt1.mlir
```

check
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt1.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-opt1.bin
$ python bin_compare.py out.bin out-opt1.bin float32 1 1 1 1000 5
```

#### 3.2 fold scale

fold consecutive scale ops into one

```
$ ./bin/mlir-opt \
    --fold-scale \
    resnet-50-opt1.mlir \
    -o resnet-50-opt2.mlir
```

check
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt2.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-opt2.bin
$ python bin_compare.py out.bin out-opt2.bin float32 1 1 1 1000 5
```

#### 3.3 merge scale into conv

```
$ ./bin/mlir-opt \
    --fuse-scale-into-conv \
    resnet-50-opt2.mlir \
    -o resnet-50-opt3.mlir
```

check
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt3.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-opt3.bin
$ python bin_compare.py out.bin out-opt3.bin float32 1 1 1 1000 5
```

#### 3.4 Pass Manager

* TODO: to select and run multiple passes at once

### 4. calibration

The only information we need from the calibration process is a threshold value for each
activation tensor (threshold_y). The threshold is calculated based on histogram of each tensor during runtime. KLD is used to generate the threshold for now. Any other information (rshift, etc.) can be devived later in compiler.

we use calibration_caffe for now. But only use the threshold_y.

*TODO: do calibration based on mlir-interpreter

#### 4.1 import calibration-table from prototxt file

Import calibration table from externel file. The calibration table is simply a map of operation name vs. their threshold_y.

```
$ ./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt3.mlir \
    -o resnet-50-cali.mlir
```

#### 4.2 do calibration with mlir-interpreter

* TODO:

### 5. quantization

We do not import int8 caffemodel directly (the old version int8 caffemodel format is obsoleted). We do quantization from mlir fp32 model to mlir int8 model, based on threshold_y information of each operation.

#### 5.1 int8 per-layer quantization

```
$ cp ResNet-50-model-opt3.npz ResNet-50-model.npz
$ ./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8.mlir
```

check
```
$ vim resnet-50-quant-int8.mlir
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_rshift

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin
$ python bin_compare.py out.bin out-quant-int8.bin float32 1 1 1 1000 5
```

#### 5.2 int8 per-channel quantization

```
$ cp ResNet-50-model-opt3.npz ResNet-50-model.npz
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-per-channel.mlir
```

check
```
$ vim resnet-50-quant-int8-per-channel.mlir
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_rshift

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-per-channel.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-per-channel.bin
$ python bin_compare.py out.bin out-quant-int8-per-channel.bin float32 1 1 1 1000 5
```

#### 5.3 int8 per-channel multiplier quantization

```
$ cp ResNet-50-model-opt3.npz ResNet-50-model.npz
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir
```

check
```
$ vim resnet-50-quant-int8-multiplier.mlir
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_multiplier

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-multiplier.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin
$ python bin_compare.py out.bin out-quant-int8-multiplier.bin float32 1 1 1 1000 5
```

#### 5.3 bf16 quantization

* TODO

### 6. python wrapper for interpreter

#### 6.1 python wrapper

clone pybind11 into third_party
```
$ cd third_party
$ git clone https://github.com/pybind/pybind11
```

TODO: add as submodule

to build pymlir.so
```
$ cmake --build . --target pymlir
```

find pymlir.so in ./lib, to setup PYTHONPATH
```
$ export PYTHONPATH=./lib:$PYTHONPATH
```

#### 6.2 run inference

set PYTHONPATH first
```
$ python bin_to_npy.py \
    /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
    float32 1 3 224 224 \
    resnet50_input_1_3_224_224.npy
$ python ../llvm/projects/mlir/bindings/python/tools/run_inference.py \
    resnet-50.mlir resnet50_input_1_3_224_224.npy
```

#### 6.3 accuracy regression

use mxnet.gluon to load data
```
$ pip install --user mxnet
$ pip install --user gluoncv

$ python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
    --model=resnet-50.mlir \
    --dataset=/data/dataset/imagenet/img_val_extracted \
    --mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
    --count=100
```

result of resnet-50 accuracy (fp32, int8, int8-per-channel, int8-multiplier)

| mode | Top-1 accuracy | Top-5 accuracy |
| ---  | ---            | ---            |
| fp32             | 0.7820 | 0.9386 |
| int8 Per-layer   | 0.7789 | 0.9368 |
| int8 Per-channel | 0.7786 | 0.9392 |
| int8 Multiplier  | 0.7815 | 0.9391 |

### 7. calibration with interpreter (python version)

### 8. codegen from tpu dialect

#### 8.1 assign weight address and genenrate weight bin file

This also handle weight transpose if needed.
TODO: handle transpose more explicitly, and try to remove the unessesary transpose.

```
$ cp ResNet-50-model_quant_int8.npz ResNet-50-model.npz
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    resnet-50-quant-int8.mlir \
    -o resnet-50-quant-int8-addr1.mlir
```

#### 8.2 assign neuron address

```
$ ./bin/mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-addr1.mlir \
    -o resnet-50-quant-int8-addr2.mlir

[data                                ][  150528] : [ 0x00000000 --> 0x00024c00 ]
[fc1000                              ][    1008] : [ 0x00024c00 --> 0x00024ff0 ]
... ...
[scale_conv1                         ][  802816] : [ 0x01797ff0 --> 0x0185bff0 ]
```

#### 8.3 generate cmdbuf

use interpreter for now, need to refactor into translator
```
$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-addr2.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin
```
get a cmdbuf.bin

#### 8.4 run test with runtime

run test, `test_cat_in_int8.bin` is a int8 bin file. This is the quantization result of
`test_cat_in_fp32.bin`.

```
# quantize the input with its threshold
$ python ./bin_fp32_to_int8.py \
    test_cat_in_fp32.bin \
    test_cat_in_int8.bin \
    1 3 224 224 \
    161.008057

# run test
$ ./test/test_bmnet \
    test_cat_in_int8.bin \
    ~/work/llvm-project/build/ResNet-50-model.bin \
    ~/work/llvm-project/build/cmdbuf.bin \
    out_new.bin \
    1000 150528 25542640 1
$ python ./bin_dump.py out_new.bin int8 1 1 1 1000 5

# to dump all neuron
$ ./test/test_bmnet \
    test_cat_in_int8.bin \
    ~/work/llvm-project/build/ResNet-50-model.bin \
    ~/work/llvm-project/build/cmdbuf.bin \
    out_new.bin \
    25542640 0 25542640 1
$ python ./bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
$ python ./bin_dump.py out_fc1000.bin int8 1 1 1 1000 5
```

#### 8.5 compare output

To save bin file into npz, based on neuron_map.csv info.
```
$ python ./bin_to_npz.py out_all.bin neuron_map.csv out_all.npz

# see content
$ python ./npz_list.py out_all.npz
$ python ./npz_dump.py out_all.npz data_quant
$ python ./npz_dump.py out_all.npz fc1000 5
$ python ./npz_dump.py out_all.npz scale_conv1
```

Compare out_all.npz with the interpreter dump-all-tensor output.
```
$ python ./npz_compare.py out_all.npz tensor_all_quant-int8.npz int8 [show] [5]
```

#### 8.6 meta info

TODO: to output following information from compiler to runtime in a more formal way

- batch size
- input threshold
- output size (multiple output)
- output offset (multiple output)
- total neuron size

### 11. bmkernel to bmodel assembly

### 12. bmodel to bmkernel script disassembly

### 13. tg level optimization pass (no weight transform)

1.1 fuse activation into conv/fc

1.2 fuse pooling

### 14. tg to tl lowering

clustering/slice handling

### 15. auto clustering (layer_group)

### 16. affine and searching

## Debug tips

### print flags

put all debug print inside LLVM_DEBUG() macro

define "DEBUG_TYPE" for fine grained debug info

runtime
  -debug to enable all LLVM_DEBUG()
  -debug-only=dgb_type1,dgb_type2

DEBUG_TYPE defined
```
caffe-to-mlir               - caffe importer
caffe-to-mlir_VERBOSE       - caffe importer verbose
```

#### translate debug

```
$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffemodel /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffemodel /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffemodel /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug-only=caffe-to-mlir_VERBOSE

$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin
```

### dump all tensors

add the flag `--dump-all-tensor=` when run interpreter
```
--dump-all-tensor=tensor_all.npz
```

e.g.
```
$ ./bin/mlir-tpu-interpreter \
    resnet-50.mlir \
    --tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz

$ ./bin/mlir-tpu-interpreter \
    resnet-50-quant-int8.mlir \
    --tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz
```
