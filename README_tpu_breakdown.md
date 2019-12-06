# mlir-tpu

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

### 5. Build mlir-tpu

```
$ cd llvm-project
$ mkdir build
$ cd build

$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH=$TPU_BASE/install_caffe -DMKLDNN_PATH=$TPU_BASE/install_mkldnn -DBMKERNEL_PATH=$TPU_BASE/install_bmkernel -DCMAKE_INSTALL_PREFIX=$TPU_BASE/install_mlir -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir
```

build pybind11 wrapper
```
$ cmake --build . --target pymlir
# find pymlir.so in ./lib (assuming in build dir), to setup PYTHONPATH
$ export PYTHONPATH=./lib:$PYTHONPATH
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

- output together with a weight file in npz format
- weight file name is described in .mlir file memref loadFile op
- each weight tensor save as a npy file inside the npz,
with name. eg. conv1_0, conv1_1, etc.

check
```
$ vim resnet-50.mlir
$ python npz_list.py ResNet-50-model.npz
$ python npz_dump.py ResNet-50-model.npz conv1_0
```

### 2. run inference with mlir-tpu-interpreter or python wrapper

#### 2.1 run with interpreter

inference
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
```

* TODO: handle multiple outputs (use npz for inputs and output)

check
```
$ python bin_dump.py out.bin float32 1 1 1 1000 5
$ python bin_dump.py $DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
$ python bin_compare.py out.bin \
    $DATA_PATH/test_cat_out_fp32.bin float32 1 1 1 1000 5
```

#### 2.2 run with interpreter pybind

Convert input into npy, as current python test code take npy file as input.

```
$ python bin_to_npy.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    float32 1 3 224 224 \
    resnet50_input_1_3_224_224.npy
```

run inference
```
$ python ../llvm/projects/mlir/bindings/python/tools/run_inference.py \
    resnet-50.mlir resnet50_input_1_3_224_224.npy 5
```

#### 2.3 accuracy regression with pybind

Currently, we use mxnet.gluon to load data
```
$ pip install --user mxnet
$ pip install --user gluoncv
```

Run classification test, with accuracy output.
```
$ python ../llvm/projects/mlir/bindings/python/tools/run_classification.py \
    --model=resnet-50.mlir \
    --dataset=/data/dataset/imagenet/img_val_extracted \
    --mean_file=../llvm/projects/mlir/bindings/python/tools/mean_resize.npy \
    --count=100
```

result of resnet-50 accuracy with count=10000 (fp32, int8, int8-per-channel, int8-multiplier)

gluoncv (10000)
| mode | Top-1 accuracy | Top-5 accuracy |
| ---  | ---            | ---            |
| fp32             | 0.7318 | 0.9120 |
| int8 Per-layer   | 0.7206 | 0.9108 |
| int8 Per-channel | 0.7355 | 0.9146 |
| int8 Multiplier  | 0.7318 | 0.9120 |
| fp16             | 0.7240 | 0.9085 |

pytorch (10000)
| mode | Top-1 accuracy | Top-5 accuracy |
| ---  | ---            | ---            |
| fp32             | 0.7486 | 0.9221 |
| int8 Per-layer   | 0.7446 | 0.9171 |
| int8 Per-channel | 0.7498 | 0.9200 |
| int8 Multiplier  | 0.7476 | 0.9174 |
| fp16             | 0.7504 | 0.9223 |

mobilenet-v2
| shicai           | 0.7190 | 0.9049 |
| fp32 - gluoncv   | 0.7210 | 0.9068 |
| fp32 - pytorch   | 0.7178 | 0.9029 |

mobilenet-v1
| shicai           | 0.7081 | 0.8985 |
| fp32 - gluoncv   | 0.6990 | 0.8965 |
| fp32 - pytorch   | 0.7215 | 0.9073 |

20191119
gluoncv (5000)
| mode | Top-1 accuracy | Top-5 accuracy |
| ---  | ---            | ---            |
| fp32             | 0.7248 | 0.9102 |
| int8 Per-layer   | 0.7214 | 0.9136 |
| int8 Per-channel | 0.7196 | 0.9082 |
| int8 Multiplier  | 0.7120 | 0.9124 |
| fp16             | 0.7324 | 0.9114 |
pytorch (5000)
| mode | Top-1 accuracy | Top-5 accuracy |
| ---  | ---            | ---            |
| fp32             | 0.7424 | 0.9184 |
| int8 Per-layer   | 0.7386 | 0.9154 |
| int8 Per-channel | 0.7422 | 0.9142 |
| int8 Multiplier  | 0.7550 | 0.9238 |
| fp16             | 0.7616 | 0.9288 |

mobilenet-v2
| shicai           | 0.7190 | 0.9049 |
| fp32 - gluoncv   | 0.7202 | 0.9098 |
| fp32 - pytorch   | 0.7166 | 0.9054 |

mobilenet-v2 pytorch (10000)
| fp32             | 0.7129 | 0.9003 |
| int8 Per-layer   | 0.4339 | 0.6886 |
| int8 Per-channel | 0.6999 | 0.8861 |
| int8 Multiplier  | 0.7000 | 0.8908 |

mobilenet-v2 gluoncv (10000)
| fp32             | 0.7162 | 0.9020 |
| int8 Per-layer   | 0.4058 | 0.6728 |
| int8 Per-channel | 0.6789 | 0.8807 |
| int8 Multiplier  | 0.6801 | 0.8817 |

mobilenet-v1
| shicai           | 0.7081	| 0.8985 |
| fp32 - gluoncv   | 0.6960 | 0.8966 |
| fp32 - pytorch   | 0.7210 | 0.9040 |

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
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
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
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
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
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-opt3.bin
$ python bin_compare.py out.bin out-opt3.bin float32 1 1 1 1000 5
```

#### 3.4 All-in-one

```
$ ./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    resnet-50.mlir \
    -o resnet-50-opt.mlir
```

check
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-opt.bin
$ python bin_compare.py out.bin out-opt.bin float32 1 1 1 1000 5
```

### 4. calibration

The only information we need from the calibration process is a threshold value
for each neuron tensor (threshold_y). The threshold is calculated based on
histogram of each tensor during runtime. KLD is used to generate the threshold
for now. Other information (rshift, multiplier, etc., either per-layer or
per-channel) are derived in compiler, based on the threshold_y and the value of
weight.

we use `calibration_caffe` for now. But use `threshold_y` of each layer only.

*TODO: do calibration based on mlir-interpreter

#### 4.1 import calibration-table from prototxt file

Import calibration table from externel file. The calibration table is simply
a map of operation name vs. their threshold_y.

```
$ ./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_PATH/bmnet_resnet50_calibration_table.1x10 \
    resnet-50-opt.mlir \
    -o resnet-50-cali.mlir
```

#### 4.2 do calibration with mlir-interpreter python wrapper

* TODO:

### 5. Post-Calibration optimization

Some optimization need to take place before quantization but after calibration.

#### 5.1 merge activation function into conv/eltwise/fullyconnect Ops

This needs to be done after calibration because we need to check threshold_y range
of both layers before merge them. This is need to be done before quantization,
because some activation functions, like `relu6`, will impose a new threshold_y to
the merged preceding Ops, therefore affect the quantization precess.

```
$ ./bin/mlir-opt \
    --fuse-relu \
    resnet-50-cali.mlir \
    -o resnet-50-opt-post-cali.mlir
```

check
```
$ ./bin/mlir-tpu-interpreter resnet-50-opt-post-cali.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-opt-post-cali.bin
$ python bin_compare.py out.bin out-opt-post-cali.bin float32 1 1 1 1000 5
```

### 6. quantization

We do not import int8 caffemodel directly (the legacy version int8 caffemodel
format is obsoleted). We do quantization from mlir fp32 model to mlir int8
model, based on the `threshold_y` of each layer.

Before we try different version of quantization, save the weight file first
```
$ cp ResNet-50-model.npz  ResNet-50-model-opt.npz
```

#### 6.1 int8 per-layer quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-quant-int8.mlir
```

check
```
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_rshift

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin
$ python bin_compare.py out.bin out-quant-int8.bin float32 1 1 1 1000 5
```

#### 6.2 int8 per-channel quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-quant-int8-per-channel.mlir
```

check
```
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_rshift

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-per-channel.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-per-channel.bin
$ python bin_compare.py out.bin out-quant-int8-per-channel.bin float32 1 1 1 1000 5
```

#### 6.3 int8 per-channel multiplier quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet-50-opt-post-cali.mlir \
    -o resnet-50-quant-int8-multiplier.mlir
```

check
```
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_0
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_1
$ python npz_dump.py ResNet-50-model.npz scale_conv1_quant_int8_multiplier

$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-multiplier.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin
$ python bin_compare.py out.bin out-quant-int8-multiplier.bin float32 1 1 1 1000 5
```

#### 6.4 bf16 quantization

* TODO

### 7. Post-Quantization optimization

Some optimizations need to take place after quantization.

### 8. codegen from tpu dialect

#### 8.1 assign weight address and genenrate weight bin file

This also handle weight transpose if needed.

* TODO: handle transpose more explicitly, and try removing the unessesary transpose.

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8.bin \
    resnet-50-quant-int8.mlir \
    -o resnet-50-quant-int8-addr1.mlir
```

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8-multiplier.bin \
    resnet-50-quant-int8-multiplier.mlir \
    -o resnet-50-quant-int8-multiplier-addr1.mlir
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

```
$ ./bin/mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-multiplier-addr1.mlir \
    -o resnet-50-quant-int8-multiplier-addr2.mlir

[data                                ][  150528] : [ 0x00000000 --> 0x00024c00 ]
[fc1000                              ][    1008] : [ 0x00024c00 --> 0x00024ff0 ]
... ...
[scale_conv1                         ][  802816] : [ 0x01797ff0 --> 0x0185bff0 ]
```

#### 8.3 generate cmdbuf

per-layer int8

```
$ ./bin/mlir-translate resnet-50-quant-int8-addr2.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin
```

per-channel-multplier int8

```
$ ./bin/mlir-translate resnet-50-quant-int8-multiplier-addr2.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf-multiplier.bin
```

#### 8.4 all-in-one

per-layer int8

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin
```

per-channel-multplier int8

```
$ ./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet-50-quant-int8-multiplier.mlir | \
  ./bin/mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf-multiplier.bin
```

#### 8.5 run cmdbuf test with runtime

run test, `test_cat_in_int8.bin` is a int8 bin file. This is the quantization result of
`test_cat_in_fp32.bin`.
```
# quantize the input with its threshold
$ bin_fp32_to_int8.py \
    test_cat_in_fp32.bin \
    test_cat_in_int8.bin \
    1.0 \
    161.008057
```

quant-int8 per layer
```
# run test
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight_int8.bin \
    cmdbuf.bin \
    out_cmodel.bin \
    1000 150528 25542640 1
$ bin_dump.py out_cmodel.bin int8 1 1 1 1000 5

# to dump all neuron
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight_int8.bin \
    cmdbuf.bin \
    out_all.bin \
    25542640 0 25542640 1
$ bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
$ bin_dump.py out_fc1000.bin int8 1 1 1 1000 5
```

quant-int8 per channel with multiplier
```
# run test
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight_int8-multiplier.bin \
    cmdbuf-multiplier.bin \
    out_cmodel.bin \
    1000 150528 25542640 1
$ bin_dump.py out_cmodel.bin int8 1 1 1 1000 5

# to dump all neuron
$ $TPU_BASE/install_runtime/bin/test_bmnet \
    $DATA_PATH/test_cat_in_int8.bin \
    weight_int8-multiplier.bin \
    cmdbuf-multiplier.bin \
    out_all.bin \
    25542640 0 25542640 1
$ bin_extract.py out_all.bin out_fc1000.bin int8 0x00024c00 1000
$ bin_dump.py out_fc1000.bin int8 1 1 1 1000 5
```

#### 8.5 compare output

To save bin file into npz, based on neuron_map.csv info.

```
$ bin_to_npz.py out_all.bin neuron_map.csv out_all.npz

# see content
$ npz_list.py out_all.npz
$ npz_dump.py out_all.npz data_quant
$ npz_dump.py out_all.npz fc1000 5
$ npz_dump.py out_all.npz scale_conv1
```

To generate reference `dump-all-tensor`
```
$ ./bin/mlir-tpu-interpreter \
    resnet-50-quant-int8.mlir \
    --tensor-in ~/work_cvitek/llvm-project/llvm/projects/mlir/data/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz

# compare out_all.npz with the interpreter dump-all-tensor output.
$ npz_compare.py out_all.npz tensor_all_quant-int8.npz [show] [5]
```

```
$ ./bin/mlir-tpu-interpreter \
    resnet-50-quant-int8-multiplier.mlir \
    --tensor-in ~/work_cvitek/llvm-project/llvm/projects/mlir/data/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin \
    --dump-all-tensor=tensor_all_quant-int8-multiplier.npz

# compare out_all.npz with the interpreter dump-all-tensor output.
$ npz_compare.py out_all.npz tensor_all_quant-int8-multiplier.npz [show] [5]
```

#### 8.6 meta info

TODO: to output following information from compiler to runtime in a more formal way

- batch size
- input threshold
- output size (multiple output)
- output offset (multiple output)
- total neuron size

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
$ ./bin/mlir-translate --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir

$ ./bin/mlir-translate --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug

$ ./bin/mlir-translate --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug-only=caffe-to-mlir_VERBOSE

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

## Regression

### 1. fp32

Translate from caffe into mlir.

Run original mlir interpreter, check output are `similarity @1e-5` against reference output.

Run opt1, opt2, opt3 optimization, and run interpreter respectly, check output  `similarity @1e-5` against the orignal output.

### 2. int8

Quantize to int8, using int8, pre-channel int8 and multiplier int8
Run interperter, check output against pre-saved output, need to be bit-accurate same as the output.
The pre-saved output has been manually check against the fp32 output. (as well pass run_accuracy test)

### 3. cmdbuf

Run interpreter with --dump-tensor-all
Run test_bmnet, output all neuron
Compare all neuron with dump-tensor-all result, need to be bit-accurate same.

### 4. run accuracy

(This is too slow, skip it in commit regression, maybe run it daily)
