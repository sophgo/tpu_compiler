# mlir-tpu

## build

### prerequsit

1. Caffe

```
$ cd ~/work
$ git clone caffe_int8

$ mkdir build
$ mkdir install
$ cd build
$ cmake ..
$ cmake -DUSE_OPENCV=OFF -DCMAKE_INSTALL_PREFIX=../install ..
$ make -j20 all
$ make install
```

2. MKLDNN

https://github.com/intel/mkl-dnn/releases

```
$ mkdir -p ~/work/MKLDNN
$ cd ~/work/MKLDNN
$ wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz
$ tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz
$ ln -s mkldnn_lnx_1.0.2_cpu_gomp install
```

3. CNPY

https://github.com/rogersce/cnpy

```
$ git clone https://github.com/rogersce/cnpy.git
$ cd ~/work/cnpy
$ mkdir build
$ mkdir install
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=../install ..
$ cmake --build . --target install
```

### build mlir

```
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work/caffe/install" -DMKLDNN_PATH="~/work/MKLDNN/install" -DCNPY_PATH="~/work/cnpy/install" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

# link with caffe_int8 project
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work_xtalvision/install_caffe" -DMKLDNN_PATH="~/work/MKLDNN/install" -DCNPY_PATH="~/work/cnpy/install" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir
```

## Work flow

### 1. translate from caffe model to mlir tpu dialect

translate
```
$ ./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffe-model /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet-50.mlir

- output together with a weight file in npz format
- weight file name is described in .mlir file memref loadFile op
- each weight tensor save as a npy file inside the npz, with a array name. eg. conv1_0, conv1_1, etc.
```

check
```
$ vim resnet-50.mlir
$ python npz_list.py ResNet-50-model.npz
```

### 2. run fp32 inference with mlir-tpu-interpreter

inference
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin
```

* TODO: handle multiple outputs (use npz for inputs and output)

check
```
$ python bin_dump.py out.bin float32 1 1 1 1000 5
$ python bin_compare.py out.bin /data/release/bmnet_models/resnet50/resnet50_output_1_3_224_224_ref.bin float32 1 1 1 1000 5
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
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
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
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
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
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
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

```
$ ./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table bmnet_resnet50_calibration_table.1x10 \
    resnet-50.mlir \
    -o resnet-50-cali.mlir
```

#### 4.2 do calibration with mlir-interpreter

* TODO: finish python wrapper first

### 5. quantization

We do not import int8 caffemodel directly (the old version int8 caffemodel format is obsoleted). We do quantization from mlir fp32 model to mlir int8 model, based on threshold_y information of each operation.

#### 5.1 int8 per-layer quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-per-layer.mlir
```

#### 5.2 int8 per-channel quantization

```
$ ./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet-50-cali.mlir \
    -o resnet-50-quant-int8-per-channel.mlir
```

#### 5.3 bf16 quantization

* TODO

### 6. run quantized model with interpreter

### 7. python wrapper for interpreter

### 8. calibration with interpreter

### 9. accuracy regression

### 10. codegen from tpu dialect

Codegen into bmkernel script (asm)

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

### translate debug

```
$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug-only=caffe-to-mlir_VERBOSE

$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin
```
