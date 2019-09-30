# To build

## prerequsit

1. Caffe
```
$ cd ~/work
$ git clone git@gitlab.com:learndl/caffe.git
$ git checkout -b mytpu 04ab089d
$ git push -u origin mytpu
$ git checkout -b mytpu origin/mytpu

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

## build
```
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work/caffe/install" -DMKLDNN_PATH="~/work/MKLDNN/install" -DCNPY_PATH="~/work/cnpy/install" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

# link to caffe_int8 project
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work_xtalvision/install_caffe" -DMKLDNN_PATH="~/work/MKLDNN/install" -DCNPY_PATH="~/work/cnpy/install" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON

$ cmake --build . --target check-mlir
```

### Extra regression

```
$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug

$ ./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50.mlir --debug-only=caffe-to-mlir_VERBOSE

$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin
```

v2
```
$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model-v2 /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50-v2.mlir

$ diff resnet-50-v2.mlir ../llvm/projects/mlir/resnet-50_20190921_e92b6c5c.mlir

$ ./bin/mlir-opt -print-tpu-op-stats -verify-each=true resnet-50-v2.mlir
$ ./bin/mlir-opt -print-tpu-op-stats-v0 -verify-each=true resnet-50-v2.mlir

$ ./bin/mlir-tpu-interpreter resnet-50-v2.mlir
```

### bmnet model
```
$ ./bin/mlir-translate --caffe-to-mlir-v2 \
/data/release/bmnet_models/resnet50/resnet50_deploy.prototxt \
--caffe-model /data/release/bmnet_models/resnet50/resnet50.caffemodel \
-o resnet-50-v2.mlir

$ ./bin/mlir-tpu-interpreter resnet-50-v2.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin

$ ~/work/my_models/tests/numpy_bin_dump.py out.bin float32 1 1 1 1000 5
$ ~/work/my_models/tests/numpy_bin_dump.py /data/release/bmnet_models/resnet50/resnet50_output_1_3_224_224_ref.bin float32 1 1 1 1000 5
```

### bmnet int8 model

int8 caffemodel weight is still stored as float32, however they have been quantized.
i.e. with in range (-127, 128)

quantization-table is provided outside of the caffemodel.
quantization-table contains 3 types information
- threshold_y
- right_shift_width
- threshold_x_quantized

```
$ ./bin/mlir-translate --caffe-to-mlir-v3 \
/data/release/bmnet_models/resnet50/resnet50_deploy.prototxt \
--caffe-model-int8 ~/work_xtalvision/calibration_caffe/build/bmnet_resnet50_int8.1x10.caffemodel \
--quant-table ~/work_xtalvision/calibration_caffe/build/bmnet_resnet50_calibration_table.1x10.prototxt \
-o resnet-50_int8-v3.mlir

$ ./bin/mlir-tpu-interpreter resnet-50_int8-v3.mlir \
--tensor-in /data/release/bmnet_models/resnet50/int8/resnet50_input_1_3_224_224.bin \
--tensor-out out_int8.bin

$ ~/work/my_models/tests/numpy_bin_dump.py out_int8.bin int8 1 1 1 1000 5
$ ~/work/my_models/tests/numpy_bin_dump.py /data/release/bmnet_models/resnet50/int8/resnet50_output_1_3_224_224_ref_int8.bin int8 1 1 1 1000 5
```

# Work flow

sample nvdla flow
```
./nvdla_compiler [-options] --prototxt <prototxt_file> --caffemodel <caffemodel_file> -o <outputpath>
./nvdla_compiler -h
./nvdla_runtime --loadable <loadable_file>
./nvdla_runtime --loadable <loadable_file> --image <image_file>
./nvdla_runtime -s
```

## 1. translate from caffe model to tpu dialect

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

## 2. run fp32 inference with interpreter

inference
```
$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin

- TODO: handle multiple outputs
```

check
```
$ python bin_dump.py out.bin float32 1 1 1 1000 5
$ diff out.bin /data/release/bmnet_models/resnet50/resnet50_output_1_3_224_224_ref.bin
```

## 3. model level optimization (with weight transform)

### 3.1 convert bn to scale

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
```

### 3.2 fold scale

fold multiple scale into one

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
```

### 3.3 merge scale into conv

### 3.4 merge relu into conv

## 4. calibration

The only information we need from the calibration process is to obtain a threshold value for each
activation tensor. The threshold is calculated based on histogram of each tensor during runtime.
KLD is used to generate the threshold for now. All other information can be devived later in compiler.

we use calibration_caffe for now. TODO: do calibration based on mlir-interpreter

## 5. quantization

We do not import int8 caffemodel directly (the old version int8 caffemodel format is obsoleted). We convert from mlir fp32 into mlir int8, based on the calibration table (a map of tensor name and its threshold).
```
$ ./bin/mlir-opt \
    --quantization-int8 \
    resnet-50.mlir \
    -o resnet-50-int8.mlir
```

## 6. run int8 inference with interpreter

## 7. python wrapper for interpreter

## 8. calibration with interpreter

## 9. accuracy regression

## 10. codegen from tpu dialect

Codegen into bmkernel script (asm)

## 11. bmkernel to bmodel assembly

## 12. bmodel to bmkernel script disassembly

## 13. tg level optimization pass (no weight transform)

1.1 fuse activation into conv/fc

1.2 fuse pooling

## 14. tg to tl lowering

clustering/slice handling

## 15. auto clustering (layer_group)

## 16. affine and searching

# Debug tips

put all debug print inside LLVM_DEBUG() macro

define "DEBUG_TYPE" for fine grained debug info

runtime
  -debug to enable all LLVM_DEBUG()
  -debug-only=dgb_type1,dgb_type2

DEBUG_TYPE defined
```
caffe-to-mlir               - caffe importer
caffe-to-mlir_VERBOSE       - caffe importer verbose
caffe-to-mlir-v2            - caffe importer v2
caffe-to-mlir-v2_VERBOSE    - caffe importer v2
caffe-to-mlir-v3            - caffe importer v3
```
