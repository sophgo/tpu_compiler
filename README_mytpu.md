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

## build
```
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work/caffe" -DMKLDNN_PATH="~/work/MKLDNN" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON
$ cmake --build . --target check-mlir
```

# Extra regression
```
$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt -o resnet-50-v2.mlir

$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-50-v2.mlir

$ diff resnet-50-v2.mlir ../llvm/projects/mlir/resnet-50_20190901_d46a84fb.mlir

$ ./bin/mlir-opt -print-tpu-op-stats -verify-each=true resnet-50-v2.mlir
$ ./bin/mlir-opt -print-tpu-op-stats-v0 -verify-each=true resnet-50-v2.mlir

$ ./bin/mlir-tpu-interpreter resnet-50-v2.mlir
```

bmnet model
```
$ ./bin/mlir-translate --caffe-to-mlir-v2 \
/data/release/bmnet_models/resnet50/resnet50_deploy.prototxt \
--caffe-model /data/release/bmnet_models/resnet50/resnet50.caffemodel \
-o resnet-50.mlir

$ ./bin/mlir-tpu-interpreter resnet-50.mlir \
--tensor-in /data/release/bmnet_models/resnet50/resnet50_input_1_3_224_224.bin \
--tensor-out out.bin

$ ~/work/my_models/tests/numpy_bin_dump.py out.bin float32 1 1 1 1000
$ ~/work/my_models/tests/numpy_bin_dump.py /data/release/bmnet_models/resnet50/resnet50_output_1_3_224_224_ref.bin float32 1 1 1 1000
```

# User work flow
1. translate from caffe mode to tg dialect
```
./bin/mlir-translate --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt -o resnet.mlir

./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/ResNet-50-deploy.prototxt \
    --caffe-model /data/models/caffe/ResNet-50-model.caffemodel \
    -o resnet.mlir

- output a weight.bin
- weight.bin file name is described in .mlir file memref load op (or some op)
  - with total size to check
  - [-a weight_align_size]
- in mlir, each weight tensor has an offset attribute describing the offset in weight.bin
```

2. run tg net fp32 inference with cpu
```
./bin/mlir-tpu-interpreter resnet.mlir -i input.bin -o output.bin
```

3. model level optimization (with weight transform)
3.1 fuse bn/scale into conv
```
./bin/mlir-opt --fuse-bn-scale-into-conv resnet.mlir -o resnet-opt.mlir
```

4. quantization, conversion to tg int8 dialect

5. (extra) translate from caffe int8 model to tg int8 dialect

6. run tg net int8 inference with cpu

7. codegen directly from tg dialect into bmkernel script (asm)

8. bmkernel to bmodel assembly

9. bmodel to bmkernel script disassembly

10. tg level optimization pass (no weight transform)
10.1 fuse activation into conv/fc

10.2 fuse pooling

11. tg to tl lowering
clustering/slice handling

12. auto clustering (layer_group)
