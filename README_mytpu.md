# To build
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work/caffe" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON
$ cmake --build . --target check-mlir

Extra regression
$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt -o resnet-v2.mlir
$ diff resnet-v2.mlir ../llvm/projects/mlir/resnet-50_20190816_29a4a80f-v2.mlir

$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt --caffe-model /data/models/caffe/ResNet-50-model.caffemodel -o resnet-v2.mlir

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
