To build
$ cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCAFFE_PATH="~/work/caffe" -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON
$ cmake --build . --target check-mlir

Extra regression
$ ./bin/mlir-translate --caffe-to-mlir-v2 /data/models/caffe/ResNet-50-deploy.prototxt -o resnet-v2.mlir
$ diff resnet-v2.mlir ../llvm/projects/mlir/test/Dialect/TPU/resnet-50_20190816_29a4a80f-v2.mlir
