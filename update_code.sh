#!/bin/bash
set -e

pushd llvm-project
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/backend
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/bmkernel
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cmodel
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cvibuilder
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/runtime
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/third_party/caffe
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/third_party/cnpy
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/third_party/pybind11
git pull --rebase
popd

echo "Done"
