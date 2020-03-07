#!/bin/bash
set -e

pushd llvm-project
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir
git pull --rebase
git submodule update
# NOTE:
# don't add "--remote", "--remote --merge", or "--remote rebase"
# update will ensure to checkout to the point that the repo designated
# IF you have local changes, please do "rebase" manually
popd

if false; then
pushd llvm-project/llvm/projects/mlir/externals/backend
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cvikernel
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cmodel
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cvibuilder
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/cviruntime
git pull --rebase
popd

pushd llvm-project/llvm/projects/mlir/externals/profiling
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

fi

echo "Done"
