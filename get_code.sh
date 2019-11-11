#!/bin/bash
set -e

git clone ssh://git@10.34.33.3:8422/mlir-tpu/llvm-project.git
#push llvm-project
#git checkout -b tpu origin/tpu
#pop

git clone ssh://git@10.34.33.3:8422/mlir-tpu/mlir.git
#pushd mlir
#git checkout -b tpu origin/tpu
#popd

git clone ssh://git@10.34.33.3:8422/mlir-tpu/cnpy.git
#pushd cnpy
#git checkout -b tpu origin/tpu
#popd

git clone ssh://git@10.34.33.3:8422/mlir-tpu/pybind11.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/caffe.git

git clone ssh://git@10.34.33.3:8422/mlir-tpu/python_tools.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/backend.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/bmkernel.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/support.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/cmodel.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/runtime.git
git clone ssh://git@10.34.33.3:8422/mlir-tpu/bmbuilder.git

mv cnpy mlir/third_party/
mv pybind11 mlir/third_party/
mv caffe mlir/third_party/

mv python_tools mlir/externals/
mv backend mlir/externals/
mv bmkernel mlir/externals/
mv support mlir/externals/
mv cmodel mlir/externals/
mv runtime mlir/externals/
mv bmbuilder mlir/externals/

mv mlir llvm-project/llvm/projects/

