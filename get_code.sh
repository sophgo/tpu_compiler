#!/bin/bash
set -e

#  please add "User user.name" in ~/.ssh/config
# EX:
#Host *
#        KexAlgorithms +diffie-hellman-group1-sha1
#  User user.name  <------ gerrit account
#

git clone ssh://10.34.33.3:29418/llvm-project.git
#push llvm-project
#git checkout -b tpu origin/tpu
#pop

git clone ssh://10.34.33.3:29418/mlir.git
#pushd mlir
#git checkout -b tpu origin/tpu
#popd

git clone ssh://10.34.33.3:29418/cnpy.git
#pushd cnpy
#git checkout -b tpu origin/tpu
#popd

git clone ssh://10.34.33.3:29418/caffe.git
#pushd caffe
#git checkout -b tpu origin/tpu_master
#popd

git clone ssh://10.34.33.3:29418/pybind11.git
git clone ssh://10.34.33.3:29418/flatbuffers.git
git clone ssh://10.34.33.3:29418/systemc-2.3.3.git

git clone ssh://10.34.33.3:29418/cvikernel.git
git clone ssh://10.34.33.3:29418/backend.git
git clone ssh://10.34.33.3:29418/cmodel.git
git clone ssh://10.34.33.3:29418/cviruntime.git
git clone ssh://10.34.33.3:29418/cvibuilder.git
git clone ssh://10.34.33.3:29418/profiling.git

#export GIT_LFS_SKIP_SMUDGE=1
#git clone ssh://10.34.33.3:29418/mlir-models.git
#pushd mlir-models
#git lfs install
#git lfs ls-files
## git lfs fetch --all
#git lfs pull -I "*.caffemodel"
#popd

mv cnpy mlir/third_party/
mv caffe mlir/third_party/
mv pybind11 mlir/third_party/
mv flatbuffers mlir/third_party/
mv systemc-2.3.3 mlir/third_party/

mv cvikernel mlir/externals/
mv backend mlir/externals/
mv cmodel mlir/externals/
mv cviruntime mlir/externals/
mv cvibuilder mlir/externals/
mv profiling mlir/externals/

mv mlir llvm-project/llvm/projects/

