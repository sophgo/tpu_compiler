# Third_party projects

Third party open source projects used by mlir-tpu.

Some of them are using tree build, some rely on manually build for now.

## pybind11

- https://github.com/pybind/pybind11
- original version (tested with checking out commit 34c2281e)
- mlir tree build

## MKLDNN

- https://github.com/intel/mkl-dnn/releases
- we use prebuilt release rather than building it by ourselves

  `$ wget https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz`\
  `$ tar zxf mkldnn_lnx_1.0.2_cpu_gomp.tgz`\
  `$ cp mkldnn_lnx_1.0.2_cpu_gomp ~/work_cvitek/install_mkldnn -a`

## cnpy

- https://github.com/rogersce/cnpy  ==>  git@xxx:../cnpy.git (checkout tpu branch)
- with some modifications for TensorFile usage
- mlir tree build

## caffe

- https://github.com/BVLC/caffe
- orignial version (could use caffe_int8 project as well)
- munually build for now
- build

```
$ cd third_party/caffe
$ mkdir build
$ cd build
$ cmake -G Ninja -DUSE_OPENCV=OFF -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_caffe ..
$ cmake --build . --target install
```
