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
- Use tpu branch (based on intel branch, for SSD/YOLO etc.)
  * Intel branch can't use Ninja
  * Disable followings compilation options
    * USE_OPENMP
    * USE_MKLDNN_AS_DEFAULT_ENGINE
    * USE_MLSL
  * Disable CompileNet(), which includes
    * RemoveBNScale<Dtype>
    * CompilationRuleRemoveScale
    * CompilationRuleConvReluFusion
    * CompilationRuleFuseBnRelu
    * CompilationRuleBNInplace
    * CompilationRuleSparse
    * CompilationRuleConvSumFusion
    * CompilationRuleFuseFCRelu
- munually build
- build

```
$ cd third_party/caffe
$ mkdir build
$ cd build

$ MKLDNNROOT=./external/mkldnn/install \
    cmake -DUSE_OPENCV=OFF -DDISABLE_MKLDNN_DOWNLOAD=1 \
    -DUSE_OPENMP=OFF -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF -DUSE_MLSL=OFF \
    -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_caffe ..
$ cmake --build . --target install
```

Also need to copy external/mkl/* to caffe install dir
```
$ cd third_party/caffe/external/mkl ~/work_cvitek/install_caffe -a
```
