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
- Use master branch
  * Merge caffe.proto from intel branch
  * Add layers: upsample, ssd related.
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
  * Use mkldnn in caffe/external/mkldnn/install
    * assign MKLDNNROOT
- munually build
- build

```
$ cd third_party/caffe
$ mkdir build
$ cd build
Assuming install to $TPU_BASE/install_caffe
$ export CAFFE_PATH=$TPU_BASE/install_caffe
```

To build master branch


```bash
cmake -G Ninja -DCPU_ONLY=ON -DUSE_OPENCV=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
cmake --build . --target install
```

To build intel branch

Ubuntu 18.04 (use prebuilt mkldnn)
```
$ MKLDNNROOT=$MLIR_SRC_PATH/third_party/caffe/external/mkldnn/install_ubuntu1804 \
    cmake -DUSE_OPENCV=OFF -DDISABLE_MKLDNN_DOWNLOAD=1 \
    -DUSE_OPENMP=OFF -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF -DUSE_MLSL=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
$ cmake --build . --target install
```

Ubuntu 16.04 (use prebuilt mkldnn)
```
$ MKLDNNROOT=$MLIR_SRC_PATH/third_party/caffe/external/mkldnn/install_ubuntu1604 \
    cmake -DUSE_OPENCV=OFF -DDISABLE_MKLDNN_DOWNLOAD=1 \
    -DUSE_OPENMP=OFF -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF -DUSE_MLSL=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
$ cmake --build . --target install
```

For other distributions (build and download on the fly)
```
$ cmake -DUSE_OPENCV=OFF \
    -DUSE_OPENMP=OFF -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF -DUSE_MLSL=OFF \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_PATH ..
$ cmake --build . --target install
May need to fix followings during build
$ cd $MLIR_SRC_PATH/third_party/caffe/external/mkldnn/install
$ ln -s lib lib64
```

Also need to copy external/mkl/* to caffe install dir
```
$ cd $MLIR_SRC_PATH/third_party/caffe/external/mkl $CAFFE_PATH -a
```
