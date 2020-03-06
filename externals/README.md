# External projects

CViTek developed and maintained projects that are used by mlir-tpu.

Right now, all of them rely on manually build, except backend.

Backend project rely on llvm (Debug, format, etc.), support tree build only.

Should update all projects to support tree build later.

## python_tools

- git@xxx:../python_tools.git
- dependencies
  * none

## cvikernel

- git@xxx:../cvikernel.git
- TPU Dialect link to this library for code gen
- dependencies
  * none

```
$ cd externals/cvikernel
$ mkdir build
$ cd build
$ cmake -G Ninja -DCHIP=BM1880v2 -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_cvikernel ..
$ cmake --build . --target install
```
Read cvikernel/README.md for more details.

## backend

- git@xxx:../backend.git
- TPU Dialect link to this library for code gen
- dependencies
  * cvikernel

Only support tree build for now, as it is using some llvm debug facilities.

## cmodel

- git@xxx:../cvikernel.git
- for testing in cmodel mode, not linking in mlir-tpu
- dependencies
  * cvikernel

```
$ cd externals/cmodel
$ mkdir build
$ cd build
$ cmake -G Ninja -DCHIP=BM1880v2 -DCVIKERNEL_PATH=~/work_cvitek/install_cvikernel -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_cmodel ..
$ cmake --build . --target install
```
Read cmodel/README.md for more details.

## cvibuilder

- git@xxx:../cvibuilder.git
- for testing, not linking in mlir-tpu
- dependencies
  * none

Read cvibuilder/README.md for more details.

## support

TODO: to remove, and merge into runtime

- git@xxx:../support.git
- for testing, not linking in mlir-tpu
- dependencies
  * none

```
$ cd externals/support
$ mkdir build
$ cd build
$ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_support ..
$ cmake --build . --target install
```

## runtime

- git@xxx:../runtime.git
- for testing, not linking in mlir-tpu
- dependencies
  * cvikernel
  * cvibuilder
  * cmodel
  * support

```
$ cd runtime
$ mkdir build
$ cd build
$ cmake -G Ninja -DCHIP=BM1880v2 -DRUNTIME=CMODEL -DSUPPORT_PATH=~/work_cvitek/install_support -DCVIMODEL_PATH=~/work_cvitek/install_cvibuilder -DCVIKERNEL_PATH=~/work_cvitek/install_cvikernel -DCMODEL_PATH=~/work_cvitek/install_cmodel -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_runtime ..
$ cmake --build . --target install
```
Read runtime/README.md for more details.
