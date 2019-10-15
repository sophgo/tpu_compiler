# External projects

CViTek developed and maintained projects that are used by mlir-tpu.

Right now, all of them rely on manually build.

Should consider change to tree build later.

## bmkernel

- git@xxx:../bmkernel.git
- TPU Dialect link to this library for code gen

```
$ cd externals/bmkernel
$ mkdir build
$ cd build
$ cmake -G Ninja -DCHIP=BM1880v2 -DCMAKE_INSTALL_PREFIX=~/work_cvitek/install_bmkernel ..
$ cmake --build . --target install
```
Read bmkernel/README.md for more details.

## cmodel

- git@xxx:../bmkernel.git
- for testing in cmodel mode, not linking in mlir-tpu

## runtime

- git@xxx:../runtime.git
- for testing, not linking in mlir-tpu

## builder

- git@xxx:../builder.git
- for testing, not linking in mlir-tpu
