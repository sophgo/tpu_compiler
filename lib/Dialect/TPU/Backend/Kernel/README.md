# backend

## overview

backend is a lib for TPU code generation. It calls bmkernel apis and provide a
higher level api for compiler. This is a temporary solution for reusing the
existing backend code, and should be replaced by a more formal lowering process.

## dependency

bmkernel

## build

assuming install to ../install_backend

```
$ cd backend
$ mkdir build
$ cd build
$ cmake -G Ninja -DCHIP=BM1880v2 -DCVIKERNEL_PATH=../install_bmkernel -DCMAKE_INSTALL_PREFIX=../../install_backend ..

Build
$ cmake --build .
$ cmake --build . -- -v

Install
$ cmake --build . --target install
$ cmake --build . --target install -- -v

Test
$ cmake --build . --target test -- -v

Uninstall
$ xargs rm < install_manifest.txt
```

## output

```

```

## TODO

* refactor with more formal api
* support different ASICs
* has to compile using tree build for now, as it is using llvm Debug/raw_ostream
