# docker for cvitek mlir

## Prerequisits

## Create docker

```sh
$ cp Dockerfile_ubuntu-18.04 Dockerfile
$ docker build -t cvitek_dev:1.0-ubuntu-18.04 .

$ cp Dockerfile_ubuntu-16.04 Dockerfile
$ docker build -t cvitek_dev:1.0-ubuntu-16.04 .
```

## Run docker as user of cvitek_mlir

```sh
$ docker run -itd \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek cvitek_dev:1.0-ubuntu-18.04

$ docker run -itd \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek cvitek_dev:1.0-ubuntu-16.04

$ docker exec -it cvitek bash

# cd work
# # export MODEL_PATH=$PWD/models
# source cvitek_mlir/cvitek_envs.sh
# ./cvitek_mlir/regression/generic/regression_generic.sh resnet50
```

## Run docker for build mlir

```sh

$ cd ~/work_cvitek
$ docker run -itd [--cpus="1.5"] \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek_dev_18.04 cvitek_dev:1.0-ubuntu-18.04
$ docker exec -it cvitek_dev_18.04 bash

$ docker run -itd [--cpus="1.5"] \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek_dev_16.04 cvitek_dev:1.0-ubuntu-16.04
$ docker exec -it cvitek_dev_16.04 bash

# cd work
# # export MODEL_PATH=$PWD/models
# source llvm-project/llvm/projects/mlir/envsetup.sh
# build.sh RELEASE
```
