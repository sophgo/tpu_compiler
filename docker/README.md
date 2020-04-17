# docker for cvitek mlir

## General Commands

```sh
$ sudo apt install docker.io
$ systemctl start docker
$ systemctl enable docker

$ sudo groupadd docker
$ sudo usermod -aG docker lwang
$ newgrp docker

$ docker ps
$ docker stop cvitek
$ docker rm cvitek

$ docker images
$ docker image rm cvitek/cvitek_dev:1.1-ubuntu-18.04
$ docker image prune
```

## Create docker

Copy host-tools to current dir first

```sh
$ cp Dockerfile_ubuntu-18.04 Dockerfile
$ docker build -t cvitek/cvitek_dev:1.1-ubuntu-18.04 .

$ cp Dockerfile_ubuntu-16.04 Dockerfile
$ docker build -t cvitek/cvitek_dev:1.1-ubuntu-16.04 .
```

## Run docker as user of cvitek_mlir

```sh
$ docker run -itd \
    -v $PWD:/work \
    -v /data/models:/work/models \
    -v /data/dataset:/work/dataset \
    --name cvitek cvitek/cvitek_dev:1.1-ubuntu-18.04

$ docker run -itd \
    -v $PWD:/work \
    -v /data/models:/work/models \
    -v /data/dataset:/work/dataset \
    --name cvitek cvitek/cvitek_dev:1.1-ubuntu-16.04

$ docker exec -it cvitek bash

# cd work
# # export MODEL_PATH=$PWD/models
# # export DATASET_PATH=$PWD/dataset
# source cvitek_mlir/cvitek_envs.sh
# ./cvitek_mlir/regression/generic/regression_generic.sh resnet50
```

## Run docker for build mlir

```sh
$ docker run -itd [--cpus="1.5"] \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek_dev_18.04 cvitek_dev:1.1-ubuntu-18.04
$ docker exec -it cvitek_dev_18.04 bash

$ docker run -itd [--cpus="1.5"] \
    -v $PWD:/work \
    -v /data/models:/work/models \
    --name cvitek_dev_16.04 cvitek_dev:1.1-ubuntu-16.04
$ docker exec -it cvitek_dev_16.04 bash

# cd work
# # export MODEL_PATH=$PWD/models
# # export DATASET_PATH=$PWD/dataset
# source llvm-project/llvm/projects/mlir/envsetup.sh
# build.sh RELEASE
```

