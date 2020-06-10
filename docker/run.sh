#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/docker.env

docker run --rm -it \
    --net=host \
    -v $PWD:/work \
    -v /data/models:/work/models \
    -v /data/dataset:/work/dataset \
    --privileged \
    --name=cvitek \
    $REPO/$IMAGE:$TAG_BASE bash
