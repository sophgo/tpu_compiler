#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/docker.env

CMD="
docker build \
    -t $REPO/$IMAGE:$TAG_BASE \
    -f $DIR/Dockerfile_ubuntu-${BASE_IMAGE_VERSION} \
    .
"

echo $CMD
eval $CMD
