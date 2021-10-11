#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

UBUNTU=18.04
VERSION=1.7

BASE_REPO=cvitek/cvitek_dev:${VERSION}-ubuntu-${UBUNTU}
TARGET_REPO=cvitek_mlir-${UBUNTU}-for-jenkins:${VERSION}

CMD="
docker build \
    --build-arg BASE_REPO=${BASE_REPO}
    -t ${TARGET_REPO} \
    -f $DIR/Dockerfile \
    .
"

echo $CMD
eval $CMD
