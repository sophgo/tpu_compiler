#!/bin/bash

# This script accepts two command-line arguments, which specifies preprocessing type and model to download.

# For example, to download ckptsaug efficientnet-b0, run:
#   ./download.sh ckptsaug efficientnet-b0
# And to download advprop efficientnet-b3, run:
#   ./download.sh advprop efficientnet-b3

TYPE=$1
MODEL=$2
mkdir -p ${TYPE}
pushd ${TYPE}
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/${TYPE}/${MODEL}.tar.gz
tar xvf ${MODEL}.tar.gz
rm ${MODEL}.tar.gz
popd
