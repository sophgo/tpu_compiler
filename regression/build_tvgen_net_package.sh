#! /usr/bin/env bash
#set -e

command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 || { echo >&2 "I require aarch64-elf-gcc but it's not installed.  Aborting."; exit 1; }
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MLIR_SOURCE_DIR=$PWD
BUILD_DIR=$MLIR_SOURCE_DIR/externals/cmodel/tests/regression/build
RELEASE_DIR=$PWD/tvgen_package

testcase+="resnet50  "

batch="1 "
PROJECT="bm1822"
CHIP_NAME="cv1822"
OPT_TYPE="int8_multiplier"

RED='\033[0;31m'
NC='\033[0m'
GREEN='\033[0;32m'

bm_important_msg() {
  echo -e "${GREEN}$1${NC}"
}

bm_error_msg() {
  echo -e "${RED}$1${NC}"
}

build_fw() {
  export CROSS_COMPILE=aarch64-linux-gnu-
  cp $NET_DIR/engine_*.log $MLIR_SOURCE_DIR/externals/cmodel/src/$PROJECT/tv_gen_fw/tests/
  TMP=$PWD
  pushd $MLIR_SOURCE_DIR/externals/cmodel/src/$PROJECT/tv_gen_fw/tests/
  ./build_tv_gen_fw.sh
  cp a53_build/rom.hex $TMP
  popd
}

# Gen tvgen
if [ ! -d $RELEASE_DIR ]; then
  mkdir $RELEASE_DIR
fi

pushd $RELEASE_DIR
for d in ${testcase}
do
  for b in ${batch}
  do
  CVIMODEL_REL_PATH=$MLIR_SOURCE_DIR/regression_out/${d}_bs${b}
  echo "CVIMODEL_REL_PATH=" $CVIMODEL_REL_PATH
  if [ ! -d ${d}_bs${b} ]; then
    mkdir ${d}_bs${b}
  fi
  pushd ${d}_bs${b}
  NET_DIR=$PWD
  rm -r -f *
  bm_important_msg "Generate tvgen for ${PROJECT} ${d}_bs${b} TYPE: $OPT_TYPE"

  model_runner \
      --input $CVIMODEL_REL_PATH\/${d}_in_fp32.npz \
      --model $CVIMODEL_REL_PATH\/${d}_$OPT_TYPE.cvimodel \
      --batch-num $b \
      --set-chip ${CHIP_NAME} \
      --output ${d}_cmdbuf_out.npz

  build_fw
  popd
  done
done
popd

