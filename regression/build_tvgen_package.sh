#! /usr/bin/env bash
set -e

command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 || { echo >&2 "I require aarch64-elf-gcc but it's not installed.  Aborting."; exit 1; }
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

UT_PATH=$BUILD_PATH/build_cviruntime/test
MLIR_SOURCE_DIR=$PWD
RELEASE_DIR=$PWD/tvgen_package
PROJECT=bm1822

#for example, test_1822_tensor_add
testcase+="test_1822_tensor_add "

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
  cp $GEN_DIR/engine_*.log $MLIR_SOURCE_DIR/externals/cmodel/src/$PROJECT/tv_gen_fw/tests/
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
  if [ ! -d $d ]; then
    mkdir $d
  fi
  pushd $d
  bm_important_msg "Generate tvgen for $UT_PATH/$d"
  rm -r -f *
  GEN_DIR=$PWD
  $UT_PATH/$d
  status=$?
  [ $status -eq 0 ] || { bm_error_msg "$UT_PATH/$d failed, 81"; exit -1; }
  build_fw
  popd
done
echo "Generate successful";

