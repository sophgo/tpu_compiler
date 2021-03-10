#!/bin/bash
set -xe

#
# Run after running run_regression.sh -e
#

# "/home/ftp/mlir/daily_build"
RELEASE_PATH=$1
WORKING_PATH=$2

echo "RELEASE_PATH: $RELEASE_PATH"
echo "WORKING_PATH: $WORKING_PATH"

os_ver=$( lsb_release -sr )
dest_dir=$RELEASE_PATH/$(date '+%Y-%m-%d')-${os_ver}
rm -rf $dest_dir
mkdir -p $dest_dir

# check dirs
if [ ! -e $WORKING_PATH ]; then
  echo "WORKING_PATH=$WORKING_PATH not exist"
  exit 1
fi

# cvimodels used by sample codes
sample_models_list=(
  mobilenet_v2.cvimodel
  mobilenet_v2_fused_preprocess.cvimodel
  yolo_v3_416_with_detection.cvimodel
  yolo_v3_416_fused_preprocess_with_detection.cvimodel
  alphapose.cvimodel
  alphapose_fused_preprocess.cvimodel
  retinaface_mnet25_600_with_detection.cvimodel
  retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel
  arcface_res50.cvimodel
  arcface_res50_fused_preprocess.cvimodel
)

function pack_compiler() {
  if [ ! -e cvitek_mlir ]; then
    echo "./cvitek_mlir not exist"
    exit 1
  fi
  # tpu_samples
  cp -a cvitek_mlir/tpuc/samples -a cvitek_tpu_samples
  tar zcvf $dest_dir/cvitek_tpu_samples.tar.gz cvitek_tpu_samples
  rm -rf cvitek_tpu_samples

  # cvitek toolchain
  tar zcvf $dest_dir/cvitek_mlir_ubuntu-${os_ver}.tar.gz cvitek_mlir
}

function pack_cvimodels() {
  chip_name=$1

  if [ ! -e cvimodel_release/$chip_name ]; then
    echo "./cvitek_release/$chip_name not exist"
    exit 1
  fi
  if [ ! -e regression_out/$chip_name/cvimodel_regression ]; then
    echo "./regression_out/$chip_name/cvimodel_regression not exist"
    exit 1
  fi

  mkdir -p cvimodel_samples_$chip_name
  for sample_model in ${sample_models_list[@]}
  do
   cp cvimodel_release/$chip_name/${sample_model} cvimodel_samples_$chip_name/
  done
  # copy extra yuv420 cvimodel to cvimodel_samples
  cp regression_out/$chip_name/cvimodel_regression/cvimodel_regression_fused_preprocess/mobilenet_v2_int8_yuv420.cvimodel \
    cvimodel_samples_$chip_name/
  tar zcvf $dest_dir/cvimodel_samples_${chip_name}.tar.gz cvimodel_samples_$chip_name
  rm -rf cvimodel_samples_$chip_name

  # pack regresion cvimodels
  pushd regression_out/$chip_name/cvimodel_regression
  tar zcvf $dest_dir/cvimodel_regression_bs1_${chip_name}.tar.gz cvimodel_regression_bs1
  tar zcvf $dest_dir/cvimodel_regression_bs4_${chip_name}.tar.gz cvimodel_regression_bs4
  tar zcvf $dest_dir/cvimodel_regression_bf16_${chip_name}.tar.gz cvimodel_regression_bf16
  tar zcvf $dest_dir/cvimodel_regression_fused_preprocess_${chip_name}.tar.gz cvimodel_regression_fused_preprocess
  popd
}

pushd $WORKING_PATH
pack_compiler
pack_cvimodels cv183x

# pack cv182x models
if [ ! -e cvimodel_release/cv182x ]; then
  exit 0
fi
pack_cvimodels cv182x

popd
