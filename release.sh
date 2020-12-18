#!/bin/bash
set -e

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

pushd $WORKING_PATH
###########################################################
# pack cvitek_mlir
###########################################################
if [ ! -e cvitek_mlir ]; then
  echo "./cvitek_mlir not exist"
  exit 1
fi
if [ ! -e cvimodel_release ]; then
  echo "./cvitek_release not exist"
  exit 1
fi
if [ ! -e regression_out/cvimodel_regression ]; then
  echo "./regression_out/cvimodel_regression not exist"
  exit 1
fi


# tpu_samples
cp -a cvitek_mlir/tpuc/samples -a cvitek_tpu_samples
tar zcvf $dest_dir/cvitek_tpu_samples.tar.gz cvitek_tpu_samples
rm -rf cvitek_tpu_samples

# cvitek toolchain
tar zcvf $dest_dir/cvitek_mlir_ubuntu-${os_ver}.tar.gz cvitek_mlir

###########################################################
# pack cvimodel_samples and samples
###########################################################
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

mkdir -p cvimodel_samples
for sample_model in ${sample_models_list[@]}
do
  cp cvimodel_release/${sample_model} cvimodel_samples/
done

tar zcvf $dest_dir/cvimodel_samples.tar.gz cvimodel_samples
rm -rf cvimodel_samples

###########################################################
# pack regresion cvimodels
###########################################################
# seperate bs1/bs4
pushd regression_out

mkdir -p cvimodel_regression_bs1
mkdir -p cvimodel_regression_bs4
mv cvimodel_regression/*bs4.cvimodel cvimodel_regression_bs4/
mv cvimodel_regression/*bs4_in_fp32.npz cvimodel_regression_bs4/
mv cvimodel_regression/*bs4_out_all.npz cvimodel_regression_bs4/
mv cvimodel_regression/* cvimodel_regression_bs1/
# tar
tar zcvf $dest_dir/cvimodel_regression_bs1.tar.gz cvimodel_regression_bs1
tar zcvf $dest_dir/cvimodel_regression_bs4.tar.gz cvimodel_regression_bs4
rm -rf cvimodel_regression_bs1
rm -rf cvimodel_regression_bs4

popd
popd
