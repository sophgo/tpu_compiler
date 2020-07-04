#!/bin/bash
set -e

#
# Run after running run_regression.sh -e
#

# "/home/ftp/mlir/daily_build"
RELEASE_DIR=$1
REGRESSION_DIR=$2
dest_dir=$RELEASE_DIR/$(date '+%Y-%m-%d')
rm -rf $dest_dir
mkdir -p $dest_dir

# check dirs
if [ ! -e $BUILD_PATH ]; then
  echo "BUILD_PATH=$BUILD_PATH not exist"
  exit 1
fi
if [ ! -e $INSTALL_PATH ]; then
  echo "INSTALL_PATH=$INSTALL_PATH not exist"
  exit 1
fi
if [ ! -e $REGRESSION_DIR ]; then
  echo "REGRESSION_DIR=$REGRESSION_DIR not exist"
  exit 1
fi
if [ ! -e $REGRESSION_DIR/cvimodel_regression ]; then
  echo "$REGRESSION_DIR/cvimodel_regression not exist"
  exit 1
fi

###########################################################
# pack cvitek_mlir
###########################################################
os_ver=$( lsb_release -sr )
pushd $INSTALL_PATH/..
if [ ! -e ./cvitek_mlir ]; then
  echo "./cvitek_mlir not exist"
  exit 1
fi
tar zcvf $dest_dir/cvitek_mlir_ubuntu-${os_ver}.tar.gz cvitek_mlir
popd

# exit if os_ver is not 18.04
if [ "$os_ver" != "18.04" ]; then
  exit 0
fi

###########################################################
# pack cvimodel_samples and samples
###########################################################
pushd $BUILD_PATH
tar zcvf $dest_dir/cvimodel_samples.tar.gz cvimodel_samples
# tar zcvf $dest_dir/cvimodel_release.tar.gz cvimodel_release
popd
# tpu_samples
cp -a $INSTALL_PATH/samples -a ./cvitek_tpu_samples
tar zcvf $dest_dir/cvitek_tpu_samples.tar.gz cvitek_tpu_samples
rm -rf cvitek_tpu_samples

###########################################################
# pack regresion cvimodels
###########################################################
pushd $REGRESSION_DIR
# generate int8 input data (for bs1 only)
pushd cvimodel_regression
generate_int8_data.sh
popd
# seperate bs1/bs4
mkdir -p cvimodel_regression_bs1
mkdir -p cvimodel_regression_bs4
mv cvimodel_regression/*bs4.cvimodel cvimodel_regression_bs4/
mv cvimodel_regression/*bs4_in_fp32.npz cvimodel_regression_bs4/
mv cvimodel_regression/*bs4_out_all.npz cvimodel_regression_bs4/
mv cvimodel_regression/* cvimodel_regression_bs1/
# tar
tar zcvf $dest_dir/cvimodel_regression_bs1.tar.gz cvimodel_regression_bs1
tar zcvf $dest_dir/cvimodel_regression_bs4.tar.gz cvimodel_regression_bs4
popd
