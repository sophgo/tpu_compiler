#!/bin/bash
set -e

# "/home/ftp/mlir/daily_build"
REL_DIR=$1
dest_dir=$REL_DIR/$(date '+%Y-%m-%d')
rm -rf $dest_dir
mkdir -p $dest_dir

# cvitek_mlir
if [ ! -e $INSTALL_PATH ]; then
  echo "INSTALL_PATH=$INSTALL_PATH not exist"
  exit 1
fi
pushd $INSTALL_PATH/..
if [ ! -e ./cvitek_mlir ]; then
  echo "./cvitek_mlir not exist"
  exit 1
fi
os_ver=$( lsb_release -sr )
tar zcf cvitek_mlir_ubuntu-${os_ver}.tar.gz cvitek_mlir
mv cvitek_mlir_ubuntu-${os_ver}.tar.gz $dest_dir/
popd

# "./regression_out"
REGRESSION_DIR=$2
if [ ! -e $REGRESSION_DIR ]; then
  echo "REGRESSION_DIR=$REGRESSION_DIR not exist"
  exit 1
fi
if [ ! -e $REGRESSION_DIR/cvimodel_regression ]; then
  echo "$REGRESSION_DIR/cvimodel_regression not exist"
  exit 1
fi
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
tar zcf cvimodel_regression_bs1.tar.gz cvimodel_regression_bs1
tar zcf cvimodel_regression_bs4.tar.gz cvimodel_regression_bs4
popd
# release
mv $REGRESSION_DIR/cvimodel_regression_bs1.tar.gz $dest_dir/
mv $REGRESSION_DIR/cvimodel_regression_bs4.tar.gz $dest_dir/
