#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -xe


# "/home/ftp/mlir/daily_build"
RELEASE_PATH=$1
WORKING_PATH=$2

echo "RELEASE_PATH: $RELEASE_PATH"
echo "WORKING_PATH: $WORKING_PATH"

os_ver=$( lsb_release -sr )
dest_dir=$RELEASE_PATH
rm -rf $dest_dir
mkdir -p $dest_dir

# check dirs
if [ ! -e $WORKING_PATH ]; then
  echo "WORKING_PATH=$WORKING_PATH not exist"
  exit 1
fi


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

function gencvimodel_for_sample() {
  local chip=$1
  local NET=$2
  local cvimodel=$3
  local fuse_preprocess=$4
  mkdir -p tmp
  pushd tmp
  source $SCRIPT_DIR/regression/generic/generic_models.sh
  mdef=${MODEL_DEF}
  if [[ $DO_FUSED_POSTPROCESS = "1" ]]; then
    mdef=${MODEL_DEF_FUSED_POSTPROCESS}
  fi
  model_transform.py \
    --model_type ${MODEL_TYPE} \
    --model_name ${NET} \
    --model_def ${mdef} \
    --model_data ${MODEL_DAT} \
    --image ${IMAGE_PATH} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --net_input_dims ${NET_INPUT_DIMS} \
    --raw_scale ${RAW_SCALE} \
    --mean ${MEAN} \
    --std ${STD} \
    --input_scale ${INPUT_SCALE} \
    --model_channel_order ${MODEL_CHANNEL_ORDER} \
    --gray ${BGRAY} \
    --batch_size $BATCH_SIZE \
    --tolerance ${TOLERANCE_FP32} \
    --excepts ${EXCEPTS} \
    --mlir ${NET}_fp32.mlir

  if [ $fuse_preprocess -eq 1 ]; then
    local pixel_format=$5
    local aligned_input=$6
    model_deploy.py \
      --model_name ${NET} \
      --mlir ${NET}_fp32.mlir \
      --calibration_table ${CALI_TABLE} \
      --mix_precision_table ${MIX_PRECISION_TABLE} \
      --chip ${chip} \
      --image ${IMAGE_PATH} \
      --tolerance ${TOLERANCE_INT8_MULTIPLER} \
      --excepts ${EXCEPTS} \
      --fuse_preprocess \
      --pixel_format $pixel_format \
      --aligned_input $aligned_input \
      --correctness 0.99,0.99,0.99 \
      --cvimodel $cvimodel
  else
    model_deploy.py \
      --model_name ${NET} \
      --mlir ${NET}_fp32.mlir \
      --calibration_table ${CALI_TABLE} \
      --mix_precision_table ${MIX_PRECISION_TABLE} \
      --chip ${chip} \
      --image ${IMAGE_PATH} \
      --tolerance ${TOLERANCE_INT8_MULTIPLER} \
      --excepts ${EXCEPTS} \
      --correctness 0.99,0.99,0.99 \
      --cvimodel $cvimodel
  fi
  popd
  rm -rf tmp
}

function gen_merged_model() {
  local name=$1
  local chip=$2
  local batch=$3
  local step=$4

  model_transform.py \
    --model_type ${MODEL_TYPE} \
    --model_name ${NET} \
    --model_def ${mdef} \
    --model_data ${MODEL_DAT} \
    --image ${IMAGE_PATH} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --keep_aspect_ratio ${RESIZE_KEEP_ASPECT_RATIO} \
    --net_input_dims ${NET_INPUT_DIMS} \
    --raw_scale ${RAW_SCALE} \
    --mean ${MEAN} \
    --std ${STD} \
    --input_scale ${INPUT_SCALE} \
    --model_channel_order ${MODEL_CHANNEL_ORDER} \
    --gray ${BGRAY} \
    --batch_size $batch \
    --tolerance ${TOLERANCE_FP32} \
    --excepts ${EXCEPTS} \
    --mlir ${NET}_fp32.mlir

  cmd="model_deploy.py\
      --model_name ${NET}\
      --mlir ${NET}_fp32.mlir\
      --calibration_table ${CALI_TABLE}\
      --chip ${chip} \
      --image ${IMAGE_PATH}\
      --tolerance ${TOLERANCE_INT8_MULTIPLER}\
      --correctness 0.99,0.99,0.99\
      --excepts ${EXCEPTS}\
      --cvimodel $name"
  if [ $step -eq 1 ]; then
    cmd=${cmd}" --compress_weight false"
  elif [ $step -eq 2 ]; then
    cmd=${cmd}" --compress_weight false"
    cmd=${cmd}" --merge_weight"
  fi
  eval $cmd
}

function gen_merged_cvimodel_for_sample() {
  local chip=$1
  local NET=$2
  local cvimodel=$3

  mkdir -p tmp
  pushd tmp
  source $SCRIPT_DIR/regression/generic/generic_models.sh
  mdef=${MODEL_DEF}
  if [[ $DO_FUSED_POSTPROCESS = "1" ]]; then
    mdef=${MODEL_DEF_FUSED_POSTPROCESS}
  fi
  gen_merge_model tmp_model_bs1.cvimodel ${chip} 1 1
  gen_merge_model tmp_model_bs4.cvimodel ${chip} 4 2

  cvimodel_tool \
  -a merge \
  -i tmp_model_bs1.cvimodel \
     tmp_model_bs4.cvimodel \
  -o ${cvimodel}
  popd
  rm -rf tmp
}

function pack_sampel_cvimodels() {
  local chip=$1

  local dst=$PWD/cvimodel_samples
  mkdir -p $dst
  gencvimodel_for_sample $chip mobilenet_v2 \
      $dst/mobilenet_v2.cvimodel 0
  gencvimodel_for_sample $chip mobilenet_v2 \
      $dst/mobilenet_v2_fused_preprocess.cvimodel \
      1 BGR_PACKED 0
  gencvimodel_for_sample $chip mobilenet_v2 \
      $dst/mobilenet_v2_int8_yuv420.cvimodel \
      1 YUV420_PLANAR 1
  gencvimodel_for_sample $chip yolo_v3_416 \
      $dst/yolo_v3_416_with_detection.cvimodel 0
  gencvimodel_for_sample $chip yolo_v3_416 \
      $dst/yolo_v3_416_fused_preprocess_with_detection.cvimodel \
      1 BGR_PACKED 0
  gencvimodel_for_sample $chip alphapose \
      $dst/alphapose.cvimodel 0
  gencvimodel_for_sample $chip alphapose \
      $dst/alphapose_fused_preprocess.cvimodel \
      1 BGR_PACKED 0
  gencvimodel_for_sample $chip retinaface_mnet25_600 \
      $dst/retinaface_mnet25_600_with_detection.cvimodel 0
  gencvimodel_for_sample $chip retinaface_mnet25_600 \
      $dst/retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel \
      1 BGR_PACKED 0
  gencvimodel_for_sample $chip arcface_res50 \
      $dst/arcface_res50.cvimodel 0
  gencvimodel_for_sample $chip arcface_res50 \
      $dst/arcface_res50_fused_preprocess.cvimodel \
      1 BGR_PACKED 0

  # gen merged cvimodel
  gen_merged_cvimodel_for_sample $chip mobilenet_v2 \
      $dst/mobilenet_v2_bs1_bs4.cvimodel

  tar zcvf $dest_dir/cvimodel_samples_${chip}.tar.gz cvimodel_samples
  rm -rf $dst
}

function pack_regression_cvimodels() {
  local chip=$1

  if [ ! -e regression_out/$chip/cvimodel_regression ]; then
    echo "./regression_out/$chip/cvimodel_regression not exist"
    exit 1
  fi

  # pack regresion cvimodels
  pushd regression_out/$chip/cvimodel_regression
  tar zcvf $dest_dir/cvimodel_regression_bs1_${chip}.tar.gz \
           cvimodel_regression_bs1
  tar zcvf $dest_dir/cvimodel_regression_bs4_${chip}.tar.gz \
           cvimodel_regression_bs4
  tar zcvf $dest_dir/cvimodel_regression_bf16_${chip}.tar.gz \
           cvimodel_regression_bf16_bs1
  tar zcvf $dest_dir/cvimodel_regression_bf16_${chip}.tar.gz \
           cvimodel_regression_bf16_bs4
  popd
}

pushd $WORKING_PATH
pack_compiler
pack_sampel_cvimodels cv183x
pack_regression_cvimodels cv183x

# pack cv182x models
if [ ! -e regression_out/cv182x ]; then
  exit 0
fi
pack_sampel_cvimodels cv182x
pack_regression_cvimodels cv182x

pushd $SCRIPT_DIR
echo `git describe --tags --dirt` > $dest_dir/version.txt
cp -rf doc $dest_dir/
popd
popd
