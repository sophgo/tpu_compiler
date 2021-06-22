#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export WORKING_PATH=${WORKING_PATH:-$DIR/regression_out}
WORKDIR=$WORKING_PATH/calib
if [ ! -e $WORKDIR ]; then
  mkdir -p $WORKDIR
fi

NET=mobilenet_v2
export NET=$NET
source $DIR/generic/generic_models.sh

set -x
pushd $WORKDIR
model_transform.py \
  --model_type ${MODEL_TYPE} \
  --model_name ${NET} \
  --model_def ${MODEL_DEF} \
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
  --graph \
  --mlir ${NET}_fp32.mlir

run_calibration.py \
  ${NET}_fp32.mlir \
  --dataset $DATASET_PATH/imagenet/img_val_extracted \
  --input_num 100 \
  -o ${NET}_calib.txt \

run_mix_precision.py \
    ${NET}_fp32.mlir \
    --dataset ${DATASET_PATH}/imagenet/img_val_extracted \
    --input_num=5 \
    --calibration_table ${NET}_calib.txt \
    --max_bf16_layers=6 \
    -o ${NET}_mix_table.txt

run_tune.py \
    ${NET}_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --input_num=5 \
    --calibration_table ${NET}_calib.txt \
    --mix_precision_table ${NET}_mix_table.txt \
    --tune_iteration=3 \
    --strategy greedy \
    --evaluation euclid \
    --speedup \
    -o ${NET}_tuned_calib_0.txt

run_tune.py \
    ${NET}_fp32.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --input_num=5 \
    --calibration_table ${NET}_calib.txt \
    --mix_precision_table ${NET}_mix_table.txt \
    --tune_iteration=3 \
    --strategy overall \
    --evaluation cosine \
    --speedup \
    -o ${NET}_tuned_calib_1.txt

popd
echo $0 DONE
