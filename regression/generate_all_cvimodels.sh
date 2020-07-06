#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

generic_net_list=()

if [ -z $model_list_file ]; then
  model_list_file=$DIR/generic/model_list.txt
fi
while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext
do
  [[ $net =~ ^#.* ]] && continue
  # echo "net='$net' bs1='$bs1' bs4='$bs4' acc='$acc' bs1_ext='$bs1_ext' bs4_ext='$bs4_ext' acc_ext='$acc_ext'"
  if [ "$bs1" = "Y" ]; then
    # echo "bs1 add $net"
    generic_net_list+=($net)
  fi
  if [ "$bs1_ext" = "Y" ]; then
    # echo "bs1_ext add $net"
    generic_net_list+=($net)
  fi
done < ${model_list_file}

extra_net_param()
{
  NET=$1

  if [ $NET = "retinaface_mnet25_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
  export MODEL_CHANNEL_ORDER="rgb"
  export SWAP_CHANNEL=2,1,0
  export IMAGE_RESIZE_DIMS=320,320
  export NET_INPUT_DIMS=320,320
  export RAW_SCALE=255.0
  export MEAN=0,0,0
  export INPUT_SCALE=1
  export DO_PREPROCESS=1
  if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table_preprocess
  else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
  fi
  fi

  if [ $NET = "retinaface_mnet25_600_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_600_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
  export MODEL_CHANNEL_ORDER="rgb"
  export SWAP_CHANNEL=2,1,0
  export IMAGE_RESIZE_DIMS=600,600
  export NET_INPUT_DIMS=600,600
  export RAW_SCALE=255.0
  export MEAN=0,0,0
  export INPUT_SCALE=1
  export DO_PREPROCESS=1
  if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table_preprocess
  else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
  fi
  fi

  if [ $NET = "retinaface_res50_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_res50_calibration_table
  export MODEL_CHANNEL_ORDER="rgb"
  export SWAP_CHANNEL=2,1,0
  export IMAGE_RESIZE_DIMS=600,600
  export NET_INPUT_DIMS=600,600
  export RAW_SCALE=255.0
  export MEAN=0,0,0
  export INPUT_SCALE=1
  export DO_PREPROCESS=1
  if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_res50_calibration_table_preprocess
  else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_res50_calibration_table
  fi
  fi

  if [ $NET = "yolo_v3_416_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  export MODEL_CHANNEL_ORDER="rgb"
  export SWAP_CHANNEL=2,1,0
  export IMAGE_RESIZE_DIMS=416,416
  export NET_INPUT_DIMS=416,416
  export RAW_SCALE=1.0
  export MEAN=0,0,0
  export INPUT_SCALE=1.0
  export DO_PREPROCESS=1
  if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
  else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  fi
  fi

  if [ $NET = "yolo_v3_320_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  export MODEL_CHANNEL_ORDER="rgb"
  export SWAP_CHANNEL=2,1,0
  export IMAGE_RESIZE_DIMS=320,320
  export NET_INPUT_DIMS=320,320
  export RAW_SCALE=1.0
  export MEAN=0,0,0
  export INPUT_SCALE=1.0
  export DO_PREPROCESS=1
  if [ $DO_PREPROCESS -eq 1 ]; then
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune_preprocess
  else
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  fi
  fi
}

extra_net_list=(
  "retinaface_mnet25_with_detection"
  "retinaface_mnet25_600_with_detection"
  "retinaface_res50_with_detection"
  "yolo_v3_416_with_detection"
  "yolo_v3_320_with_detection"
)

if [ ! -e cvimodel_release ]; then
  mkdir cvimodel_release
fi

err=0
pushd cvimodel_release
rm -rf working
mkdir working

# generic
for net in ${generic_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  source $DIR/generic/generic_models.sh
  echo "NET=$NET MODEL_TYPE=$MODEL_TYPE"
  if [ $MODEL_TYPE = "caffe" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_caffe_lg.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
      if [ $DO_PREPROCESS -eq 1 ]; then
        $DIR/convert_model_caffe_lg_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE} \
          ${NET}_preprocess.cvimodel
        mv ${NET}_preprocess.cvimodel ..
      fi
    else
      $DIR/convert_model_caffe_df.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
      if [ $DO_PREPROCESS -eq 1 ]; then
        $DIR/convert_model_caffe_df_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE} \
          ${NET}_preprocess.cvimodel
        mv ${NET}_preprocess.cvimodel ..
      fi
    fi
  elif [ $MODEL_TYPE = "onnx" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_onnx_lg.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    else
      $DIR/convert_model_onnx_df.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    fi
    mv ${NET}.cvimodel ..
  elif [ $MODEL_TYPE = "tensorflow" ]; then
    echo "Not supported MODEL_TYPE=$MODEL_TYPE"
  else
    echo "Invalid MODEL_TYPE=$MODEL_TYPE"
    err=1
  fi
  rm -f ./*
  popd
  if [ "$err" -ne 0 ]; then
    rm -rf working
    popd
    exit 1
  fi
done

# extra
for net in ${extra_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  extra_net_param $NET
  if [ $MODEL_TYPE = "caffe" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_caffe_lg.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
      if [ $DO_PREPROCESS -eq 1 ]; then
        $DIR/convert_model_caffe_lg_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE} \
          ${NET}_preprocess.cvimodel
        mv ${NET}_preprocess.cvimodel ..
      fi
    else
      $DIR/convert_model_caffe_df.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
      if [ $DO_PREPROCESS -eq 1 ]; then
        $DIR/convert_model_caffe_df_preprocess.sh \
          ${MODEL_DEF} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${RAW_SCALE} \
          ${MEAN} \
          ${INPUT_SCALE} \
          ${SWAP_CHANNEL} \
          ${CALI_TABLE} \
          ${NET}_preprocess.cvimodel
        mv ${NET}_preprocess.cvimodel ..
      fi
    fi
  elif [ $MODEL_TYPE = "onnx" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_onnx_lg.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    else
      $DIR/convert_model_onnx_df.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    fi
    mv ${NET}.cvimodel ..
  elif [ $MODEL_TYPE = "tensorflow" ]; then
    echo "Not supported MODEL_TYPE=$MODEL_TYPE"
  else
    echo "Invalid MODEL_TYPE=$MODEL_TYPE"
    err=1
  fi
  rm -f ./*
  popd
  if [ "$err" -ne 0 ]; then
    rm -rf working
    popd
    exit 1
  fi
done

rm -rf working
popd
