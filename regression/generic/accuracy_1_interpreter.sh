#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval imagenet with pytorch dataloader"
  EVAL_FUNC=eval_classifier.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval imagenet with gluoncv dataloader"
  EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

if [ $DO_ACCURACY_FP32_INTERPRETER -eq 1 ]; then
  echo "Eval fp32 with interpreter"
  if [ "$EVAL_MODEL_TYPE" = "imagenet" ]; then
    $EVAL_FUNC \
        --mlir_file=${NET}.mlir \
        --dataset=$DATASET_PATH/imagenet/img_val_extracted \
        --label_file=$LABEL_FILE \
        --net_input_dims $NET_INPUT_DIMS \
        --image_resize_dims $IMAGE_RESIZE_DIMS \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --std $STD \
        --input_scale $INPUT_SCALE \
        --model_type mlir \
        --model_channel_order $MODEL_CHANNEL_ORDER \
        --count=$1

  elif [ "$EVAL_MODEL_TYPE" = "voc2012" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_VOC
    $_EVAL_FUNC \
      --mlir=${NET}.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --dataset=$DATASET_PATH/VOCdevkit

  else
    echo "Unknown EVAL_MODEL_TYPE $EVAL_MODEL_TYPE"
    exit 1
  fi
fi

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  echo "Eval int8_per_tensor with interpreter"
  mlir-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8.csv \
    -o ${NET}_quant_int8_per_tensor.mlir

  $EVAL_FUNC \
    --mlir_file=${NET}_quant_int8_per_tensor.mlir \
    --label_file=$LABEL_FILE \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then
  echo "Eval int8_rshift_only with interpreter"
  mlir-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8.csv \
    -o ${NET}_quant_int8_rshift_only.mlir

  $EVAL_FUNC \
    --mlir_file=${NET}_quant_int8_rshift_only.mlir \
    --label_file=$LABEL_FILE \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  echo "Eval int8_multiplier with interpreter"
  mlir-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8.csv \
    -o ${NET}_quant_int8_multiplier.mlir

  if [ "$EVAL_MODEL_TYPE" = "imagenet" ]; then
    $EVAL_FUNC \
      --mlir_file=${NET}_quant_int8_multiplier.mlir \
      --label_file=$LABEL_FILE \
      --dataset=$DATASET_PATH/imagenet/img_val_extracted \
      --net_input_dims $NET_INPUT_DIMS \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --std $STD \
      --input_scale $INPUT_SCALE \
      --model_type mlir \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --count=$1

  elif [ "$EVAL_MODEL_TYPE" = "widerface" ]; then
    _EVAL_FUNC=eval_retinaface_on_widerface.py

    #rm ${NET}_interpreter_result_int8 -rf
    if [ $DO_ACCURACY_FUSED_PREPROCESS -eq 1 ]; then
      echo "$0 DO_ACCURACY_FUSED_PREPROCESS under refactor yet, exit"
      exit 1
      $_EVAL_FUNC \
          --model ${NET}_quant_int8_multiplier.mlir \
          --net_input_dims $NET_INPUT_DIMS \
          --obj_threshold $OBJ_THRESHOLD \
          --nms_threshold $NMS_THRESHOLD \
          --images=$DATASET \
          --annotation=$ANNOTATION \
          --model_do_preprocess=True \
          --result=./${NET}_interpreter_result_int8 \
          --int8
    else
      $_EVAL_FUNC \
          --model ${NET}_quant_int8_multiplier.mlir \
          --net_input_dims $NET_INPUT_DIMS \
          --obj_threshold $OBJ_THRESHOLD \
          --nms_threshold $NMS_THRESHOLD \
          --images=$DATASET \
          --annotation=$ANNOTATION \
          --result=./${NET}_interpreter_result_int8 \
          --int8
    fi

  elif [ "$EVAL_MODEL_TYPE" = "lfw" ]; then
    _EVAL_FUNC=eval_arcface.py
    if [ $DO_ACCURACY_FUSED_PREPROCESS -eq 1 ];then
      echo "$0 DO_ACCURACY_FUSED_PREPROCESS under refactor yet, exit"
      exit 1
      $_EVAL_FUNC \
        --model=${NET}_quant_int8_multiplier.mlir \
        --dataset=$DATASET_PATH/lfw/lfw \
        --pairs=$DATASET_PATH/lfw/pairs.txt \
        --model_do_preprocess=True \
        --show=True
    else
      $_EVAL_FUNC \
        --model=${NET}_quant_int8_multiplier.mlir \
        --dataset=$DATASET_PATH/lfw/lfw \
        --pairs=$DATASET_PATH/lfw/pairs.txt \
        --show=True
    fi

  elif [ "$EVAL_MODEL_TYPE" = "coco" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_INT8
    if [ $DO_ACCURACY_FUSED_PREPROCESS -eq 1 ]; then
      echo "$0 DO_ACCURACY_FUSED_PREPROCESS under refactor yet, exit"
      exit 1
      $_EVAL_FUNC \
        --model=${NET}_quant_int8_multiplier.mlir \
        --net_input_dims ${NET_INPUT_DIMS} \
        --coco_image_path=$DATASET_PATH/coco/val2017/ \
        --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
        --coco_result_jason_file=./${NET}_coco_results_int8_multiplier.json \
        --model_do_preprocess=True \
        --count=$1
    else
      $_EVAL_FUNC \
        --model=${NET}_quant_int8_multiplier.mlir \
        --net_input_dims ${NET_INPUT_DIMS} \
        --coco_image_path=$DATASET_PATH/coco/val2017/ \
        --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
        --coco_result_jason_file=./${NET}_coco_results_int8_multiplier.json \
        --count=$1
    fi

  elif [ "$EVAL_MODEL_TYPE" = "voc2012" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_VOC
    $_EVAL_FUNC \
      --mlir=${NET}_quant_int8_multiplier.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --dataset=$DATASET_PATH/VOCdevkit

  elif [ "$EVAL_MODEL_TYPE" = "isbi" ]; then
    _EVAL_FUNC=eval_unet.py
    $_EVAL_FUNC \
      --mlir_file=${NET}_quant_int8_multiplier.mlir \
      --dataset=$DATASET_PATH/unet/ \
      --net_input_dims $NET_INPUT_DIMS \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --std $STD \
      --input_scale $INPUT_SCALE \
      --model_type mlir \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --count=$1

  else
    echo "Unknown EVAL_MODEL_TYPE $EVAL_MODEL_TYPE"
    exit 1
  fi
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  echo "Eval bf16 with interpreter"
  $EVAL_FUNC \
    --mlir_file=${NET}_quant_bf16.mlir \
    --label_file=$LABEL_FILE \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

if [ $INT8_MODEL -eq 1 ]; then
  echo "Eval int8 model to mlir with interpreter"
  $EVAL_FUNC \
    --mlir_file=${NET}_quant.mlir \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --std $STD \
    --input_scale $INPUT_SCALE \
    --model_type mlir \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --count=$1
fi

echo $0 DONE
