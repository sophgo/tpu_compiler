#!/bin/bash
set -e

# DIR="$( cd "$(dirname "$0")" ; pwd -P )"
MODEL_DO_PREPROCESS=0
MLIR_FILES=()
MLIR_TYPES=()
DEQUANTS=()

# assuming run after run regression_XXX.sh
if [ $2 = "pytorch" ]; then
  echo "Eval with pytorch dataloader on ${EVAL_MODEL_TYPE}"
  EVAL_FUNC=eval_classifier.py
elif [ $2 = "gluoncv" ]; then
  echo "Eval with gluoncv dataloader on ${EVAL_MODEL_TYPE}"
  EVAL_FUNC=eval_imagenet_gluoncv.py
else
  echo "invalid dataloader, choose [pytorch | gluoncv]"
  return 1
fi

if [ $DO_ACCURACY_FP32_INTERPRETER -eq 1 ]; then
  MLIR_FILES+=(${NET}_fp32.mlir)
  MLIR_TYPES+=("fp32")
  DEQUANTS+=(0)
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  MLIR_FILES+=(${NET}_quant_bf16.mlir)
  MLIR_TYPES+=("bf16")
  DEQUANTS+=(1)
fi

if [ $DO_QUANT_INT8 -eq 1 ]; then
  MLIR_FILES+=(${NET}_quant_int8.mlir)
  MLIR_TYPES+=("int8")
  DEQUANTS+=(1)
fi

if [ $DO_ACCURACY_FUSED_PREPROCESS -eq 1 ]; then
  echo "$0 DO_ACCURACY_FUSED_PREPROCESS under refactor yet, exit"
  exit 1
  # MODEL_DO_PREPROCESS=1
fi

for ((i=0; i<${#MLIR_FILES[@]}; i++))
do
  echo "Eval ${MLIR_TYPES[i]} with interpreter on ${EVAL_MODEL_TYPE}"
  if [ "$EVAL_MODEL_TYPE" = "imagenet" ]; then
    $EVAL_FUNC \
        --mlir_file=${MLIR_FILES[i]} \
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

  elif [ "$EVAL_MODEL_TYPE" = "widerface" ]; then
    _EVAL_FUNC=eval_retinaface_on_widerface.py
    rm ${NET}_interpreter_result_${MLIR_TYPES[i]} -rf
    $_EVAL_FUNC \
        --model ${MLIR_FILES[i]} \
        --net_input_dims $NET_INPUT_DIMS \
        --obj_threshold $OBJ_THRESHOLD \
        --nms_threshold $NMS_THRESHOLD \
        --images=$DATASET \
        --annotation=$ANNOTATION \
        --model_do_preprocess=${MODEL_DO_PREPROCESS} \
        --result=./${NET}_interpreter_result_${MLIR_TYPES[i]} \
        --dequant=${DEQUANTS[i]}

  elif [ "$EVAL_MODEL_TYPE" = "lfw" ]; then
    _EVAL_FUNC=eval_arcface.py
    $_EVAL_FUNC \
      --model=${MLIR_FILES[i]} \
      --dataset=$DATASET_PATH/lfw/lfw \
      --pairs=$DATASET_PATH/lfw/pairs.txt \
      --model_do_preprocess=${MODEL_DO_PREPROCESS} \
      --show=False

  elif [ "$EVAL_MODEL_TYPE" = "coco" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_INT8
    $_EVAL_FUNC \
      --model=${MLIR_FILES[i]} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --coco_image_path=$DATASET_PATH/coco/val2017/ \
      --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --coco_result_jason_file=./${NET}_coco_results_${MLIR_TYPES[i]}.json \
      --model_do_preprocess=${MODEL_DO_PREPROCESS} \
      --count=$1

  elif [ "$EVAL_MODEL_TYPE" = "voc2012" ]; then
    _EVAL_FUNC=$EVAL_SCRIPT_VOC
    $_EVAL_FUNC \
      --mlir=${MLIR_FILES[i]} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --dataset=$DATASET_PATH/VOCdevkit \
      --count=$1
  else
    echo "Unknown EVAL_MODEL_TYPE $EVAL_MODEL_TYPE"
    exit 1
  fi
done
echo $0 DONE
