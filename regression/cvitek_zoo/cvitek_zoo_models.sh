#!/bin/bash
set -e

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export EXCEPTS=-
export DO_CALIBRATION=0
export CALIBRATION_IMAGE_COUNT=1000
export MLIR_OPT_FE_PRE="--convert-bn-to-scale"
export MLIR_OPT_FE_POST="--eltwise-early-stride"
export MLIR_OPT_BE="--tg-fuse-leakyrelu --conv-ic-alignment"
export DO_QUANT_INT8=1
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_INT8_MULTIPLER=1
export DO_CMDBUF_INT8=1
export DO_QUANT_BF16=1
export DO_CMDBUF_BF16=1
export DO_DEEPFUSION=1
export DO_MEMOPT=1
export DO_LAYERGROUP=0
export DO_QUANT_MIX=0
export DO_ACCURACY_CAFFE=1
export DO_ACCURACY_INTERPRETER=1
export DO_E2E=1

if [ -z "$DO_BATCHSIZE" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=$DO_BATCHSIZE
fi
export BATCH_SIZE
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/generic/regression_0_caffe.sh

if [ $NET = "bmface_v3" ]; then
export MODEL_DEF=$MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.prototxt
export MODEL_DAT=$MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/cvitek_zoo/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/cvitek_zoo/data/cali_tables/bmface_v3_cali1024_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.6
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
fi

if [ $NET = "liveness" ]; then
export MODEL_DEF=$MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.prototxt
export MODEL_DAT=$MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.caffemodel
export FP32_INFERENCE_SCRIPT=$REGRESSION_PATH/cvitek_zoo/data/run_caffe/regression_${NET}_0_caffe.sh
export CALI_TABLE=$REGRESSION_PATH/cvitek_zoo/data/cali_tables/${NET}_calibration_table
export INPUT=data
export TOLERANCE_INT8_MULTIPLER=0.9,0.9,0.7
export DO_QUANT_INT8_PER_TENSOR=0
export DO_QUANT_INT8_RFHIFT_ONLY=0
export DO_QUANT_BF16=0
export DO_E2E=0
fi
