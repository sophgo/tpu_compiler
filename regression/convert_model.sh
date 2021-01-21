#!/bin/bash
set -e
set +x

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

usage()
{
  echo ""
  echo "Usage: convert_model.sh [-i model_file] [-d model_data] [-t model_type]"
  echo "                        [-b batch_size] [-q quant_cali_table] [-v chip_ver]"
  echo "                        [-o output_cvimodel] [-l do_layer_group] [-p do_fused_preprocess]"
  echo "                        [-r raw_scale] [-m mean] [-s std] [-a input_scale] [-w channel_order]"
  echo "                        [-z image_dims] [-c do_crop] [-f crop_offset]"
  echo -e "\t-i Model file (prototxt or onnx or tf)"
  echo -e "\t-d Model data file (caffe only, .caffemodel file)"
  echo -e "\t-t Model type (caffe|onnx|tf)"
  echo -e "\t-b Batch size                                  [default: 1]"
  echo -e "\t-q Quant calibration table file"
  echo -e "\t-v Chip version (cv183x|cv182x)                [default: cv183x]"
  echo -e "\t-o Output cvimodel file"
  echo -e "\t-l Do layergroup optimization                  [default: 1]"
  echo -e "\t-p Do fused preprocess                         [default: 0]"
  echo -e "\t-z Fused preprocess net input dims in h,w"
  echo -e "\t-r Fused preprocess raw scale                  [default: 255.0]"
  echo -e "\t-m Fused preprocess mean                       [default: 0.0,0.0,0.0]"
  echo -e "\t-s Fused preprocess std                        [default: 1.0,1.0,1.0]"
  echo -e "\t-a Fused preprocess input scale                [default: 1.0,1.0,1.0]"
  echo -e "\t-w Fused preprocess channel order (rgb|bgr)    [default: bgr]"
  echo -e "\t-y Fused preprocess image resize dims          [default: [none], use net dim]"
  echo -e "\t-f Fused preprocess crop offset                [default: [none], center]"
  echo -e "\t-g Do TPU Softmax inference                    [default: 0]"
  echo -e "\t-x Don't dequant results to fp32"
  echo -e "\t-n Quant full Op to BF16"
  echo -e "\t-h help"
  exit 1
}

bs="1"
do_layergroup="1"
do_fused_preprocess="0"
do_crop="0"
do_tpu_softmax="0"
chip_ver="cv183x"
dequant_results_to_fp32="true"
do_quant_full_bf16="0"

while getopts "i:d:t:b:q:v:o:l:pz:r:m:s:a:w:y:g:f:n:xu" opt
do
  case "$opt" in
    i ) model_def="$OPTARG" ;;
    d ) model_data="$OPTARG" ;;
    t ) model_type="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    q ) cali_table="$OPTARG" ;;
    v ) chip_ver="$OPTARG" ;;
    o ) output="$OPTARG" ;;
    l ) do_layergroup="$OPTARG" ;;
    p ) do_fused_preprocess="1" ;;
    z ) net_input_dims="$OPTARG" ;;
    r ) raw_scale="$OPTARG" ;;
    m ) mean="$OPTARG" ;;
    s ) std="$OPTARG" ;;
    a ) input_scale="$OPTARG" ;;
    w ) channel_order="$OPTARG" ;;
    y ) image_resize_dims="$OPTARG" ;;
    f ) crop_offset="$OPTARG" ;;
    g ) do_tpu_softmax="$OPTARG" ;;
    n ) do_quant_full_bf16="$OPTARG" ;;
    x ) dequant_results_to_fp32="false" ;;
    u ) usage ;;
  esac
done

fused_preprocess_opt=""
fused_preprocess_opt+="--net_input_dims ${net_input_dims} "
if [ ! -z "$raw_scale" ]; then
  fused_preprocess_opt+="--raw_scale ${raw_scale} "
fi
if [ ! -z "$mean" ]; then
  fused_preprocess_opt+="--mean ${mean} "
fi
if [ ! -z "$std" ]; then
  fused_preprocess_opt+="--std ${std} "
fi
if [ ! -z "$input_scale" ]; then
  fused_preprocess_opt+="--input_scale ${input_scale} "
fi
if [ ! -z "$raw_scale" ]; then
  fused_preprocess_opt+="--raw_scale ${raw_scale} "
fi
if [ ! -z "$channel_order" ]; then
  fused_preprocess_opt+="--model_channel_order ${channel_order} "
fi
if [ ! -z $image_resize_dims ]; then
  fused_preprocess_opt+="--image_resize_dims ${image_resize_dims} "
fi

set -x
name=$(basename "$model_def" | cut -d. -f1)
if [[ "$model_type" == "caffe" ]]; then
  cvi_model_convert.py \
      --model_path $model_def \
      --model_dat $model_data \
      --model_name $name \
      --model_type $model_type \
      --batch_size $bs \
      ${fused_preprocess_opt} \
      --mlir_file_path ${name}.mlir
elif [[ "$model_type" == "onnx" || "$model_type" == "tensorflow" ]]; then
  cvi_model_convert.py \
      --model_path $model_def \
      --model_name $name \
      --model_type $model_type \
      --batch_size $bs \
      ${fused_preprocess_opt} \
      --mlir_file_path ${name}.mlir
else
  echo "Invalid model_type $model_type"
  exit 1
fi

layergroup_opt=""
dce_opt=""
if [ $do_layergroup = "1" ]; then
  layergroup_opt="--group-ops "
  dce_opt="--dce "
fi

tpu_softmax_opt=""
if [ $do_tpu_softmax = "1" ]; then
  tpu_softmax_opt="--quant-bf16-softmax  "
fi

opt_mlir="${name}_int8.mlir"
opt_opt_info="op_info_int8.csv"
opt_cali="--import-calibration-table --calibration-table $cali_table"

tpu_quant_full_bf16=""
if [ $do_quant_full_bf16 = "1" ]; then
  tpu_quant_full_bf16="--quant-full-bf16  "
  opt_mlir="${name}_bf16.mlir"
  opt_opt_info="op_info_bf16.csv"
  opt_cali=
fi

tpuc-opt ${name}.mlir \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv | \
tpuc-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    ${opt_cali} \
    --assign-chip-name \
    --chipname $chip_ver \
    --tpu-quant \
    ${tpu_quant_full_bf16} \
    ${tpu_softmax_opt} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${opt_opt_info} \
    -o $opt_mlir

if [ $do_fused_preprocess = "1" ]; then
  tpuc-opt \
    --add-tpu-preprocess \
    --pixel_format BGR_PACKED \
    --input_aligned=false \
    $opt_mlir \
    -o "${name}_fused_preprocess.mlir"
    opt_mlir="${name}_fused_preprocess.mlir"
fi

${DIR}/mlir_to_cvimodel.sh \
    -i $opt_mlir \
    -o $output \
    --dequant-results-to-fp32=$dequant_results_to_fp32
