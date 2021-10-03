#!/bin/bash
set -e
set +x

function usage() {
  echo "Usage:"
  echo "  $0"
  echo -e "\t-i input_mlir_file (required)"
  echo -e "\t-o output_cvimodel (required)"
  echo -e "\t--dequant-results-to-fp32=true|false (DEPRECATED, default: true)"
  echo -e "\t--results-type=fp32|[fp32/bf16/int8/keep] (option, default: fp32)"
  echo -e "\t--expose-bf16-inputs=true|false (option, default: false)"
  echo -e "\t--compress-weight=true|false (option, default: true)"
  echo -e "\t--append-weight=true|false (option, default: false)"
  echo -e "\t--compress-instruction=true|false (option, default: false)"
  echo -e "\t--tg-op-divide=true|false (option, default: false)"
  echo -e "\t--using-dmabuf=true|false (option, default: false)"
  echo -e "\t--model-version=version (option, default: latest, such as 1.2"
}

SHORT=hi:o:
LONG0=dequant-results-to-fp32:
LONG1=results-type:
LONG2=compress-weight:
LONG3=append-weight:
LONG4=compress-instruction:
LONG5=expose-bf16-inputs:
LONG6=tg-op-divide:
LONG7=model-version:

OPTS=$(getopt --options $SHORT --long $LONG0 --long $LONG1 --long $LONG2 --long $LONG3 --long $LONG4 --long $LONG5 --long $LONG6 --long $LONG7 --name "$0" -- "$@")
if [ $? != 0 ]; then
  echo "Failed to parse options...."
  exit 1
fi
eval set -- "$OPTS"

while true; do
  case "$1" in
    -h )
      usage
      exit 1
      ;;
    -i )
      mlir_file="$2"
      shift 2
      ;;
    -o )
      out_cvimodel="$2"
      shift 2
      ;;
    --dequant-results-to-fp32 )
      dequant_to_fp32="$2"
      shift 2
      ;;
    --results-type )
      results_type="$2"
      shift 2
      ;;
    --expose-bf16-inputs )
      expose_bf16_inputs="$2"
      shift 2
      ;;
    --compress-weight )
      compress_weight="$2"
      shift 2
      ;;
    --append-weight )
      append_weight="$2"
      shift 2
      ;;
    --compress-instruction )
      compress_instruction="$2"
      shift 2
      ;;
    --tg-op-divide )
      tg_op_divide="$2"
      shift 2
      ;;
    --model-version )
      model_version="$2"
      shift 2
      ;;
    --using-dmabuf )
      using_dmabuf="$2"
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    * )
      echo "Invalid Argumnets..."
      exit 1
      ;;
  esac
done

if [ x"$mlir_file" == x ]; then
  echo "missing option '-i'"
  exit 1
fi
if [ x"$out_cvimodel" == x ]; then
  echo "missing option '-o'"
  exit 1
fi

# DEPRECATED
if [ x"$dequant_to_fp32" == x"true" ]; then
  results_type="fp32"
fi
if [ x"$dequant_to_fp32" == x"false" ]; then
  results_type="keep"
fi

if [ x"$results_type" == x ]; then
  results_type="fp32"
fi
if [ x"$expose_bf16_inputs" == x ]; then
  expose_bf16_inputs=false
fi
if [ x"$compress_weight" == x ]; then
  compress_weight=true
fi
compress_weight_opt=""
if [ $compress_weight = true ]; then
  compress_weight_opt="--compress-weight"
fi
if [ x"$compress_instruction" == x ]; then
  compress_instruction=false
fi
compress_instruction_opt=""
if [ $compress_instruction = true ]; then
  compress_instruction_opt="-z"
fi
if [ x"$append_weight" == x ]; then
  append_weight=false
fi
tg_op_divide_opt=""
if [ x"$tg_op_divide" = x"true" ]; then
  tg_op_divide_opt="--tg-op-divide"
fi
using_dmabuf_opt=""
if [ x"$using_dmabuf" == x"true" ]; then
  using_dmabuf_opt="--using-dmabuf"
fi

version_opt="--model-version $model_version"

optimized_mlir="__lower_opt.mlir"
final_mlir="__final.mlir"

set -x
tpuc-opt $mlir_file \
    --tpu-lower \
    --results-type=$results_type \
    --expose-bf16-inputs=$expose_bf16_inputs \
    --reorder-op \
    --eltwise-early-stride \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment \
    $tg_op_divide_opt \
    --group-ops \
    --dce \
    --deep-fusion-group-slice \
    --deep-fusion-opt \
    -o $optimized_mlir

tpuc-opt $optimized_mlir \
    --tg-op-tile \
    $compress_weight_opt \
    --assign-weight-address \
    --tpu-append-weight=$append_weight \
    --tpu-weight-address-align=16 \
    --tpu-weight-bin-filename=_weight.bin \
    --tpu-weight-map-filename=_weight_map.csv \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=_neuron_map.csv \
    --divide-ops-to-func \
    -o $final_mlir

tpuc-translate $final_mlir \
    --mlir-to-cvimodel \
    --weight-file _weight.bin \
    ${version_opt} \
    ${compress_instruction_opt} \
    ${using_dmabuf_opt} \
    -o $out_cvimodel
