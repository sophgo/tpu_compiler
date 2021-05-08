#!/bin/bash
set -e
set +x

function usage() {
  echo "Usage:"
  echo "  $0"
  echo -e "\t-i input_mlir_file (required)"
  echo -e "\t-o output_cvimodel (required)"
  echo -e "\t--dequant-results-to-fp32=true|false (option, default: true)"
  echo -e "\t--compress-weight=true|false (option, default: true)"
  echo -e "\t--append-weight=true|false (option, default: false)"
}

SHORT=hi:o:
LONG0=dequant-results-to-fp32:
LONG1=compress-weight:
LONG2=append-weight:

OPTS=$(getopt --options $SHORT --long $LONG0 --long $LONG1 --long $LONG2 --name "$0" -- "$@")
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
    --compress-weight )
      compress_weight="$2"
      shift 2
      ;;
    --append-weight )
      append_weight="$2"
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
if [ x"$dequant_to_fp32" == x ]; then
  dequant_to_fp32=true
fi
if [ x"$compress_weight" == x ]; then
  compress_weight=true
fi
compress_weight_opt=""
if [ $compress_weight = true ]; then
  compress_weight_opt="--compress-weight"
fi
if [ x"$append_weight" == x ]; then
  append_weight=false
fi

optimized_mlir="__lower_opt.mlir"
final_mlir="__final.mlir"

set -x
tpuc-opt $mlir_file \
    --tpu-lower \
    --dequant-results-to-fp32=$dequant_to_fp32 \
    --reorder-op \
    --eltwise-early-stride \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment \
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
    --compress-activation \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=_neuron_map.csv \
    --divide-ops-to-func \
    -o $final_mlir

tpuc-translate $final_mlir \
    --mlir-to-cvimodel \
    --weight-file _weight.bin \
    -o $out_cvimodel
