#!/bin/bash
set -e
set +x

function usage() {
  echo "Usage:"
  echo "  $0"
  echo -e "\t-i input_mlir_file (required)"
  echo -e "\t-o output_cvimodel (required)"
  echo -e "\t--dequant-results-to-file=true|fale (option, default: true)"
}

SHORT=hi:o:
LONG=dequant-results-to-fp32:

OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")
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

optimized_mlir="_lower_opt_$mlir_file"
final_mlir="_final_$mlir_file"

set -x
mlir-opt $mlir_file \
    --tpu-lower \
    --dequant-results-to-fp32=$dequant_to_fp32 \
    --reorder-op \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment \
    --group-ops \
    --dce \
    --deep-fusion-group-slice \
    --deep-fusion-opt \
    -o $optimized_mlir

mlir-opt $optimized_mlir \
    --tg-op-tile \
    --compress-activation \
    --compress-weight \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --tpu-generate-compressed-weight \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --divide-ops-to-func \
    -o $final_mlir

mlir-translate $final_mlir \
    --mlir-to-cvimodel \
    --weight-file weight.bin \
    -o $out_cvimodel
