#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 fp32_layers_name_csv_file mlir_calied"
   exit 1
}

if [[ $# -ne 2 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

fp32_layers_name_csv_file=$1
mlir_calied=$2
layers_column_name=$(head -1 ${fp32_layers_name_csv_file} | tr "," "\n" | head -1)

${DIR}/../cvi_mix_precision.py \
    --all_layers_name_csv_file $1 \
    --layers_column_name $layers_column_name \
    --gen_cmd_script ${DIR}/gen_mix_precision.sh \
    --model ${mlir_calied}

