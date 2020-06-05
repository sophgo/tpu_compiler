#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel batch_size cali_table"
   exit 1
}

if [[ $# -ne 4 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bf16_layers=",167,165,168,170,171,173,179,174,177"
# caffe threshold
# 0 pretend all int8
bf16_layers="0,167,165,168,170,171"

shuffix=""
bf16_layers_file="bf16_layers"

# all bf16
${DIR}/convert_model_caffe_bf16.sh "$@" out_bf16.cvimodel

cat <<EOF > ${bf16_layers_file}
EOF

for i in $(echo ${bf16_layers} | tr "," "\n" ); do
    cat <<EOF >> ${bf16_layers_file}
$i
EOF
    shuffix="${shuffix}_$i"
    ${DIR}/convert_model_caffe.sh "$@" out${shuffix}.cvimodel $bf16_layers_file
done
