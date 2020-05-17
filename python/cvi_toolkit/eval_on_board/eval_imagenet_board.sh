#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 file_name board_path board_ip model_runner_path [login_user] [login_pass]"
   exit 1
}

if [[ $# -lt 4 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
name=$1
board_path=$2
board_ip=$3
model_runner_path=$4
user=$5
passwd=$6
start_time="$(date -u +%s)"

${DIR}/eval_imagenet_pytorch.py \
    --model=${name} \
    --dataset=${DATASET_PATH}/imagenet/img_val_extracted \
    --mean=0.485,0.456,0.406 \
    --loader_transforms 1 \
    --input_scale=0.875 \
    --count=50000 \
    --board_path=${board_path} \
    --board_ip=${board_ip} \
    --model_runner_path=${model_runner_path} \
    --user=$user \
    --passwd=$passwd

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo -e "${name} in ${board_ip}:${board_path} Total of \033[1;32m$elapsed\033[0m seconds elapsed for process"

