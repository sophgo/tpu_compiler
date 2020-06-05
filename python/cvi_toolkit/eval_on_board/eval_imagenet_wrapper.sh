#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# config
board_path=/tmp
board_ip=192.168.1.23
model_runner_path=/tmp/cvitek_tpu_sdk/bin/model_runner
script_path=$MLIR_SRC_PATH/python/cvi_toolkit/eval_on_board
eval_list="out_bf16.cvimodel,out_0.cvimodel,out_0_167.cvimodel,out_0_167_165.cvimodel,out_0_167_165_168.cvimodel,out_0_167_165_168_170.cvimodel,out_0_167_165_168_170_171.cvimodel"
user="root"
passwd="cvitek"
#eval_list="mobilenet0.25_int8_167.cvimodel,test.cvimodel"

for i in $(echo $eval_list | tr "," "\n"); do
    name=$i
    echo "do $name"
    sshpass -p $passwd scp $name $user@${board_ip}:/${board_path}
    ${script_path}/eval_imagenet_board.sh $name $board_path $board_ip $model_runner_path
done
