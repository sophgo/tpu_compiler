#!/bin/bash
#set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel name batch_size cali_table chip_name input_npz board_ip board_path model_runner_path"
   echo -ne "\nThe possible command as:\n"
cat <<EOF
   ./scripts/verify_host_board.sh  ./global_max_out.prototxt ./global_max_out.caffemodel liveness 9 ./bmnet_tune_calibration_table.threshold_table  cv1880v2 ../imx327_20200613_batch9_in.npz 192.168.1.13 /var/empty/ /var/empty/cvitek_tpu_sdk/bin/model_runner
EOF
   echo ""
   exit 1
}

if [[ $# -ne 10 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

export SET_CHIP_NAME=$6
#SET_CHIP_NAME=$6

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

OUTPUT_CVIMODEL=out.cvimodel
BATCH_SIZE=$4
INPUT=$7
BOARD_IP=$8
BOARD_PATH=$9
MODEL_RUNNER_PATH=${10}
OUTPUT_CVIMODEL_RESULT=out.npz
RUNNER_ARGS=" --model $OUTPUT_CVIMODEL --batch-num ${BATCH_SIZE} \
        --set-chip ${SET_CHIP_NAME} --output $OUTPUT_CVIMODEL_RESULT"
HOST_MD5=
IS_MATCH=0


caffe_lg()
{
    echo "try caffe lg strategy..."
    ${DIR}/../regression/convert_model_caffe_lg.sh $1 $2 $3 $4 $5 $OUTPUT_CVIMODEL > lg.log 2>&1
}

caffe_df()
{
    echo "try caffe df strategy..."
    ${DIR}/../regression/convert_model_caffe_df.sh $1 $2 $3 $4 $5 $OUTPUT_CVIMODEL > df.log 2>&1
}

gen_ref()
{
    # get fp32 inference result on host
    mlir-tpu-interpreter fp32.mlir --tensor-in $INPUT \
        --tensor-out fp32.out.npz \
        --dump-all-tensor tensor_all_fp32.npz

    # get int8 inference result on host
    mlir-tpu-interpreter int8.mlir --tensor-in $INPUT \
        --tensor-out int8.out.npz \
        --dump-all-tensor tensor_all_int8.npz

    # compare it, just show info
    cvi_npz_tool.py compare \
        tensor_all_fp32.npz \
        tensor_all_int8.npz \
        --op_info op_info_int8.csv \
        --dequant -v \
        --tolerance="0.5,0.5,0.5"

    # get result in host
    model_runner --input $INPUT $RUNNER_ARGS
    HOST_MD5=`md5sum $OUTPUT_CVIMODEL_RESULT`
}

compare_with_board()
{
    filename=`basename $INPUT`
    scp $INPUT $OUTPUT_CVIMODEL root@${BOARD_IP}:${BOARD_PATH}
    board_md5=`ssh -t root@${BOARD_IP} "cd ${BOARD_PATH} && ${MODEL_RUNNER_PATH} --input ${filename} $RUNNER_ARGS > /dev/null 2>&1 && md5sum $OUTPUT_CVIMODEL_RESULT"`
    board_md5=`echo $board_md5 | cut -d' ' -f1`
    HOST_MD5=`echo $HOST_MD5 | cut -d' ' -f1`
    if [ "$board_md5" != "$HOST_MD5" ]; then
        echo "host result ($HOST_MD5) not eq with board (${board_md5}), try dp strategy"
        IS_MATCH=0
    else
        echo "export $OUTPUT_CVIMODEL success"
        IS_MATCH=1
    fi
}

echo "remove *.cvimodel"
rm -rf $OUTPUT_CVIMODEL

# first using lg strategy
caffe_lg $1 $2 $3 $4 $5

# check convert is success
out_nr=`find . -name $OUTPUT_CVIMODEL | wc -l`
if [ $out_nr -eq 1 ]; then
    gen_ref
    compare_with_board
fi

if [ $IS_MATCH -eq 0 ]; then
    echo "convert fail, try caffe df strategy..."
    echo "remove *.cvimodel"
    rm -rf $OUTPUT_CVIMODEL

    # using df strategy
    caffe_df $1 $2 $3 $4 $5

    # check convert is success
    out_nr=`find . -name $OUTPUT_CVIMODEL | wc -l`
    if [ $out_nr -eq 1 ]; then
        gen_ref
        compare_with_board
    fi
fi

# VERDICT
if [ $IS_MATCH -eq 0 ]; then
    echo $0 FAILED
    exit 1
else
    echo $0 PASSED
fi
