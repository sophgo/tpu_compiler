#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

ctc_greedy_decoder.py \
    --npz_file $1 \
    --tensor $2 \
    --label_file $LABEL_MAP \
    --encoding gb18030

# VERDICT
echo $0 PASSED