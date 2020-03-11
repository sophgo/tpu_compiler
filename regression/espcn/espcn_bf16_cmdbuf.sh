# !/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    -debug \
    -debug-only=caffe-to-mlir,caffe-to-mlir_VERBOSE \
    --caffe-to-mlir $MODEL_PATH/caffe/espcn_2x.prototxt \
    --caffemodel $MODEL_PATH/caffe/espcn_2x.caffemodel \
    -o espcn.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --assign-layer-id \
    --merge-scale-into-conv \
    --fuse-relu \
    espcn.mlir \
    -o espcn-opt.mlir

################################
# prepare bf16 input
################################
bin_fp32_to_bf16.py \
    $DATA_PATH/test_cat_in_fp32.bin \
    in_bf16.bin


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $MODEL_PATH/caffe/espcn_2x_calibration_table.1x10 \
    espcn-opt.mlir \
    -o espcn-quant-cali.mlir

# scale by threshold x/ MUST place following import calibration table
mlir-opt \
    -debug \
    --gen-tanh-table \
    espcn-quant-cali.mlir \
    -o espcn-opt.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    espcn-opt.mlir \
    -o espcn-quant-bf16.mlir

# assign weight address & neuron address
mlir-opt \
    -debug \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    espcn-quant-bf16.mlir \
    -o espcn-quant-bf16-addr.mlir

# assign weight address & neuron address
mlir-translate \
    -debug \
    -debug-only=interpreter,mlir-to-cmdbuf,activation_kernel \
    --mlir-to-cmdbuf \
    espcn-quant-bf16-addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all.bin \
    101393408 0 101393408 1

# run interpreter, to generate reference tensor all npz
mlir-tpu-interpreter espcn-quant-bf16.mlir\
    -debug \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out-quant-bf16.bin \
    --dump-all-tensor=ref_tensor_all_quant-bf16.npz

# compare all tensors
bin_to_npz.py out_all.bin neuron_map_bf16.csv out_all_bf16.npz
npz_compare.py out_all_bf16.npz ref_tensor_all_quant-bf16.npz show 0

# VERDICT
echo $0 PASSED
