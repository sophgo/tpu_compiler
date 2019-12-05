# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# for run cmdbuf
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmkernel/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_support/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_bmbuilder/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPU_BASE_DIR/install_cmodel/lib:$LD_LIBRARY_PATH

# translate from caffe
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel /data/models/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# apply all possible pre-calibration optimizations
./bin/mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    --fuse-relu \
    mobilenet_v2.mlir \
    -o mobilenet_v2-opt.mlir

# fp32 inference
./bin/mlir-tpu-interpreter mobilenet_v2-opt.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz

################################
# prepare bf16 input
################################
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_bf16.py \
    $DATA_DIR/test_cat_in_fp32.bin \
    in_bf16.bin \
    0.017
# check
diff in_bf16.bin $DATA_DIR/test_cat_in_mobilenet_v2_bf16.bin

################################
# quantization
################################

# quantization
./bin/mlir-opt \
    --quant-bf16 \
    mobilenet_v2-opt.mlir \
    -o mobilenet_v2-quant-bf16.mlir

# bf16 inference
./bin/mlir-tpu-interpreter \
    mobilenet_v2-quant-bf16.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-bf16.bin \
    --dump-all-tensor=tensor_all_quant-bf16.npz
#python ../llvm/projects/mlir/externals/python_tools/bin_compare.py \
#    out.bin out-quant-bf16.bin float32 1 1 1 1000 5 5

# assign wieght address & neuron address
./bin/mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
  --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    mobilenet_v2-quant-bf16.mlir \
    -o mobilenet_v2-quant-bf16-addr.mlir

# backend translate into cmdbuf
./bin/mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2-quant-bf16-addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
~/work_cvitek/install_runtime/bin/test_bmnet \
    in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    out_all.bin \
    18811152 0 18811152 1

python ../llvm/projects/mlir/externals/python_tools/bin_extract.py \
    out_all.bin out_fc7_bf16.bin bf16 0x00049800 1000
diff out_fc7_bf16.bin $DATA_DIR/test_cat_out_fc7-bf16.bin

# compare all tensors
python ../llvm/projects/mlir/externals/python_tools/bin_to_npz.py \
    out_all.bin neuron_map_bf16.csv out_all_bf16.npz
python ../llvm/projects/mlir/externals/python_tools/npz_compare.py \
    out_all_bf16.npz tensor_all_quant-bf16.npz

# VERDICT
echo $0 PASSED
