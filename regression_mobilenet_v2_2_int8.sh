# !/bin/bash
set -e

export TPU_BASE_DIR=../..
export MLIR_BASE_DIR=$TPU_BASE_DIR/llvm-project/llvm/projects/mlir
export DATA_DIR=$MLIR_BASE_DIR/data

# translate from caffe
./bin/mlir-translate \
    --caffe-to-mlir /data/models/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel /data/models/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# apply all possible pre-calibration optimizations
./bin/mlir-opt \
    --convert-bn-to-scale \
    mobilenet_v2.mlir \
    -o mobilenet_v2-opt1.mlir
./bin/mlir-opt \
    --fold-scale \
    mobilenet_v2-opt1.mlir \
    -o mobilenet_v2-opt2.mlir
./bin/mlir-opt \
    --fuse-scale-into-conv \
    mobilenet_v2-opt2.mlir \
    -o mobilenet_v2-opt3.mlir

# import calibration table
./bin/mlir-opt \
    --import-calibration-table \
    --calibration-table $DATA_DIR/bmnet_mobilenet_v2_calibration_table.1x10 \
    mobilenet_v2-opt3.mlir \
    -o mobilenet_v2-cali.mlir

# apply all possible post-calibration optimizations
./bin/mlir-opt \
    --fuse-relu \
    mobilenet_v2-cali.mlir \
    -o mobilenet_v2-opt-post-cali.mlir

# quantization 1: per-layer int8
./bin/mlir-opt \
    --quant-int8 \
    mobilenet_v2-opt-post-cali.mlir \
    -o mobilenet_v2-quant-int8.mlir

./bin/mlir-tpu-interpreter mobilenet_v2-quant-int8.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc7 out_fc7.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc7.bin out_fc7_int8.bin 1 1 1 1000
# python ../llvm/projects/mlir/externals/python_tools/bin_dump.py out_fc7_int8.bin int8 1 1 1 1000 5
diff out_fc7_int8.bin $DATA_DIR/test_cat_out_fc7-int8.bin

# quantization 2: per-channel int8
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    mobilenet_v2-opt-post-cali.mlir \
    -o mobilenet_v2-quant-int8-per-channel.mlir

./bin/mlir-tpu-interpreter mobilenet_v2-quant-int8-per-channel.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-per-channel.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8-per-channel.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc7 out_fc7.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc7.bin out_fc7_int8.bin 1 1 1 1000
# python ../llvm/projects/mlir/externals/python_tools/bin_dump.py out_fc7_int8.bin int8 1 1 1 1000 5
diff out_fc7_int8.bin $DATA_DIR/test_cat_out_fc7-int8-per-channel.bin

# quantization 3: per-channel int8 with multiplier
./bin/mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    mobilenet_v2-opt-post-cali.mlir \
    -o mobilenet_v2-quant-int8-multiplier.mlir

./bin/mlir-tpu-interpreter mobilenet_v2-quant-int8-multiplier.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_DIR/test_cat_in_fp32.bin \
    --tensor-out out-quant-int8-multiplier.bin \
    --dump-all-tensor=tensor_all.npz
# python bin_compare.py out.bin out-quant-int8-multiplier.bin float32 1 1 1 1000 5
python ../llvm/projects/mlir/externals/python_tools/npz_to_bin.py \
    tensor_all.npz fc7 out_fc7.bin
python ../llvm/projects/mlir/externals/python_tools/bin_fp32_to_int8.py \
    out_fc7.bin out_fc7_int8.bin 1 1 1 1000
# python ../llvm/projects/mlir/externals/python_tools/bin_dump.py out_fc7_int8.bin int8 1 1 1 1000 5
diff out_fc7_int8.bin $DATA_DIR/test_cat_out_fc7-int8-multiplier.bin

# VERDICT
echo $0 PASSED
