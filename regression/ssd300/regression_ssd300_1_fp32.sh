#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_INFERENCE_RESULT=0


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/ssd300/deploy_.prototxt \
    --caffemodel $MODEL_PATH/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300.mlir \
    -o ssd300_id.mlir


# test mlir interpreter
mlir-tpu-interpreter ssd300.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_fp32.npz \
    --dump-all-tensor=ssd300_tensor_all_fp32.npz
npz_compare.py ssd300_out_fp32.npz ssd300_out_fp32_ref.npz -v

# mark this for caffe conv layer merged with relu so conv layer comare fail.
# npz_compare.py \
#     ssd300_tensor_all_fp32.npz \
#     ssd300_blobs.npz \
#     --op_info ssd300_op_info.csv \
#     --tolerance=0.9999,0.9999,0.999 -vvv    

# opt1, fuse relu with conv
mlir-opt \
    --fuse-relu \
    ssd300_id.mlir \
    -o ssd300_opt1.mlir

# test opt1
mlir-tpu-interpreter ssd300_opt1.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_opt_out_fp32_opt1.npz
npz_compare.py ssd300_opt_out_fp32_opt1.npz ssd300_out_fp32_ref.npz -v


#opt2, convert priorbox to loadweight 
mlir-opt \
    --convert-priorbox-to-loadweight \
    ssd300_opt1.mlir \
    -o ssd300_opt2.mlir

# test opt2
mlir-tpu-interpreter ssd300_opt2.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_opt_out_fp32_opt2.npz
npz_compare.py ssd300_opt_out_fp32_opt2.npz ssd300_out_fp32_ref.npz -v

if [ $CHECK_INFERENCE_RESULT -eq 1 ]; then
run_mlir_detector_ssd.py \
      --model ssd300_opt2.mlir \
      --net_input_dims 300,300 \
      --dump_blobs ssd300_blobs.npz \
      --obj_threshold 0.5 \
      --dump_weights ssd300_weights.npz \
      --input_file $REGRESSION_PATH/ssd300/data/000000000724.jpg \
      --label_file $MODEL_PATH/caffe/ssd300/labelmap_coco.prototxt  \
      --draw_image ssd300_fp32_mlir_opt2_result.jpg
fi

# VERDICT
echo $0 PASSED
