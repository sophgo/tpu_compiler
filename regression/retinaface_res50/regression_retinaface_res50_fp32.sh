set -e

export PATH=$PYTHON_TOOLS_PATH/model/retinaface:$PATH

CAFFE_MODEL_PATH=$MODEL_PATH/face_detection/retinaface_res50/caffe/fp32/2019.06.19
mlir-translate --caffe-to-mlir \
    $CAFFE_MODEL_PATH/R50-0000.prototxt \
    -debug \
    --debug-only=caffe-to-mlir \
    --caffemodel $CAFFE_MODEL_PATH/R50-0000.caffemodel \
    -o retinaface_res50.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info.csv \
    retinaface_res50.mlir \
    -o retinaface_res50_id.mlir

# extract blob from caffe
INPUT_IMG=/workspace/llvm-project/llvm/projects/mlir/externals/python_tools/data/faces/test.jpg
run_caffe_retinaface.py \
    --model_def $CAFFE_MODEL_PATH/R50-0000.prototxt \
    --pretrained_model $CAFFE_MODEL_PATH/R50-0000.caffemodel \
    --input_file $INPUT_IMG \
    --dump_blobs retinaface_res50_caffe_blobs.npz

npz_extract.py retinaface_res50_caffe_blobs.npz retinaface_res50_in_fp32.npz data
npz_extract.py retinaface_res50_caffe_blobs.npz retinaface_res50_out_fp32_caffe.npz face_rpn_cls_prob_reshape_stride8,face_rpn_bbox_pred_stride8,face_rpn_landmark_pred_stride8,face_rpn_cls_prob_reshape_stride16,face_rpn_bbox_pred_stride16,face_rpn_landmark_pred_stride16,face_rpn_cls_prob_reshape_stride32,face_rpn_bbox_pred_stride32,face_rpn_landmark_pred_stride32

# test mlir interpreter
mlir-tpu-interpreter retinaface_res50.mlir \
    --tensor-in retinaface_res50_in_fp32.npz \
    --tensor-out retinaface_res50_out_fp32.npz \
    --dump-all-tensor=retinaface_res50_tensor_all_fp32.npz

npz_compare.py retinaface_res50_out_fp32.npz retinaface_res50_out_fp32_caffe.npz -v
npz_compare.py \
    retinaface_res50_tensor_all_fp32.npz \
    retinaface_res50_caffe_blobs.npz \
    --op_info retinaface_res50_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vvv

# VERDICT
echo $0 PASSED