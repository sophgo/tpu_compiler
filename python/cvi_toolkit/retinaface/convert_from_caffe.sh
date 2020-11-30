/workspace/llvm-project/build/bin/tpuc-translate --caffe-to-mlir \
    $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt \
    -debug \
    --debug-only=caffe-to-mlir \
    --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel \
    -o retinaface_res50.mlir
