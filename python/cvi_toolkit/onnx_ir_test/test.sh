
test_cases=(
  "test_Sum"
)

TOOLKIT_PATH=${MLIR_SRC_PATH}/python/cvi_toolkit
#cp ${TOOLKIT_PATH}/transform/* ${INSTALL_PATH}/python/cvi_toolkit/transform/ -rf
mkdir -p out
pushd out
for test_case in ${test_cases}
do
    rm *.npz *.csv -rf
    python3 ${TOOLKIT_PATH}/onnx_ir_test/${test_case}.py
    python3 ${TOOLKIT_PATH}/cvi_model_convert.py \
        --model_path ${test_case}.onnx \
        --model_name ${test_case} \
        --model_type onnx \
        --mlir_file_path ${test_case}.mlir

    mlir-opt ${test_case}.mlir \
        --assign-layer-id \
        --convert-bn-to-scale \
        --canonicalize \
        --eltwise-early-stride \
        --print-tpu-op-info \
        --tpu-op-info-filename op_info.csv \
        -o ${test_case}_fp32.mlir

    mlir-tpu-interpreter ${test_case}_fp32.mlir \
        --tensor-in input.npz \
        --tensor-out output_fp32.npz

    #avoid different output name
    output_fp32_name=`cvi_npz_tool.py dump output_fp32.npz`
    cvi_npz_tool.py rename output_fp32.npz ${output_fp32_name} output
    cvi_npz_tool.py compare output.npz output_fp32.npz
done

popd