from onnx import onnx, numpy_helper
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.model.mlir_model import MLIRModel
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime
import numpy as np

TEST_ONNX_IR = [
    "Neg",
    "Relu",
    "Sub",
    "Sum",
]

def onnx_inference(input, model_def, model_name):
    onnx.save(model_def, model_name)
    ort_session = onnxruntime.InferenceSession(model_name)
    ort_inputs = {'input': input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def onnx_convert_and_infernece(input_data, model_def, model_name):
    c = OnnxConverter(model_name, model_def, "{}.mlir".format(model_name))
    c.run()

    onnx_out = onnx_inference(input_data, model_def, model_name)

    m = MLIRModel()
    m.load_model("{}.mlir".format(model_name))
    mlir_out = m.inference(input_data)

    #print(mlir_out, onnx_out)
    try:
        np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)
        print("{} test PASS".format(model_name))
    except:
        print("{} test FAILD".format(model_name))




def test_Neg():
    test_case = 'test_Neg'
    input_shape = [4, 3, 27, 27]
    output_shape = [4, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    neg_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['output'],  # outputs
    )
    graph_def = helper.make_graph(
        [neg_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)


def test_Relu():
    print("hi")
    test_case = 'test_Relu'
    input_shape = [1, 3, 224, 224]
    node_def = helper.make_node(
        "Relu", # node name
        ['input'], # inputs
        ['y'], # outputs
    )

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        test_case,
        [input],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)


def test_Sub():
    test_case = 'test_Sub'
    input_shape = [4, 3, 27, 27]
    output_shape = [4, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    x1_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
    )

    sub_def = helper.make_node(
        'Sub',  # node name
        ['input', 'X1'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [x1_def, sub_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)

    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)

def test_Sum():

    test_case = 'test_Sum'
    input_shape = [4, 3, 27, 27]
    output_shape = [4, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    x1_def = helper.make_node(
        'Transpose',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
        perm=[0, 1, 2, 3]
    )

    #test only one input
    x2_def = helper.make_node(
        'Sum',  # node name
        ['input'],  # inputs
        ['X2'],  # outputs
    )

    #test three input
    sum_def = helper.make_node(
        'Sum',  # node name
        ['input', 'X1', 'X2'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [x1_def, x2_def, sum_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    onnx.checker.check_model(model_def)

    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)


if __name__ == "__main__":
    if "Neg" in TEST_ONNX_IR:
        test_Neg()
    if "Relu" in TEST_ONNX_IR:
        test_Relu()
    if "Sub" in TEST_ONNX_IR:
        test_Sub()
    if "Sum" in TEST_ONNX_IR:
        test_Sum()
