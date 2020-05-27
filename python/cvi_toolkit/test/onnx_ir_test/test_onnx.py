#!/usr/bin/env python3

from onnx import onnx, numpy_helper
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.model.mlir_model import MLIRModel
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime
import numpy as np
import os
import sys

TEST_ONNX_IR = [
    "AveragePool",
    "GlobalMaxPool",
    "LeakyRelu",
    "Max",
    "Min",
    "Neg",
    # "Relu",
    "Slice",
    "Sub",
    "Sum",
]

def export_test_data(tensors, model_name):
    # simple calibration table
    f = open("{}_cali_table".format(model_name), 'wt')
    for name in tensors:
        t = 1.1 * max(np.abs(tensors[name].flatten())) + 0.01
        f.write("{} {}\n".format(name,t))
    f.close
    # input and all tensor data
    np.savez("{}_onnx_all_fp32.npz".format(model_name), **tensors)
    np.savez("{}_input.npz".format(model_name), input=tensors['input'])

def _onnx_inference(input, model_name, input_name="input"):
    ort_session = onnxruntime.InferenceSession(model_name)
    ort_inputs = {input_name: input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def onnx_inference(input, model_def, model_name):
    onnx.save(model_def, model_name)
    return _onnx_inference(input, model_name)

def onnx_convert_and_infernece(input_data, model_def, model_name):
    c = OnnxConverter(model_name, model_def, "{}.mlir".format(model_name))
    c.run()

    onnx_out = onnx_inference(input_data, model_def, model_name)

    m = MLIRModel()
    m.load_model("{}.mlir".format(model_name))
    mlir_out = m.inference(input_data)
    export_test_data(m.get_all_tensor(), model_name)

    #print(mlir_out, onnx_out)
    np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)

def test_model(input_shape, model_path, input_name="input"):
    if isinstance(input_shape, list):
        input_shape = [int(x) for x in input_shape]
        input_shape = tuple(input_shape)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    model_name = model_path.split("/")[-1].split(".")[0]
    onnx_model = onnx.load(model_path)

    c = OnnxConverter(model_name, onnx_model, "{}.mlir".format(model_name))
    c.run()
    exit()
    onnx_out = _onnx_inference(input_data, model_path, input_name)

    m = MLIRModel()
    m.load_model("{}.mlir".format(model_name))
    mlir_out = m.inference(input_data)

    np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)
    print("PASS")

def test_AveragePool():
    test_case = 'test_AveragePool'
    input_data = np.random.randn(1, 3, 28, 28).astype(np.float32)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])
    node_def = onnx.helper.make_node(
        "AveragePool",
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[3, 3],
        strides=[1,1],
        pads=[2, 2, 2, 2],
        count_include_pad=1
    )
    graph_def = helper.make_graph(
        [node_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    onnx.checker.check_model(model_def)

    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_GlobalMaxPool():
    test_case = 'test_GlobalMaxPool'
    input_data = np.random.randn(1, 3, 28, 28).astype(np.float32)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 1])
    node_def = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=['input'],
        outputs=['output'],
    )
    graph_def = helper.make_graph(
        [node_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    onnx.checker.check_model(model_def)

    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_LeakyRelu():
    alpha = 0.01
    test_case = "test_LeakyRelu"
    input_shape = [1, 3, 224, 224]
    x1_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
    )
    node_def = helper.make_node(
        "LeakyRelu", # node name
        ['X1'], # inputs
        ['output'], # outputs
        alpha=alpha
    )

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [x1_def, node_def],
        test_case,
        [input],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.randn(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)

    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_Max():
    test_case = 'test_Max'
    input_shape = [1, 3, 27, 27]
    output_shape = [1, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    #test only one input
    x1_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
    )

    x2_def = helper.make_node(
        'Max',  # node name
        ['input'],  # inputs
        ['X2'],  # outputs
    )

    x3_def = helper.make_node(
        'Sum',  # node name
        ['input', 'X2'],  # inputs
        ['X3'],  # outputs
    )

    #test three input
    max_def = helper.make_node(
        'Max',  # node name
        ['input', 'X1', 'X2', 'X3'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [x1_def, x2_def, x3_def, max_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.randn(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_Min():
    test_case = 'test_Min'
    input_shape = [1, 3, 27, 27]
    output_shape = [1, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    #test only one input
    x1_def = helper.make_node(
        'Min',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
    )

    x2_def = helper.make_node(
        'Sum',  # node name
        ['input', 'X1'],  # inputs
        ['X2'],  # outputs
    )

    x3_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X3'],  # outputs
    )

    #test four input
    min_def = helper.make_node(
        'Min',  # node name
        ['input', 'X1', 'X2', 'X3'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [x1_def, x2_def, x3_def, min_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_Neg():
    test_case = 'test_Neg'
    input_shape = [1, 3, 27, 27]
    output_shape = [1, 3, 27, 27]

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

    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)


def test_Relu():
    test_case = 'test_Relu'
    input_shape = [1, 3, 224, 224]
    node_def = helper.make_node(
        "Relu", # node name
        ['input'], # inputs
        ['output'], # outputs
    )

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        test_case,
        [input],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name=test_case)
    input_data = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)

    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_Slice():
    test_case = 'test_Slice'
    x = np.random.randn(1, 20, 10, 5).astype(np.float32)
    input_shape = [1, 20, 10, 5]
    y = x[0:3, 0:10]
    output_shape = y.shape
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    start_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['starts'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.INT64,
            dims=starts.shape,
            vals=starts.flatten().astype(int),
        ),
    )
    ends_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['ends'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.INT64,
            dims=ends.shape,
            vals=ends.flatten().astype(int),
        ),
    )
    axes_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.INT64,
            dims=axes.shape,
            vals=axes.flatten().astype(int),
        ),
    )
    node_def = helper.make_node(
        'Slice',  # node name
        ['input', 'starts', 'ends', 'axes'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [start_node, ends_node, axes_node, node_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)

    input_data = np.random.rand(input_shape[0], input_shape[1],
                    input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)

def test_Sub():
    test_case = 'test_Sub'
    input_shape = [1, 3, 27, 27]
    output_shape = [1, 3, 27, 27]

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

    input_data = np.random.randn(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)
    onnx_convert_and_infernece(input_data, model_def, test_case)
    onnx.checker.check_model(model_def)

def test_Sum():
    test_case = 'test_Sum'
    input_shape = [1, 3, 27, 27]
    output_shape = [1, 3, 27, 27]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, output_shape)

    x1_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X1'],  # outputs
    )

    x2_def = helper.make_node(
        'Neg',  # node name
        ['input'],  # inputs
        ['X2'],  # outputs
    )

    #test only one input
    x3_def = helper.make_node(
        'Sum',  # node name
        ['input'],  # inputs
        ['X3'],  # outputs
    )

    #test three input
    sum_def = helper.make_node(
        'Sum',  # node name
        ['input', 'X1', 'X2', 'X3'],  # inputs
        ['output'],  # outputs
    )

    graph_def = helper.make_graph(
        [x1_def, x2_def, x3_def, sum_def],
        test_case,
        [input],
        [output],
    )
    model_def = helper.make_model(graph_def, producer_name=test_case)
    onnx.checker.check_model(model_def)

    input_data = np.random.randn(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)

    onnx.checker.check_model(model_def)
    onnx_convert_and_infernece(input_data, model_def, test_case)

def test_int8_cmdbuf(name):
    ret = os.system('../test_int8_cmdbuf.sh test_{}'.format(i))
    if ret != 0:
        raise Exception("test_{} int8 cmdbuf failed".format(name))

test_function = {
    "AveragePool": test_AveragePool,
    "LeakyRelu": test_LeakyRelu,
    "GlobalMaxPool": test_GlobalMaxPool,
    "LeakyRelu": test_LeakyRelu,
    "Max": test_Max,
    "Min": test_Min,
    "Neg": test_Neg,
    "Relu": test_Relu,
    "Slice": test_Slice,
    "Sub": test_Sub,
    "Sum": test_Sum,
}

if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")
    if len(sys.argv) >= 3:
        input_shape = sys.argv[1].split(",")
        if len(sys.argv) >= 4:
            test_model(input_shape, sys.argv[2], sys.argv[3])
        else:
            test_model(input_shape, sys.argv[2])
        exit(0)
    elif len(sys.argv) == 1:
        pass_list = list()
        fail_list = list()
        err_msg = list()
        for i in TEST_ONNX_IR:
            try:
                test_function.get(i)()
                test_int8_cmdbuf(i)
                pass_list.append(i)
            except Exception as err :
                fail_list.append(i)
                err_msg.append(str(err))
        print("{} PASS {}".format("="*4, "="*4))
        for i in pass_list:
            print(i)
        if len(fail_list) != 0:
            print("{} FAILD {}".format("="*4, "="*4))
            for i, msg in zip(fail_list, err_msg) :
                print(i)
                print("msg: ".format(msg))
            exit(-1)
    else:
        print("Usage: exe.py [input_shape] [model]")
        exit(-1)
