import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import onnxruntime

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
model_name = '{}.onnx'.format(test_case)
onnx.save(model_def, model_name)

input = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)

ort_session = onnxruntime.InferenceSession(model_name)
ort_inputs = {'input': input}
ort_outs = ort_session.run(None, ort_inputs)

np.savez("input", input=input)
np.savez("output", output=ort_outs[0])
