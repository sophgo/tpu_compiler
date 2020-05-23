import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import onnxruntime

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
model_name = '{}.onnx'.format(test_case)
onnx.save(model_def, model_name)

input = np.random.rand(input_shape[0], input_shape[1],
                       input_shape[2], input_shape[3]).astype(np.float32)

ort_session = onnxruntime.InferenceSession(model_name)
ort_inputs = {'input': input}
ort_outs = ort_session.run(None, ort_inputs)

np.savez("input", input=input)
np.savez("output", output=ort_outs[0])
