import torch
import torch.onnx
from collections import namedtuple
from PIL import Image
from torchvision import transforms
import inspect
import numpy as np
import logging
from onnx import onnx, numpy_helper
from transform.onnx_converter import OnnxConverter
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-4s %(filename)2s %(lineno)d: %(message)s',
                    datefmt='%m-%d %H:%M')

# Create one input (ValueInfoProto)
n, c, h, w = 1, 2, 3, 4
INPUT = np.arange(n * c * h * w).astype(np.float32)

# make input, half part set to < 0 for test relu case
s = np.array_split(INPUT, 2)
s[0] *= -1
INPUT = np.concatenate(s)

# reshape for real input
INPUT = INPUT.reshape(n, c, h, w)
np.savez("test_in_fp32.npz", INPUT)

data = helper.make_tensor_value_info('data', TensorProto.FLOAT, list(INPUT.shape))
#value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, INPUT)
#value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, list(INPUT.shape))


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(INPUT.shape))

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Relu', # node name
    #['data', 'value'], # inputs
    ['data', 'value'], # inputs
    ['Y'], # outputs
    mode='constant', # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    #[data, value],
    [data],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-test')

c = OnnxConverter("test", model_def, "test.mlir")
c.run()
