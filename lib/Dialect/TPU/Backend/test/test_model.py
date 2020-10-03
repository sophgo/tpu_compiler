import math
import numpy as np
from cvi_toolkit.transform import mlirimporter

# store weight data to npz
tensorNpz = dict()
tensorNpz['x_conv_w'] = np.ones((1, 1, 1, 1))
tensorNpz['y_conv_w'] = np.ones((1, 4096, 1, 1))
tensorNpz['fc1_w'] = np.ones((1, 256))
np.savez('model_weight.npz', **tensorNpz)

# setup input & output shapes of model
input_shapes = [[1,1,256,1], [1,4096,256,1]]
output_shapes = [[1, 4096]]
mlir = mlirimporter.MLIRImporter(input_shapes, output_shapes, input_type='FP32')
mlir.add_weight_file_op('model_weight.npz')

# create input op
input_x = mlir.add_input_op('input_x', 0)
input_y = mlir.add_input_op('input_y', 1)

# load weight
x_conv_w = mlir.add_load_file_op('x_conv_w', (1, 1, 1, 1))
conv_param = {
    'dilation_h': 1,
    'dilation_w': 1,
    'stride_h': 1,
    'stride_w': 1,
    'padding': 'VALID',
    'padding_t': 0,
    'padding_b': 0,
    'padding_l': 0,
    'padding_r': 0,
    'group': 1,
    'is_dw': False,
    'with_bias': False,
    'do_relu': False,
    'ins': [],
}
# create conv op
x1 = mlir.add_conv_op('x1', [input_x, x_conv_w], (1, 1, 256, 1), **conv_param)

y_conv_w = mlir.add_load_file_op('y_conv_w', (1, 4096, 1, 1))
conv_param = {
    'dilation_h': 1,
    'dilation_w': 1,
    'stride_h': 1,
    'stride_w': 1,
    'padding': 'VALID',
    'padding_t': 0,
    'padding_b': 0,
    'padding_l': 0,
    'padding_r': 0,
    'group': 4096,
    'is_dw': False,
    'with_bias': False,
    'do_relu': False,
    'ins': [],
}
y1 = mlir.add_conv_op('y1', [input_y, y_conv_w], (1, 4096, 256, 1), **conv_param)
sub = mlir.add_broadcast_sub_op('sub', [y1, x1], (1, 4096, 256, 1))
mul = mlir.add_square_op('mul', [sub], (1, 4096, 256, 1))
mul_reshape = mlir.add_reshape_op('mul_reshape', [mul], (4096,256))

fc1_w = mlir.add_load_file_op('fc1_w', (1, 256))
fc = mlir.add_fully_connected_op('fc', [mul_reshape, fc1_w], (4096, 1))
out = mlir.add_reshape_op('out', [fc], (1, 4096))
# create return op
mlir.add_return_op([out])

# print mlir
mlir_txt = mlir.print_module()
print(mlir_txt)
# save mlir
with open('model_fp32.mlir', 'w') as f:
    f.write(mlir_txt)